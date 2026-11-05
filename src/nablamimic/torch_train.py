"""Current NablaMimic recipe in Torch, using the same native NablaEngine.

Compilation covers the differentiable control body and functional optimizer
updates. Explicit CUDA graphs capture the complete window/backward, collector,
regression and critic work. No compile/capture error silently selects eager mode.
Random samples are explicit graph inputs and are generated outside captures.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import random
import signal
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import torch

from . import torch_nets as nets
from .torch_env import TorchTrackingEnv, select_reset


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def clone_tree(tree):
    return nets.tree_map(lambda x: x.detach().clone(), tree)


def as_numpy(tree):
    return nets.tree_map(lambda x: x.detach().cpu().numpy(), tree)


def state_where(done, reset, state):
    return select_reset(done, reset, state)


def make_control_body(env, gamma, max_episode_steps, act_clip, term_penalty, *, enc_hoist=True):
    """One differentiable control step; all objective accumulators stay tensors."""
    def body(actor, critic, cnorm, state, reset, acc, discount, total, reset_age,
             latent, reset_latents, action_noise, proprio_noise, goal_noise, delta, probe):
        history, goal = env.policy_obs(state, goal_noise)
        if enc_hoist:
            index = torch.clamp_min(reset_age, 0).to(torch.int64)
            reset_z = reset_latents[index, torch.arange(index.shape[0], device=index.device)]
            z = torch.where((reset_age >= 0)[:, None], reset_z, latent)
            policy_action = nets.actor_decode(actor, z, history, goal[:, -6:])
        else:
            policy_action = nets.actor_apply(actor, history, goal)
        policy_action = env.bound_action(policy_action)
        action = policy_action + action_noise
        if act_clip is not None:
            action = nets.clip_rows_to_median(action, act_clip, probe)
        action = action + delta
        stepped = env.step(state, action, proprio_noise)
        action = stepped['prev_a']
        reward, terms = env.reward(stepped, action, state['prev_a'], prev=state)
        if term_penalty is not None:
            weight, height_start, tilt_start = term_penalty
            hazard = env.hazard_features(stepped)
            zero = torch.zeros((), device=reward.device, dtype=reward.dtype)
            penalty = -weight * .02 * (
                (torch.maximum(env.cfg.term_height-hazard[:, 0]-height_start, zero) /
                 (env.cfg.term_height-height_start)) ** 2 +
                (torch.maximum(env.cfg.term_tilt-hazard[:, 2]-tilt_start, zero) /
                 (env.cfg.term_tilt-tilt_start)) ** 2)
            reward = reward + penalty
            terms = {**terms, 'term_pen': penalty}
        terminated = env.terminated(stepped)
        done = terminated | env.clip_ended(stepped) | (stepped['ep'] >= max_episode_steps)
        cobs = env.critic_obs(stepped)
        value = nets.critic_apply(critic, nets.normalize(cnorm, cobs))[:, 0]
        reward = torch.where(torch.isfinite(reward), reward, 0.0)
        value = torch.where(torch.isfinite(value), value, 0.0)
        next_acc = acc + discount * reward
        next_discount = discount * gamma
        total = total + torch.where(done, next_acc + next_discount * value * (~terminated), 0.0)
        next_state = state_where(done, reset, stepped)
        reset_age = torch.where(done, 0, torch.where(reset_age >= 0, reset_age + 1, -1))
        outputs = dict(cobs=cobs.detach(), cobs_pre=env.critic_obs(state).detach(),
                       r=reward.detach(), term=terminated, done=done, v2=value.detach(),
                       frame=stepped['frame'], terms={k: v.detach() for k, v in terms.items()},
                       hist=history.detach().to(torch.bfloat16), goal=goal.detach().to(torch.bfloat16),
                       a_pi=policy_action.detach())
        return (next_state, torch.where(done, 0.0, next_acc),
                torch.where(done, 1.0, next_discount), total, reset_age), outputs
    return body


def make_window(env, actor, critic, cnorm, horizon, gamma, body, encode):
    """Eager host orchestration around compiled body calls, captured as one graph."""
    def window(start, reset, action_noise, proprio_noise, goal_noise, delta, probe):
        continuous_latents, reset_latents = encode(actor, start['frame'], reset['frame'])
        n = start['frame'].shape[0]
        zero = torch.zeros(n, device=start['qpos'].device)
        carry = (start, zero, torch.ones_like(zero), zero,
                 torch.full((n,), -1, dtype=torch.int32, device=zero.device))
        records = []
        for t in range(horizon):
            carry, row = body(actor, critic, cnorm, carry[0], reset, *carry[1:],
                              continuous_latents[t], reset_latents,
                              action_noise[t], torch.where((carry[4] >= 0)[:, None],
                                  proprio_noise[1, torch.clamp_min(carry[4], 0).to(torch.int64),
                                                torch.arange(n, device=zero.device)],
                                  proprio_noise[0, t]), goal_noise[t], delta[t], probe)
            records.append(row)
        state, acc, discount, total, _ = carry
        last_value = nets.critic_apply(critic, nets.normalize(cnorm, env.critic_obs(state)))[:, 0]
        objective = -(total + acc + discount * torch.where(torch.isfinite(last_value), last_value, 0.0)).mean()
        gradients, clip_statistics = torch.autograd.grad(objective, (delta, probe))
        outputs = nets.tree_map(lambda *rows: torch.stack(rows), *records)
        return objective.detach(), gradients.detach(), clip_statistics.detach()/horizon, outputs
    return window


class ResetSampler:
    """SuperMimic bin mixture, with explicit Torch random streams."""
    def __init__(self, env, max_episode_steps, gamma, uniform_over='bins'):
        self.env = env
        self.bin_index = torch.arange(env.n_frames, device=env.device, dtype=torch.int64) // 50
        self.n_bins = (env.n_frames + 49) // 50
        self.valid = env.valid.to(torch.float32)
        self.bin_count = torch.zeros(self.n_bins, device=env.device).scatter_add_(0, self.bin_index, self.valid)
        nonempty = (self.bin_count > 0).to(torch.float32)
        self.bin_q = torch.stack((torch.ones_like(nonempty), nonempty/nonempty.sum()))
        self.g_max = env.max_reward * (1-gamma**max_episode_steps)/(1-gamma)
        self.uniform_over = uniform_over
        self.probs = self.frame_probabilities()

    def bin_weights(self, stat):
        weights = torch.where(self.bin_count > 0,
                              torch.softmax(torch.where(self.bin_count > 0, -25*stat, -1e9), dim=0), 0.0)
        return weights / weights.sum()

    def frame_probabilities(self):
        def per_frame(weights):
            return torch.where(self.valid > 0, weights[self.bin_index] /
                               torch.clamp_min(self.bin_count[self.bin_index], 1), 0.0)
        if self.uniform_over == 'bins':
            nonempty = (self.bin_count > 0).to(torch.float32)
            uniform = per_frame(nonempty/nonempty.sum())
        else:
            uniform = self.valid/self.valid.sum()
        return .5*uniform + .4*per_frame(self.bin_weights(self.bin_q[0])) + .1*per_frame(self.bin_q[1]/self.bin_q[1].sum())

    def sample(self, count):
        with torch.no_grad():
            frames = torch.multinomial(self.probs, count, replacement=True).to(torch.int32)
            return self.env.reset(frames, self.env.sample_reset_noise(count))

    def update(self, outs, targets, cnorm):
        normalizer = nets.update_normalizer(cnorm, outs['cobs'].reshape(-1, self.env.critic_dim))
        bins = self.bin_index[outs['frame'].reshape(-1).to(torch.int64)]
        sums = torch.zeros_like(self.bin_count).scatter_add(0, bins, (targets/self.g_max).reshape(-1))
        counts = torch.zeros_like(self.bin_count).scatter_add(0, bins, torch.ones_like(targets).reshape(-1))
        stat = torch.where(counts > 0, .9*self.bin_q[0] + .1*sums/torch.clamp_min(counts, 1), self.bin_q[0])
        past = .995*self.bin_q[1] + .005*self.bin_weights(stat)
        nets.copy_tree_(cnorm, normalizer)
        self.bin_q.copy_(torch.stack((stat, past)))
        self.probs.copy_(self.frame_probabilities())


class TorchTrainer:
    """Fixed current-paper workload; graph-owned storage is never replaced."""
    def __init__(self, env, actor, critic, args):
        from .torch_execution import compile_fullgraph, CudaGraphCall
        self.env, self.actor, self.critic, self.args = env, actor, critic, args
        self.device = env.device
        self.n, self.nc, self.h = args.num_envs, args.collect_envs or args.num_envs, args.horizon
        self.m = self.n*self.h
        self.actor_params = nets.actor_trainable(actor)
        self.actor_opt, self.critic_opt = nets.adam_init(self.actor_params), nets.adam_init(critic)
        self.ema, self.ema_count = nets.init_policy_ema(actor)
        self.cnorm = (torch.zeros(env.critic_dim, device=self.device), torch.ones(env.critic_dim, device=self.device))
        self.sampler = ResetSampler(env, args.max_episode_steps, args.gamma, args.uniform_over)
        self.collector = self.sampler.sample(self.nc)
        self.collector['ep'].copy_(torch.randint(args.max_episode_steps, (self.nc,), device=self.device, dtype=torch.int32))
        self.replay_size = args.replay_steps*self.nc
        compact = env.compact(self.collector)
        self.replay = {k: torch.empty((self.replay_size, *v.shape[1:]), dtype=v.dtype, device=v.device)
                       for k, v in compact.items()}
        self.replay_ptr = torch.tensor([self.nc % self.replay_size, self.nc], device=self.device, dtype=torch.int64)
        self.outer_step = torch.zeros((), device=self.device, dtype=torch.int32)
        self.total_collector_steps = 0
        self.completed_updates = 0
        self.col_accumulator = 0.0
        self.graphs = {}
        self.setup_record = {}
        self._initial = dict(actor=clone_tree(actor), critic=clone_tree(critic), collector=clone_tree(self.collector),
                             cnorm=clone_tree(self.cnorm), bin_q=self.sampler.bin_q.clone(),
                             rng=torch.cuda.get_rng_state(), cpu_rng=torch.get_rng_state())
        self.reset_setup_state()
        # The pinned Engine lazily constructs its autograd class on the first
        # differentiable step. Register it through the public API before Dynamo
        # traces the control body; one forward needs no backward or RNG draw.
        with torch.enable_grad():
            env.step({key: value[:1] for key, value in self.collector.items()},
                     torch.zeros((1, env.nu), device=self.device, requires_grad=True),
                     torch.zeros((1, 93), device=self.device))

        def encode(actor_, frames, reset_frames):
            offsets = torch.arange(self.h, device=self.device)[:, None]
            def sequence(frame):
                f = torch.minimum(frame[None]+offsets, env.ref_last[frame][None])
                return nets.actor_encode(actor_, env.goal_ref(f.reshape(-1))).reshape(self.h, self.n, -1)
            return sequence(frames), sequence(reset_frames)

        body = make_control_body(env, args.gamma, args.max_episode_steps, args.act_cot_clip,
                                 tuple(args.term_penalty), enc_hoist=True)
        self.body = compile_fullgraph(body) if args.execution != 'eager' else body
        self.encode = compile_fullgraph(encode) if args.execution != 'eager' else encode
        self.window = make_window(env, actor, critic, self.cnorm, self.h, args.gamma, self.body, self.encode)
        self.target_fn = lambda a, g: nets.action_targets(
            a, g, env.action_scale, args.action_target[2], args.act_cot_clip,
            outer_step=self.outer_step, decay_steps=args.lr_decay_steps,
            delta_floor=args.action_target_delta_floor)
        self.target_fn = compile_fullgraph(self.target_fn) if args.execution != 'eager' else self.target_fn
        self.td = lambda r, term, done, value: nets.td_lambda(r, term, done, value, args.gamma, args.lam)
        self.td = compile_fullgraph(self.td) if args.execution != 'eager' else self.td

        start, reset, action_noise, proprio, goal, delta, probe = self.window_inputs()
        setup_started = time.monotonic()
        def progress(stage):
            write_json(Path(args.out)/'setup-progress.json', dict(
                stage=stage, elapsed_s=time.monotonic()-setup_started,
                execution=args.execution, graphs={key: value.record for key, value in self.graphs.items()}))
            print('Torch setup: '+stage, flush=True)
        progress('window')
        if args.execution == 'compile-graph':
            self.graphs['window'] = CudaGraphCall(self.window, (start, reset, action_noise, proprio, goal, delta, probe),
                                                 restore=self.reset_setup_state, compile_function=False)
            result = self.graphs['window'](start, reset, action_noise, proprio, goal, delta, probe)
        else:
            result = self.window(start, reset, action_noise, proprio, goal, delta, probe)
        self.window_outputs = result
        self.training_data = result[3]
        progress('action and critic targets')
        targets = self.target_fn(self.training_data['a_pi'], result[1])
        self.target_data = clone_tree(targets)
        self.critic_targets = self.td(self.training_data['r'], self.training_data['term'], self.training_data['done'], self.training_data['v2']).detach()
        # A captured update reads these stable buffers; subsequent windows copy in-place.
        self.reg_hist = self.training_data['hist'].reshape(self.m, env.hist_dim)
        self.reg_goal = self.training_data['goal'].reshape(self.m, env.goal_dim)
        self.reg_targets = self.target_data['targets'].reshape(self.m, env.nu)
        self.reg_weights = self.target_data['weights'].reshape(self.m)
        self.critic_inputs = self.training_data['cobs_pre'].reshape(self.m, env.critic_dim)
        self.critic_y = self.critic_targets.reshape(self.m)
        self._build_update_functions(compile_fullgraph)
        sample_indices = torch.randint(self.m, (args.regression_batch,), device=self.device)
        critic_indices = torch.randperm(self.m, device=self.device).reshape(args.critic_minibatches, -1)[0]
        for name, fn, inputs in (
            ('regression', self.regression, (sample_indices,)),
            ('critic', self.critic_update, (critic_indices,)),
            ('collector', self.collect, self.collector_inputs()),
            ('statistics', self.statistics, ())):
            progress(name)
            if args.execution == 'compile-graph':
                self.graphs[name] = CudaGraphCall(fn, inputs, restore=self.reset_setup_state, compile_function=False)
            else:
                fn(*inputs)
        self.reset_setup_state()
        torch.cuda.synchronize()
        self.setup_record = dict(seconds=time.monotonic()-setup_started, execution=args.execution,
                                 compilation_scope='per-control differentiable body, encoder, targets, TD, optimizers, collector transition, statistics; collector commits are separate captured kernels',
                                 graph_scope=f'complete {self.h}-step forward/backward and separate reusable update/collector/statistics graphs',
                                 graphs={name: graph.record for name, graph in self.graphs.items()},
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
        progress('ready')

    def reset_setup_state(self):
        """Undo all warmup/capture mutations; no simulated setup enters training."""
        with torch.no_grad():
            nets.copy_tree_(self.actor, self._initial['actor'])
            nets.copy_tree_(self.critic, self._initial['critic'])
            nets.copy_tree_(self.ema, self._initial['actor'])
            for optimizer in (self.actor_opt, self.critic_opt):
                nets.tree_map(lambda x: x.zero_(), optimizer)
            self.ema_count.zero_(); self.outer_step.zero_()
            nets.copy_tree_(self.cnorm, self._initial['cnorm'])
            self.sampler.bin_q.copy_(self._initial['bin_q'])
            self.sampler.probs.copy_(self.sampler.frame_probabilities())
            nets.copy_tree_(self.collector, self._initial['collector'])
            for key, value in self.env.compact(self.collector).items():
                self.replay[key].reshape(self.args.replay_steps, self.nc, *value.shape[1:]).copy_(value.unsqueeze(0))
            self.replay_ptr.copy_(torch.tensor([self.nc % self.replay_size, self.nc], device=self.device))
        torch.cuda.set_rng_state(self._initial['rng'])
        torch.set_rng_state(self._initial['cpu_rng'])

    def window_inputs(self):
        filled = min(self.nc*(1+self.total_collector_steps), self.replay_size)
        indices = torch.randint(filled, (self.n,), device=self.device)
        with torch.no_grad():
            start = self.env.restore({k: v.index_select(0, indices) for k, v in self.replay.items()})
            reset = self.sampler.sample(self.n)
            action_noise = self.args.act_noise*torch.randn((self.h, self.n, self.env.nu), device=self.device)
            noise = self.env.sample_observation_noise(self.h*self.n)
            reset_proprio = 2*torch.rand((self.h, self.n, 93), device=self.device)-1
        delta = torch.zeros_like(action_noise, requires_grad=True)
        probe = torch.zeros(2, device=self.device, requires_grad=True)
        proprio = torch.stack((noise['proprio'].reshape(self.h, self.n, -1), reset_proprio))
        return start, reset, action_noise, proprio, noise['goal'].reshape(self.h, self.n, -1), delta, probe

    def collector_inputs(self):
        noise = self.env.sample_observation_noise(self.nc)
        return self.sampler.sample(self.nc), self.args.collect_noise*torch.randn((self.nc, self.env.nu), device=self.device), noise['proprio'], noise['goal']

    def _build_update_functions(self, compile_fullgraph):
        args, env = self.args, self.env
        def regression(indices):
            h = self.reg_hist[indices].to(torch.float32)
            g = self.reg_goal[indices].to(torch.float32)
            y, w = self.reg_targets[indices], self.reg_weights[indices]
            def loss(parameters):
                actor = nets.with_actor_parameters(self.actor, parameters)
                return nets.weighted_action_mse(nets.actor_apply(actor, h, g), y, w, env.action_scale)
            gradients, value = torch.func.grad_and_value(loss)(self.actor_params)
            lr = nets.lr_schedule(self.actor_opt['count'], args.action_target_lr,
                warmup_steps=args.lr_warmup_steps, decay_steps=args.lr_decay_steps,
                decay_to=args.lr_decay_to, steps_per_update=int(args.action_target[0]))
            params, optimizer, norm = nets.adam_step(self.actor_params, gradients, self.actor_opt, lr)
            nets.copy_tree_(self.actor_params, params); nets.copy_tree_(self.actor_opt, optimizer)
            return value, norm, lr
        def critic_update(indices):
            x = nets.normalize(self.cnorm, self.critic_inputs[indices])
            y = self.critic_y[indices]
            def loss(parameters):
                prediction = nets.critic_apply(parameters, x)[:, 0]
                return ((prediction-y)**2).mean()
            gradients, value = torch.func.grad_and_value(loss)(self.critic)
            lr = nets.lr_schedule(self.critic_opt['count'], args.lr,
                warmup_steps=args.lr_warmup_steps, decay_steps=args.lr_decay_steps,
                decay_to=args.lr_decay_to, steps_per_update=args.critic_epochs*args.critic_minibatches)
            params, optimizer, norm = nets.adam_step(self.critic, gradients, self.critic_opt, lr,
                                                    b2=.999, element_clip=None)
            nets.copy_tree_(self.critic, params); nets.copy_tree_(self.critic_opt, optimizer)
            return value, norm, lr
        def collector_transition(collector, reset, action_noise, proprio_noise, goal_noise):
            with torch.no_grad():
                history, goal = env.policy_obs(collector, goal_noise)
                action = nets.actor_apply(self.actor, history, goal)+action_noise
                stepped = env.step(collector, action, proprio_noise)
                term = env.terminated(stepped)
                done = term | env.clip_ended(stepped) | (stepped['ep'] >= args.max_episode_steps)
                return state_where(done, reset, stepped), term.to(torch.float32).mean()

        self.collector_transition = (compile_fullgraph(collector_transition)
                                     if args.execution != 'eager' else collector_transition)

        def collect(reset, action_noise, proprio_noise, goal_noise):
            # Materialize the complete next state before mutating collector or
            # replay storage. Compiling transition and copy-back together lets
            # Inductor fuse the shifted history read with its overlapping write,
            # producing a cross-block race. This boundary preserves the copy
            # contract; CUDA capture still records all commit kernels below.
            with torch.no_grad():
                state, termination_rate = self.collector_transition(
                    self.collector, reset, action_noise, proprio_noise, goal_noise)
                position = (self.replay_ptr[0] + torch.arange(self.nc, device=self.device)) % self.replay_size
                for key, value in env.compact(state).items():
                    self.replay[key].index_copy_(0, position, value)
                nets.copy_tree_(self.collector, state)
                self.replay_ptr.copy_(torch.stack(((self.replay_ptr[0]+self.nc) % self.replay_size,
                                                    torch.clamp_max(self.replay_ptr[1]+self.nc, self.replay_size))))
                return termination_rate
        def statistics():
            with torch.no_grad():
                self.sampler.update(self.training_data, self.critic_targets, self.cnorm)
                average, count = nets.update_policy_ema(self.ema, self.actor, self.ema_count)
                nets.copy_tree_(self.ema, average); self.ema_count.copy_(count)
                self.outer_step.add_(1)
                return self.ema_count.clone()
        self.collect = collect
        for name, function in dict(regression=regression, critic_update=critic_update,
                                   statistics=statistics).items():
            setattr(self, name, compile_fullgraph(function) if args.execution != 'eager' else function)

    def update(self):
        started = time.monotonic()
        self.col_accumulator += 1/self.args.utd
        collector_steps = int(self.col_accumulator)
        self.col_accumulator -= collector_steps
        collector_term = []
        for _ in range(collector_steps):
            function = self.graphs.get('collector', self.collect)
            collector_term.append(function(*self.collector_inputs()).clone())
            self.total_collector_steps += 1
        result = self.graphs.get('window', self.window)(*self.window_inputs())
        if self.args.execution != 'compile-graph':
            nets.copy_tree_(self.training_data, result[3])
        target = self.target_fn(self.training_data['a_pi'], result[1])
        nets.copy_tree_(self.target_data, target)
        targets = self.td(self.training_data['r'], self.training_data['term'], self.training_data['done'], self.training_data['v2'])
        self.critic_targets.copy_(targets)
        actor_losses, critic_losses = [], []
        for _ in range(int(self.args.action_target[0])):
            indices = torch.randint(self.m, (self.args.regression_batch,), device=self.device)
            actor_losses.append(self.graphs.get('regression', self.regression)(indices)[0].clone())
        # EMA depends only on actor updates, so its statistics-call placement after
        # critic fitting is numerically identical and never feeds training weights.
        for _ in range(self.args.critic_epochs):
            permutation = torch.randperm(self.m, device=self.device).reshape(self.args.critic_minibatches, -1)
            for indices in permutation:
                critic_losses.append(self.graphs.get('critic', self.critic_update)(indices)[0].clone())
        self.graphs.get('statistics', self.statistics)()
        self.completed_updates += 1
        torch.cuda.synchronize()
        from nablaengine import torch_ffi
        diagnostics = torch_ffi.read_diagnostics(reset=True)
        invalid = {key: int(value) for key, value in diagnostics.items() if key in (
            'fwd_unsupported_worlds', 'fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds',
            'nonfinite_rows', 'nonfinite_cotangent_rows', 'replay_mismatch_worlds') and value}
        metrics = dict(completed_updates=self.completed_updates, iteration_s=time.monotonic()-started,
            actor_loss=float(result[0]), regression_loss=float(torch.stack(actor_losses).mean()),
            critic_loss=float(torch.stack(critic_losses).mean()),
            reward_mean=float(self.training_data['r'].mean()/self.env.max_reward),
            termination_rate=float(self.training_data['term'].float().mean()),
            collector_termination_rate=float(torch.stack(collector_term).mean()),
            ema_updates=int(self.ema_count), actor_optimizer_steps=int(self.actor_opt['count']),
            critic_optimizer_steps=int(self.critic_opt['count']), engine_diagnostics=diagnostics,
            clip_stats=result[2].detach().cpu().tolist(), target_stats=self.target_data['at_clip'].cpu().tolist(),
            memory=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        def numerical_range(value):
            value = value.detach()
            def json_number(number):
                number = float(number)
                return number if math.isfinite(number) else str(number)
            return dict(minimum=json_number(value.amin()), maximum=json_number(value.amax()),
                        absolute_max=json_number(value.abs().amax()),
                        nonfinite=int((~torch.isfinite(value)).sum()))
        metrics['numerical_ranges'] = dict(
            reward_terms={name: numerical_range(value) for name, value in self.training_data['terms'].items()},
            critic_observation=numerical_range(self.training_data['cobs_pre']),
            policy_history=numerical_range(self.training_data['hist']),
            policy_action=numerical_range(self.training_data['a_pi']),
            action_target=numerical_range(self.target_data['targets']),
            action_cotangent=numerical_range(result[1]),
            target_setpoint_change=numerical_range(
                (self.target_data['targets']-self.training_data['a_pi'])*self.env.action_scale),
            critic_target=numerical_range(self.critic_targets),
            cnorm_mean=numerical_range(self.cnorm[0]),
            cnorm_std=numerical_range(torch.sqrt(self.cnorm[1]+1.e-5)),
            collector_qpos=numerical_range(self.collector['qpos']),
            collector_qvel=numerical_range(self.collector['qvel']))
        if self.completed_updates != metrics['ema_updates'] or metrics['actor_optimizer_steps'] != self.completed_updates*int(self.args.action_target[0]) or metrics['critic_optimizer_steps'] != self.completed_updates*self.args.critic_epochs*self.args.critic_minibatches:
            raise RuntimeError('Training/EMA/optimizer counters disagree')
        if invalid or not all(math.isfinite(metrics[key]) for key in ('actor_loss','regression_loss','critic_loss','reward_mean')):
            raise RuntimeError('Invalid training condition: '+json.dumps(dict(invalid=invalid,metrics=metrics)))
        return metrics

    def restore(self, path):
        """Restore current-format full state into its allocated storage owners."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state['format'] != 'nablamimic_torch_training_v1':
            raise ValueError('Current Torch full training state required')
        if state['perception'] != self.env.perception_contract:
            raise ValueError('Resume physical perception contract differs')
        if state['observation'] != self.env.observation_contract:
            raise ValueError('Resume policy observation contract differs')
        controls = {'out', 'config', 'resume', 'iters', 'hours', 'save_wall_minutes',
                    'ckpt_every', 'full_state_at_end', 'initial_checkpoint'}
        current, saved = vars(self.args), state['resolved_config']
        for key in set(current) | set(saved):
            if key not in controls and current.get(key) != saved.get(key):
                raise ValueError(f'Resume training input differs for {key}')
        owners = dict(actor=self.actor, critic=self.critic, ema=self.ema, ema_count=self.ema_count,
            actor_optimizer=self.actor_opt, critic_optimizer=self.critic_opt, cnorm=self.cnorm,
            bin_q=self.sampler.bin_q, probabilities=self.sampler.probs, collector=self.collector,
            replay=self.replay, replay_ptr=self.replay_ptr, outer_step=self.outer_step)
        with torch.no_grad():
            for key, owner in owners.items():
                def restore_tensor(destination, source):
                    if destination.shape != source.shape or destination.dtype != source.dtype:
                        raise ValueError(f'Resume tensor shape or dtype differs in {key}')
                    destination.copy_(source)
                nets.tree_map(restore_tensor, owner, state[key])
        self.completed_updates = int(state['completed_updates'])
        self.total_collector_steps = int(state['total_collector_steps'])
        self.col_accumulator = float(state['col_accumulator'])
        if int(self.outer_step) != self.completed_updates or int(self.ema_count) != self.completed_updates:
            raise ValueError('Resume update and EMA counters differ')
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        return float(state['wall_s'])

    def save(self, output, elapsed, *, full=False):
        iteration = self.completed_updates
        checkpoint = dict(actor=as_numpy(self.actor), actor_ema=as_numpy(self.ema), critic=as_numpy(self.critic),
            cnorm=as_numpy(self.cnorm), bin_q=as_numpy(self.sampler.bin_q), iter=iteration,
            completed_updates=iteration, ema_updates=int(self.ema_count), evaluation_policy='ema',
            wall_s=elapsed, env_steps=iteration*self.n*self.h,
            training_framework='torch', execution=self.args.execution, perception=self.env.perception_contract,
            observation=self.env.observation_contract)
        destination = output/'ckpt'/f'it_{iteration:06d}.pkl'
        if not destination.exists():
            temporary = destination.with_suffix('.tmp')
            with temporary.open('wb') as handle: pickle.dump(checkpoint,handle)
            temporary.replace(destination)
        if full:
            state = dict(format='nablamimic_torch_training_v1', completed_updates=iteration, wall_s=elapsed,
                perception=self.env.perception_contract, observation=self.env.observation_contract,
                actor=self.actor, critic=self.critic, ema=self.ema, ema_count=self.ema_count,
                actor_optimizer=self.actor_opt, critic_optimizer=self.critic_opt, cnorm=self.cnorm,
                bin_q=self.sampler.bin_q, probabilities=self.sampler.probs, collector=self.collector,
                replay=self.replay, replay_ptr=self.replay_ptr, outer_step=self.outer_step,
                total_collector_steps=self.total_collector_steps, col_accumulator=self.col_accumulator,
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(), resolved_config=vars(self.args))
            destination = output/f'training_state_it{iteration:06d}.pt'
            if destination.exists():raise FileExistsError(destination)
            temporary=destination.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(destination)
        return str(destination)


def build_parser():
    from .training_options import build_parser as donor_parser
    parser = donor_parser()
    parser.prog = 'python -m nablamimic.torch_train'
    parser.description = __doc__
    parser.add_argument('--execution',choices=('eager','compile','compile-graph'),default='compile-graph')
    parser.add_argument('--regression-batch',type=int,default=16384)
    parser.add_argument('--initial-checkpoint',type=Path,help='Current-format initial weights for a matched comparison; update count must be zero')
    parser.add_argument('--save-wall-minutes',type=float,nargs='+',default=[8,20,60,90,180])
    parser.add_argument('--full-state-at-end',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--terrain-model", type=Path, help="Prepared physical terrain MJCF or MJB model")
    parser.add_argument("--terrain-model-contract", type=Path)
    parser.add_argument("--terrain-sensor", type=Path)
    parser.add_argument("--terrain-sensor-ablation", choices=("measured", "constant"), default="measured")
    parser.add_argument("--actor-anchor-position", action="store_true")
    parser.add_argument("--action-limit", type=float, default=None)
    parser.add_argument("--future-stride", type=int, default=1)
    parser.add_argument("--reference-root-displacement", action="store_true")
    return parser


def validate_args(args):
    if args.future_stride < 1:
        raise ValueError("Future reference stride must be positive")
    if args.physics!='engine' or not args.policy_ema or args.mask_after_done or not args.no_remat:
        raise ValueError('Torch path currently implements the unmasked, no-rematerialization EMA Engine recipe')
    if args.shac_strict or args.actor_obs_norm or args.v_bound or args.clip_end_boot!='next':
        raise ValueError('Unsupported recipe variant')
    if not args.anchor_to_decoder or args.enc_hoist!='on' or not args.critic_pre_state:
        raise ValueError('Current-paper anchor/encoder-hoist/pre-state-critic settings required')
    if not args.action_target or args.action_target[:2]!=[8.,1.] or args.replay_steps<=0 or args.reset_sampling!='supermimic':
        raise ValueError('Eight regression steps, one target window and replay with SuperMimic resets required')
    if args.num_envs<1 or args.horizon<1 or args.regression_batch<1 or args.num_envs*args.horizon%args.critic_minibatches:
        raise ValueError('Invalid fixed batch dimensions')
    if not 0 < args.utd <= 1:
        raise ValueError('This collector requires at least one step per outer update (0 < utd <= 1)')
    if args.term_penalty is None or len(args.term_penalty)!=3:
        raise ValueError('Three termination-penalty parameters required')


def build_environment(args):
    """Construct the one physical scene and its checked input bindings."""
    from .env import BM_BODIES, EnvConfig
    from .motion import all_clip_names, load_library
    from nablaengine import torch_ffi
    config = {}
    if args.motion_lib:
        from .motion_lib import ShardedLibrary
        source = ShardedLibrary(args.motion_lib, filtered=False)
        if args.clips != ['all']:
            source.clips = {name: source.clips[name] for name in args.clips}
        shards = sorted({clip['shard'] for clip in source.clips.values()})
        library = source.working_set(shards, sum(clip['length'] for clip in source.clips.values()))
    else:
        names=all_clip_names() if args.clips==['all'] else args.clips
        if args.clips==['all'] and len(names)!=40:raise ValueError('Expected the complete 40-clip library')
        library=load_library(names,bodies=BM_BODIES)
    declared_bodies = tuple(library.bodies)
    if len(declared_bodies) != len(BM_BODIES) or set(declared_bodies) != set(BM_BODIES):
        raise ValueError('Reference tracked-body names do not bind the physical tracking task')
    body_indices = [declared_bodies.index(name) for name in BM_BODIES]
    from dataclasses import replace
    library = replace(library, bodies=BM_BODIES, **{
        name: getattr(library, name)[:, body_indices]
        for name in ('body_pos', 'body_quat', 'body_lin_vel', 'body_ang_vel')})
    config['reference_body_binding'] = dict(declared_order=list(declared_bodies),
        canonical_order=list(BM_BODIES), source_indices=body_indices)
    model = None
    if args.terrain_sensor:
        if args.terrain_model is None:
            raise ValueError("Terrain sensor requires the matching physical terrain model")
        import mujoco
        if args.terrain_model_contract is None:
            raise ValueError("Prepared terrain model requires its physical contract")
        model_contract = json.loads(args.terrain_model_contract.read_text())
        model = (mujoco.MjModel.from_binary_path(str(args.terrain_model))
                 if args.terrain_model.suffix == '.mjb' else
                 mujoco.MjModel.from_xml_path(str(args.terrain_model)))
        if model.body(model.jnt_bodyid[0]).name != 'pelvis':
            raise ValueError("Terrain training requires the validated pelvis-root model")
        actual_order = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
        if actual_order != model_contract['actuator_joint_names']:
            raise ValueError("Constructed actuator binding differs from physical contract")
        if list(BM_BODIES) != model_contract['tracked_body_names']:
            raise ValueError("Constructed tracking body order differs from physical contract")
        if model_contract['terrain_artifact_id'] != json.loads((args.terrain_sensor/'contract.json').read_text())['artifact_id']:
            raise ValueError("Sensor and physical terrain artifact bindings differ")
        config['physical_model'] = model_contract
    env=TorchTrackingEnv(library,EnvConfig(solver_iterations=args.solver_iterations,
        residual_action=args.residual_action,action_rate_space=args.action_rate_space,
        future_stride=args.future_stride, reference_root_displacement=args.reference_root_displacement,
        action_limit=args.action_limit, actor_anchor_position=args.actor_anchor_position), model=model)
    if args.terrain_sensor:
        env.bind_terrain_sensor(args.terrain_sensor, args.terrain_sensor_ablation)
        env.perception_contract['physical_scene_artifact_id'] = model_contract['artifact_id']
        env.perception_contract['physical_frame'] = model_contract['frame']
    config['perception'] = env.perception_contract
    config['observation'] = env.observation_contract
    if args.terrain_sensor:
        expected = np.asarray(model_contract['foot_pairs'], dtype=np.int32)
        actual = []
        for index in range(env.m.npair):
            geoms = [int(env.m.pair_geom1[index]), int(env.m.pair_geom2[index])]
            names = [env.m.geom(geom).name for geom in geoms]
            if any('_foot' in name for name in names) and any(int(env.m.geom_bodyid[geom]) == 0 for geom in geoms):
                actual.append(index)
        if sorted(expected.tolist()) != actual or len(actual) == 0:
            raise ValueError('Prepared foot-pair contract differs from compiled foot/static terrain pairs')
        env.foot_pairs = expected
    engine=torch_ffi.init(model=env.m,foot_pairs=np.asarray(env.foot_pairs),device=env.device)
    config['engine'] = engine
    return env, config


def main(argv=None):
    from .env import BM_BODIES, EnvConfig
    from .motion import all_clip_names,load_library
    from .params import resolve_params,git_state
    from .training_options import FIXED
    from nablaengine import torch_ffi
    resolved, info = resolve_params(sys.argv[1:] if argv is None else argv)
    args=build_parser().parse_args(resolved);validate_args(args)
    output=Path(args.out)
    if output.exists() and any(output.iterdir()):raise ValueError('Fresh output root required')
    (output/'ckpt').mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one provisioned CUDA device required')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    code=os.environ.get('NABLA_SOURCE_REVISION') or git_state()[0]
    if not code:raise ValueError('Declared committed source revision required')
    config={**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},**FIXED,'training_framework':'torch','git_commit':code,'devices':1,'processes':1,
            'gpu':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,
            'random_stream':'Torch default CPU/CUDA generators; same seed label does not imply the JAX random stream',
            'clock':'setup/compilation/capture excluded; per-update RNG, diagnostics, logging and weight saves included; terminal full-state serialization reported separately'}
    config['initial_checkpoint']=str(args.initial_checkpoint) if args.initial_checkpoint else None
    config['inputs']={key:os.environ.get(key) for key in ('LAFAN1_G1_DIR','MOTION_CACHE_DIR','MENAGERIE_DIR')}
    config['declared_environment']=os.environ.get('NABLA_VERIFIED_IMAGE_REFERENCE')
    write_json(output/'config.json',config)
    env, bindings = build_environment(args)
    config.update(bindings)
    write_json(output/'config.json', config)
    if args.initial_checkpoint and not args.resume:
        with args.initial_checkpoint.open('rb') as handle:initial=pickle.load(handle)
        if initial['completed_updates']!=0 or initial['ema_updates']!=0:raise ValueError('Matched initial checkpoint must contain zero completed updates')
        if initial['perception'] != env.perception_contract:
            raise ValueError('Initial checkpoint perception layout differs from constructed environment')
        if initial['observation'] != env.observation_contract:
            raise ValueError('Initial checkpoint observation layout differs from constructed environment')
        actor=nets.tree_from_jax(initial['actor'],env.device)
        critic=nets.tree_from_jax(initial['critic'],env.device)
    else:
        import jax
        from . import nets as jax_nets
        key,ka,kc=jax.random.split(jax.random.PRNGKey(args.seed),3)
        actor=nets.tree_from_jax(jax_nets.actor_init(ka,env,anchor_to_decoder=6),env.device)
        critic=nets.tree_from_jax(jax_nets.mlp_init(kc,[env.critic_dim,*jax_nets.CRITIC_HIDDEN,1]),env.device)
    started=time.monotonic()
    try:
        trainer=TorchTrainer(env,actor,critic,args)
    except BaseException as error:
        write_json(output/'FAILED.json',dict(stage='setup',error=repr(error),
            setup_s=time.monotonic()-started,completed_updates=0))
        raise
    write_json(output/'execution.json',trainer.setup_record)
    torch_ffi.read_diagnostics(reset=True)
    restored_wall = trainer.restore(args.resume) if args.resume else 0.0
    if trainer.completed_updates > args.iters:
        raise ValueError('Requested final update precedes restored state')
    training_started=time.monotonic()-restored_wall
    trainer.save(output,restored_wall)
    stop_requested=False
    def stop(_signal,_frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    deadlines=sorted(set(60*x for x in args.save_wall_minutes));next_deadline=0
    while next_deadline < len(deadlines) and deadlines[next_deadline] <= restored_wall:
        next_deadline += 1
    status='failed'
    try:
        with (output/'log.jsonl').open('x') as log:
            for _ in range(trainer.completed_updates, args.iters):
                metrics=trainer.update();elapsed=time.monotonic()-training_started
                metrics['wall_s']=elapsed
                log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
                if trainer.completed_updates%10==0 or trainer.completed_updates<=2:print(json.dumps(metrics),flush=True)
                due=next_deadline<len(deadlines) and elapsed>=deadlines[next_deadline]
                if trainer.completed_updates%args.ckpt_every==0 or due:
                    trainer.save(output,elapsed)
                    while next_deadline<len(deadlines) and elapsed>=deadlines[next_deadline]:next_deadline+=1
                if stop_requested or (args.hours is not None and elapsed>=args.hours*3600):break
        elapsed=time.monotonic()-training_started
        checkpoint=trainer.save(output,elapsed,full=args.full_state_at_end)
        status='interrupted' if stop_requested else 'completed'
        write_json(output/'terminal.json',dict(status=status,completed_updates=trainer.completed_updates,
            training_s=elapsed,total_s=time.monotonic()-started,checkpoint=checkpoint,
            setup=trainer.setup_record,graphs={k:v.record for k,v in trainer.graphs.items()}))
        if not stop_requested:(output/'DONE').write_text('completed\n')
    except BaseException as error:
        write_json(output/'FAILED.json',dict(error=repr(error),completed_updates=trainer.completed_updates,
            training_s=time.monotonic()-training_started))
        raise


if __name__=='__main__':main()
