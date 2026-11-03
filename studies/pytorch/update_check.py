"""One mathematical outer-update comparison on a fresh synthetic window.

Run on a provisioned CPU environment. The JAX oracle executes the current
train.main action-target, critic and statistics functions extracted from their
AST; only existing scan-loss arrays are added to two return tuples. Torch uses
the actual TorchTrainer regression/critic/statistics closures. Both consume the
same window and the exact indices/permutations drawn by the JAX code.

This does not execute physics, a real rollout, the full network architecture,
Torch compilation or CUDA graphs, and establishes no training-speed result.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


RTOL, ATOL = 1e-4, 1e-6
HORIZON, BATCH, REGRESSION_BATCH, SEED = 4, 8, 64, 20261103
ORACLE_FUNCTIONS = ('bin_weights', 'frame_probs', 'action_target_update', 'critic_step', 'stats_step')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def extract_oracle(path, namespace, output):
    source = path.read_text()
    main = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    nodes = []
    for name in ORACLE_FUNCTIONS:
        node = copy.deepcopy(next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == name))
        if name in ('action_target_update', 'critic_step'):
            # Expose the existing scan's losses; its computation and all updates
            # remain unchanged. Critic ls contains the five epoch means.
            final = node.body[-1]
            if not isinstance(final, ast.Return) or not isinstance(final.value, ast.Tuple):
                raise ValueError(f'Current oracle return contract changed: {name}')
            final.value.elts.append(ast.Name(id='ls', ctx=ast.Load()))
        nodes.append(node)
    # Execute the official averaging expression unchanged; the Torch port does
    # not introduce a second JAX EMA implementation or modify train.py.
    nodes.append(copy.deepcopy(next(node for node in main.body if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == 'ema_step' for target in node.targets))))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    output.write_text(ast.unparse(module) + '\n')
    exec(compile(module, str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  source_revision=args.source_revision, argv=sys.argv, checks=[], failures=[], scope=__doc__,
                  protocol=dict(rtol=RTOL, atol=ATOL, finite_required=True, integer_boolean_exact=True,
                  horizon=HORIZON, batch=BATCH, regression_batch=REGRESSION_BATCH,
                  history_dim=930, goal_dim=686, action_dim=29, critic_dim=825,
                  actor_hidden=[12, 8], actor_latent=4, critic_hidden=[12, 8],
                  actor_updates=8, critic_epochs=5, critic_minibatches=4, ema_updates=1,
                  initial_optimizer_count=0, initial_optimizer_moments='zero',
                  lr_warmup_outer_steps=100, lr_decay_outer_steps=16000, seed=SEED,
                  precision='CPU float32; highest matmul precision; BF16 actor buffer as in donor',
                  input_source='Fresh deterministic synthetic window; no historical artifacts',
                  sampling='Actual JAX random.split/randint/permutation indices shared with Torch',
                  oracle_instrumentation='Append existing actor scan losses and critic epoch means to returns'))
    started = time.monotonic()
    write(root/'result.json', record)  # Freeze margins before executing either implementation.
    try:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        import torch
        from nablamimic import nets as jn, torch_nets as tn
        from nablamimic.torch_train import ResetSampler, TorchTrainer

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        jn.set_matmul_precision('highest')
        if jax.default_backend() != 'cpu':
            raise RuntimeError('The mathematical update check requires the CPU backend')
        record['versions'] = dict(jax=jax.__version__, optax=optax.__version__, torch=torch.__version__, numpy=np.__version__)
        record['oracle_source'] = str(Path(jn.__file__).with_name('train.py'))

        def host(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
            array = np.asarray(value)
            return array.astype(np.float32) if str(array.dtype) == 'bfloat16' else array

        def leaves(tree, prefix=''):
            if isinstance(tree, dict):
                return {key: value for name in sorted(tree) for key, value in leaves(tree[name], prefix+'/'+name).items()}
            if isinstance(tree, (list, tuple)):
                return {key: value for i, item in enumerate(tree) for key, value in leaves(item, prefix+'/'+str(i)).items()}
            return {prefix: host(tree)}

        def archive(name, tree):
            np.savez(root/(name+'.npz'), **leaves(tree))

        def compare(name, left, right):
            aa, bb = leaves(left), leaves(right)
            if aa.keys() != bb.keys():
                raise AssertionError(f'{name}: differing tree layouts')
            maximum = 0.0
            passed = True
            for key in aa:
                a, b = aa[key], bb[key]
                same = a.shape == b.shape and a.dtype == b.dtype
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                exact = a.dtype.kind in 'biu'
                equal = same and finite and (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=RTOL, atol=ATOL))
                if same and finite and a.size:
                    maximum = max(maximum, float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max()))
                if not equal:
                    record['failures'].append(name+key)
                passed &= equal
            record['checks'].append(dict(name=name, passed=bool(passed), leaves=len(aa), max_abs_difference=maximum))
            write(root/'result.json', record)

        rng = np.random.default_rng(SEED)
        cfg = SimpleNamespace(action_target=[8., 1., .03], action_target_delta_floor=.3,
            action_target_lr=6e-4, act_cot_clip=3., lr=2e-4, lr_warmup_steps=100,
            lr_decay_steps=16000, lr_decay_to=.1, horizon=HORIZON, gamma=.99, lam=.95,
            critic_pre_state=True, critic_epochs=5, critic_minibatches=4,
            reset_sampling='supermimic', uniform_over='bins', max_episode_steps=500, execution='eager',
            actor_ema=.999, actor_ema_warmup=True)
        record['resolved_mathematics'] = vars(cfg)
        env = SimpleNamespace(hist_dim=930, goal_dim=686, nu=29, critic_dim=825, n_frames=201,
            max_reward=.1, action_scale=jnp.asarray(np.linspace(.15, .8, 29, dtype=np.float32)))
        key_actor, key_critic, key_regression, key_fit = jax.random.split(jax.random.PRNGKey(SEED), 4)
        actor = jn.actor_init(key_actor, env, hidden=(12, 8), latent=4, anchor_to_decoder=6)
        critic = jn.mlp_init(key_critic, [env.critic_dim, 12, 8, 1])
        cnorm = (jnp.zeros(env.critic_dim), jnp.ones(env.critic_dim))
        def normal(shape):
            return jnp.asarray(rng.normal(0, .2, shape).astype(np.float32))
        history = normal((HORIZON, BATCH, env.hist_dim))
        goal = normal((HORIZON, BATCH, env.goal_dim))
        cobs, cobs_pre = normal((HORIZON, BATCH, env.critic_dim)), normal((HORIZON, BATCH, env.critic_dim))
        term = jnp.zeros((HORIZON, BATCH), dtype=bool).at[1, 2].set(True).at[3, 4].set(True)
        done = term.at[0, 0].set(True).at[2, 5].set(True)
        outs = dict(hist=history.astype(jnp.bfloat16), goal=goal.astype(jnp.bfloat16),
            a_pi=jn.actor_apply(actor, history, goal), cobs=cobs, cobs_pre=cobs_pre,
            r=jnp.asarray(rng.uniform(-.02, .1, (HORIZON, BATCH)).astype(np.float32)),
            term=term, done=done, frame=jnp.asarray(rng.integers(0, 200, (HORIZON, BATCH), dtype=np.int32)),
            v2=jn.critic_apply(critic, jn.normalize(cnorm, cobs))[..., 0])
        gradient = normal((HORIZON, BATCH, env.nu)).at[0, 0].set(0.).at[1, 1].multiply(100.)
        def schedule(per, lr):
            return optax.warmup_cosine_decay_schedule(0., lr, 100*per, 16000*per, end_value=lr*.1)
        reg_schedule, critic_schedule = schedule(8, cfg.action_target_lr), schedule(20, cfg.lr)
        opt_r = optax.chain(optax.clip_by_global_norm(1.), optax.clip(.1),
                            optax.adam(reg_schedule, b1=.9, b2=.95, eps=1e-8))
        opt_c = optax.chain(optax.clip_by_global_norm(1.), optax.adam(critic_schedule))
        initial_cstate = opt_c.init(critic)
        buf = dict(hist=jnp.zeros((1, *outs['hist'].shape), jnp.bfloat16),
            goal=jnp.zeros((1, *outs['goal'].shape), jnp.bfloat16), tgt=jnp.zeros((1, *outs['a_pi'].shape)),
            w=jnp.zeros((1, HORIZON, BATCH)), count=jnp.int32(0), it=jnp.int32(0), opt=opt_r.init(actor))
        tenv = SimpleNamespace(**{**vars(env), 'device': torch.device('cpu'),
            'action_scale': tn.tree_from_jax(env.action_scale),
            'valid': torch.tensor([i < 200 and i % 50 != 49 for i in range(201)])})
        sampler = ResetSampler(tenv, cfg.max_episode_steps, cfg.gamma, cfg.uniform_over)
        aux = dict(bins=jnp.asarray(host(sampler.bin_index), dtype=jnp.int32),
                   valid=jnp.asarray(host(sampler.valid)), bin_count=jnp.asarray(host(sampler.bin_count)))
        bin_q = jnp.asarray(host(sampler.bin_q))
        namespace = dict(jax=jax, jnp=jnp, optax=optax, nets=jn, actor_apply=jn.actor_apply,
            critic_apply=jn.critic_apply, normalize=jn.normalize, args=cfg, env=env, n=BATCH,
            n_bins=sampler.n_bins, g_max=sampler.g_max, opt_r=opt_r, opt_c=opt_c, AT_BATCH=REGRESSION_BATCH)
        extract_oracle(Path(jn.__file__).with_name('train.py'), namespace, root/'extracted_jax_oracle.py')
        replicate = lambda tree: jax.tree.map(lambda value: value[None], tree)
        first = lambda tree: jax.tree.map(lambda value: value[0], tree)
        def actor_reference(a, b, g, o, k):
            result = namespace['action_target_update'](a, b, g, o, k)
            return (*result, o['at_clip'])

        # These are exactly the donor reg()/epoch() draws, including the otherwise
        # redundant K=1 window draw. Torch never tries to match them by a seed label.
        reg_indices = []
        for key in jax.random.split(key_regression, 8):
            ka, kb, kc = jax.random.split(key, 3)
            si = jax.random.randint(ka, (REGRESSION_BATCH,), 0, 1)
            ti = jax.random.randint(kb, (REGRESSION_BATCH,), 0, HORIZON)
            ei = jax.random.randint(kc, (REGRESSION_BATCH,), 0, BATCH)
            reg_indices.append(jnp.stack((si, ti, ei)))
        reg_indices = jnp.stack(reg_indices)
        permutations = jnp.stack([jax.random.permutation(key, HORIZON*BATCH)
                                  for key in jax.random.split(key_fit, 5)])
        archive('fresh_inputs', dict(actor=actor, critic=critic, normalizer=cnorm, window=outs,
            loss_action_gradient=gradient, action_scale=env.action_scale, bin_q=bin_q, bin_tables=aux,
            regression_indices=reg_indices, critic_permutations=permutations))
        ja, jb, jloss, jlosses, jclip = first(jax.pmap(actor_reference, axis_name='d')(
            *map(replicate, (actor, buf, gradient, outs, key_regression))))
        jc, js, jcloss, jy, jfit, jepochs = first(jax.pmap(namespace['critic_step'], axis_name='d')(
            *map(replicate, (critic, initial_cstate, cnorm, outs, key_fit))))
        jnorm, jbins = first(jax.pmap(namespace['stats_step'], axis_name='d')(
            *map(replicate, (cnorm, bin_q, outs, aux, jy))))
        javerage = namespace['ema_step'](actor, ja, jnp.float32(min(cfg.actor_ema, 1 / 10)))
        jcount = jnp.int32(1)
        jprobs = namespace['frame_probs'](jbins, aux)
        def to_torch(value):
            if str(value.dtype) == 'bfloat16':
                return torch.from_numpy(np.array(value, dtype=np.float32)).to(torch.bfloat16)
            return torch.from_numpy(np.array(value, copy=True))
        data = jax.tree.map(to_torch, outs)
        ta, tc, tnorm = tn.tree_from_jax(actor), tn.tree_from_jax(critic), tn.tree_from_jax(cnorm)
        target = tn.action_targets(data['a_pi'], tn.tree_from_jax(gradient), tenv.action_scale,
            cfg.action_target[2], cfg.act_cot_clip, outer_step=torch.tensor(0, dtype=torch.int32),
            decay_steps=cfg.lr_decay_steps, delta_floor=cfg.action_target_delta_floor)
        ty = tn.td_lambda(data['r'], data['term'], data['done'], data['v2'], cfg.gamma, cfg.lam)
        average, count = tn.init_actor_ema(ta)
        trainable = tn.actor_trainable(ta)
        trainer = SimpleNamespace(args=cfg, env=tenv, actor=ta, actor_params=trainable, critic=tc,
            actor_opt=tn.adam_init(trainable), critic_opt=tn.adam_init(tc), cnorm=tnorm,
            reg_hist=data['hist'].reshape(-1, env.hist_dim), reg_goal=data['goal'].reshape(-1, env.goal_dim),
            reg_targets=target['targets'].reshape(-1, env.nu), reg_weights=target['weights'].reshape(-1),
            critic_inputs=data['cobs_pre'].reshape(-1, env.critic_dim), critic_y=ty.reshape(-1),
            training_data=data, critic_targets=ty, sampler=sampler, ema=average, ema_count=count,
            outer_step=torch.tensor(0, dtype=torch.int32))
        TorchTrainer._build_update_functions(trainer, lambda function: function)
        actor_losses, actor_rates = [], []
        for indices in np.asarray(reg_indices):
            flat = torch.from_numpy((indices[1]*BATCH+indices[2]).astype(np.int64))
            loss, _, rate = trainer.regression(flat)
            actor_losses.append(loss.detach().clone()); actor_rates.append(rate.detach().clone())
        critic_losses, critic_rates = [], []
        for permutation in np.asarray(permutations):
            for indices in permutation.reshape(4, -1):
                loss, _, rate = trainer.critic_update(torch.from_numpy(indices.astype(np.int64)))
                critic_losses.append(loss.detach().clone()); critic_rates.append(rate.detach().clone())
        trainer.statistics()
        # Fit uses the OLD normalizer in the actual critic step; check that exact
        # support, then compare the separate post-fit normalizer update below.
        old_normalized = tn.normalize(tn.tree_from_jax(cnorm), trainer.critic_inputs)
        before = ((tn.critic_apply(tn.tree_from_jax(critic), old_normalized)[:, 0]-trainer.critic_y)**2).mean()
        after = ((tn.critic_apply(tc, old_normalized)[:, 0]-trainer.critic_y)**2).mean()
        variance = ((trainer.critic_y-trainer.critic_y.mean())**2).mean()
        tfit = torch.stack((before, after, 1-before/(variance+1e-12), 1-after/(variance+1e-12)))
        archive('jax_outputs', dict(actor=ja, critic=jc, actor_optimizer=jb['opt'], critic_optimizer=js,
            targets=jb['tgt'][0], weights=jb['w'][0], actor_losses=jlosses, critic_epoch_losses=jepochs,
            td_targets=jy, critic_fit=jfit, normalizer=jnorm, bin_q=jbins, probabilities=jprobs,
            ema=javerage, ema_count=jcount))
        archive('torch_outputs', dict(actor=ta, critic=tc, actor_optimizer=trainer.actor_opt,
            critic_optimizer=trainer.critic_opt, targets=target['targets'], weights=target['weights'],
            actor_losses=torch.stack(actor_losses), critic_losses=torch.stack(critic_losses), td_targets=ty,
            critic_fit=tfit, normalizer=tnorm, bin_q=sampler.bin_q, probabilities=sampler.probs,
            ema=trainer.ema, ema_count=trainer.ema_count))
        compare('action_targets', target['targets'], jb['tgt'][0])
        compare('target_weights', target['weights'], jb['w'][0])
        compare('target_clipping_statistics', target['at_clip'], jclip)
        physical_rms = torch.sqrt((((target['targets']-data['a_pi'])*tenv.action_scale)**2 *
            target['weights'][..., None]).sum()/(target['weights'].sum()*env.nu))
        compare('target_setpoint_rms', physical_rms, jnp.asarray(.03, dtype=jnp.float32))
        compare('actor_regression_losses', torch.stack(actor_losses), jlosses)
        compare('actor_mean_loss', torch.stack(actor_losses).mean(), jloss)
        compare('critic_epoch_losses', torch.stack(critic_losses).reshape(5, 4).mean(1), jepochs)
        compare('critic_mean_loss', torch.stack(critic_losses).mean(), jcloss)
        compare('actor_parameters', ta, ja)
        compare('critic_parameters', tc, jc)
        for name, ts, state, per, rates, schedule_fn in (
            ('actor', trainer.actor_opt, jb['opt'][2], 8, actor_rates, reg_schedule),
            ('critic', trainer.critic_opt, js[1], 20, critic_rates, critic_schedule)):
            for moment in ('mu', 'nu'):
                reference = getattr(state[0], moment)
                if name == 'actor': reference = tn.actor_trainable(reference)
                compare(name+'_'+moment, ts[moment], reference)
            compare(name+'_adam_counter', ts['count'], state[0].count)
            compare(name+'_schedule_counter', ts['count'], state[1].count)
            compare(name+'_learning_rates', torch.stack(rates), jnp.stack([schedule_fn(i) for i in range(per)]))
            compare(name+'_required_updates', ts['count'], jnp.int32(per))
        compare('anchor_zero_first_moment', jb['opt'][2][0].mu['anchor_dec'], jnp.zeros(6))
        compare('anchor_zero_second_moment', jb['opt'][2][0].nu['anchor_dec'], jnp.zeros(6))
        compare('td_lambda_targets', ty, jy)
        compare('critic_fit', tfit, jfit)
        compare('normalizer', tnorm, jnorm)
        compare('bin_statistics', sampler.bin_q, jbins)
        compare('reset_frame_probabilities', sampler.probs, jprobs)
        compare('ema_policy', trainer.ema, javerage)
        compare('ema_counter', trainer.ema_count, jcount)
        compare('outer_counter', trainer.outer_step, jb['it'])
        record['status'] = 'passed' if not record['failures'] else 'failed'
        record['elapsed_s'] = time.monotonic()-started
        write(root/'result.json', record)
        if record['failures']:
            raise AssertionError('Mathematical outer-update comparison failed: '+', '.join(record['failures']))
        print(json.dumps(dict(status=record['status'], checks=len(record['checks']), elapsed_s=record['elapsed_s'])))
    except BaseException as error:
        record.update(status='failed', error=repr(error), elapsed_s=time.monotonic()-started)
        write(root/'result.json', record)
        raise


if __name__ == '__main__':
    main()
