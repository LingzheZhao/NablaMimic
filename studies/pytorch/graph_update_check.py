"""Compare actual eager or compiled TorchTrainer updates with CUDA-graph updates.

One native model and all forty reference clips are shared. Each trainer starts
from the same initial weights and seed, with the real randomized collector,
replay, differentiable window, actor/critic Adam updates and policy EMA. This is
a bounded transformation check, not a long-training or motion-fidelity result.
The default eager reference tests compilation plus graph capture. A compile
reference isolates capture/storage and does not establish compiler-versus-eager
equivalence; it is a separate condition with the same strict numerical rules.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time

from engine_contract import (add_engine_capacity_arguments, expected_engine_capacity,
                             capacity_validator_command, capacity_validator_environment)

from engine_check import BAD_DIAGNOSTICS, OPTIONS, host_inputs
from training_check import write


RTOL, ATOL = 1e-4, 1e-5
SEED, BATCH, HORIZON, UPDATES = 0, 8, 4, 2
REPLAY_STEPS, REGRESSION_BATCH = 8, 64
GRAPH_NAMES = ('window', 'regression', 'critic', 'collector', 'statistics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]/'configs/ours_torch.args')
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--capacity-validator', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--engine-base-revision', required=True)
    parser.add_argument('--engine-capacity-revision', required=True)
    parser.add_argument('--jax-cache-seed', type=Path, default=os.environ.get('NABLA_ENGINE_CACHE_SEED'))
    parser.add_argument('--expected-gpu-model', default='NVIDIA CMP 170HX')
    parser.add_argument('--reference-execution', choices=('eager', 'compile'), default='eager',
                        help='Reference path: eager tests compilation plus capture; compile isolates capture/storage '
                             'and does not test compiler-versus-eager equivalence (default: eager)')
    parser.add_argument('--matmul-precision', choices=('fp32', 'tf32'), default='fp32',
                        help='Precision for both paths: fp32 disables TF32; tf32 uses production CUDA matmul '
                             'TF32 permission with cuDNN TF32 disabled (default: fp32)')
    add_engine_capacity_arguments(parser)
    args = parser.parse_args()
    capacity = expected_engine_capacity(args)
    reference_mode = args.reference_execution
    modes = (reference_mode, 'compile-graph')
    if os.environ.get('DK_ALLOW_UNSUPPORTED'):
        parser.error('Unsupported-physics bypass is forbidden')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    record = dict(expected_engine_capacity=capacity,compiler_environment=dict(TRITON_DEFAULT_FP_FUSION=os.environ.get('TRITON_DEFAULT_FP_FUSION', 'unset; Triton default true')), status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, python=sys.executable, checks=[], failures=[], modes={},
        source_revision=args.source_revision, engine_base_revision=args.engine_base_revision,
        engine_capacity_revision=args.engine_capacity_revision,
        initial_checkpoint=str(args.initial_checkpoint.resolve()), reference_cache=str(args.reference_cache.resolve()),
        scope=('Compilation and CUDA Graph capture versus eager execution.' if reference_mode == 'eager' else
               'CUDA Graph capture/storage versus uncaptured compiled execution; excludes compiler-versus-eager equivalence.'),
        protocol=dict(seed=SEED, batch=BATCH, horizon=HORIZON, updates=UPDATES,
            replay_steps=REPLAY_STEPS, regression_batch=REGRESSION_BATCH, modes=list(modes),
            reference_execution=reference_mode,
            rtol=RTOL, atol=ATOL, finite_required=True, integer_boolean_exact=True,
            initial_persistent_state_bitwise=True, cpu_cuda_rng_bitwise=True,
            matmul_precision=args.matmul_precision,
            precision=('FP32 highest, CUDA matmul/cudnn TF32 disabled' if args.matmul_precision == 'fp32' else
                       'FP32 tensors, production CUDA matmul TF32 permission enabled, cuDNN TF32 disabled'),
            engine_options=OPTIONS, randomized_resets=True, observation_noise=True, domain_randomization=True,
            termination_rule='Matched term/done/frame support arrays exactly; falls and partial episodes retained',
            exact_support_fields=['training_data.term', 'training_data.done', 'training_data.frame',
                                  'collector.frame', 'collector.ep', 'replay.frame', 'replay.ep'],
            independent_history_rule='For every saved collector transition, continuing histories shift exactly '
                                    'one row; a reset repeats its first row ten times. Compare bits in each mode.',
            automatic_fallback=False,
            bfloat16_storage='Saved as exact FP32 values in NPZ; runtime regression storage remains BF16',
            optimizer_schedule='Current recipe including warmup; no schedule acceleration for this check',
            window_observation='Actual uncaptured reference call return, or graph.outputs immediately after update; never stale setup output',
            compared=['actor', 'critic', 'Adam moments/counts', 'EMA/count', 'normalizer', 'reset-bin statistics',
                      'frame probabilities', 'collector', 'complete small replay', 'training data', 'action targets',
                      'critic targets', 'actual window inputs and explicit noise',
                      'window loss/action gradients/clipping statistics', 'CPU/CUDA RNG', 'counters'],
            untested=['full-size memory/performance']+
                     (['TF32 training transform'] if args.matmul_precision == 'fp32' else [])+
                     ['long-training fidelity']+
                     (['compiler-versus-eager equivalence'] if reference_mode == 'compile' else [])))
    started = time.monotonic()
    private_tmp = None

    def save():
        write(root/'result.json', record)

    def check(name, passed, **fields):
        record['checks'].append(dict(name=name, passed=bool(passed), **fields))
        if not passed:
            record['failures'].append(name)
        return bool(passed)

    def require(name, passed, **fields):
        if not check(name, passed, **fields):
            save()
            raise AssertionError(name)

    save()  # Freeze the rules before runtime setup or either trainer's execution.
    try:
        private_tmp = tempfile.mkdtemp(prefix='nabla-graph-update-check-', dir='/tmp')
        os.environ['TMPDIR'] = private_tmp
        tempfile.tempdir = private_tmp
        # JAX is used only by the existing CPU model/reference imports and parser.
        # Initializing its CUDA runtime before capture is deliberately unnecessary.
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['MOTION_CACHE_DIR'] = str(args.reference_cache.resolve())
        for key, name in (('DK_CACHE_DIR', 'native'), ('JAX_COMPILATION_CACHE_DIR', 'jax'),
                          ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'),
                          ('CUDA_CACHE_PATH', 'cuda')):
            path = root/'cache'/name
            path.mkdir(parents=True)
            os.environ[key] = str(path)
        if args.jax_cache_seed:
            libraries = sorted(args.jax_cache_seed.resolve().glob('*.so'))
            require('native_cache_seed_available', bool(libraries))
            for path in libraries:
                shutil.copyfile(path, root/'cache'/'native'/path.name)
            record['cache_seed'] = dict(path=str(args.jax_cache_seed), libraries=[p.name for p in libraries])
        validator = subprocess.run(capacity_validator_command(args, root), capture_output=True,
                                   text=True, env=capacity_validator_environment(), timeout=180)
        (root/'capacity-validator.log').write_text(validator.stdout+validator.stderr)
        require('corrected_source_capacity_validator', validator.returncode == 0)

        import numpy as np
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step as engine_step, tape as engine_tape
        from nablamimic.env import BM_BODIES, EnvConfig, build_model
        from nablamimic.motion import all_clip_names, load_library
        from nablamimic.params import resolve_params
        from nablamimic import torch_nets as nets
        from nablamimic.torch_env import TorchTrackingEnv
        from nablamimic.torch_train import TorchTrainer, build_parser, validate_args, validate_initial_checkpoint

        require('engine_source_location', Path(tf.__file__).resolve().is_relative_to(args.engine_root.resolve()),
                module=str(Path(tf.__file__).resolve()))
        require('single_visible_gpu', torch.cuda.is_available() and torch.cuda.device_count() == 1)
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = args.matmul_precision == 'tf32'
        torch.backends.cudnn.allow_tf32 = False
        record['observed_precision'] = dict(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
        require('matmul_precision_configuration',
                record['observed_precision']['cuda_matmul_allow_tf32'] == (args.matmul_precision == 'tf32') and
                not record['observed_precision']['cudnn_allow_tf32'], observed=record['observed_precision'])
        torch._dynamo.config.suppress_errors = False
        require('expected_gpu', torch.cuda.get_device_name(0) == args.expected_gpu_model,
                actual=torch.cuda.get_device_name(0))
        cfg = EnvConfig(solver_iterations=10, residual_action=True, action_rate_space='setpoint')
        record['env_config'] = asdict(cfg)
        names = all_clip_names()
        require('forty_current_clips', len(names) == 40)
        _, validated_rows = host_inputs(args.reference_cache, build_model(cfg)[0])
        record['reference_rows'] = validated_rows
        env = TorchTrackingEnv(load_library(names, bodies=BM_BODIES), cfg)
        info = tf.init(model=env.m, foot_pairs=env.foot_pairs)
        tf.set_diagnostics(True)
        require('corrected_engine_capacity', engine_step.NEFC == capacity['constraint_rows'] and info['maxcon'] == capacity['max_contacts'] and
                info['rec_words'] == engine_tape.rec_words(env.m) == capacity['record_words'] and
                engine_tape.layout(env.m)['zone'][1] == capacity['zone_words'] and info['kinematics_error'] is None)
        record['engine'] = {key: info[key] for key in
                            ('version', 'arch', 'nq', 'nv', 'nu', 'nbody', 'maxcon', 'rec_words')}
        record['engine']['nefc'] = int(engine_step.NEFC)
        record['versions'] = dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__)
        with args.initial_checkpoint.open('rb') as handle:
            initial = pickle.load(handle)
        validate_initial_checkpoint(initial, env)
        require('initial_checkpoint_zero_updates', initial['iter'] == 0 and initial['env_steps'] == 0)
        require('current_actor', 'anchor_dec' in initial['actor'] and
                initial['actor']['anchor_dec'].shape == (6,) and 'obs_norm' not in initial['actor'])

        def cpu(value):
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if value.dtype == torch.bfloat16:
                    value = value.float()
                return value.cpu().numpy().copy()
            return np.asarray(value)

        def flatten(tree, prefix=''):
            if isinstance(tree, dict):
                return {name: value for key in sorted(tree)
                        for name, value in flatten(tree[key], prefix+key+'.').items()}
            if isinstance(tree, (tuple, list)):
                return {name: value for i, item in enumerate(tree)
                        for name, value in flatten(item, prefix+str(i)+'.').items()}
            return {prefix.rstrip('.'): cpu(tree)}

        def snapshot(trainer, window_result, metrics=None, window_inputs=None):
            state = dict(actor=trainer.actor, critic=trainer.critic, ema=trainer.ema,
                actor_opt=trainer.actor_opt, critic_opt=trainer.critic_opt, ema_count=trainer.ema_count,
                cnorm=trainer.cnorm, bin_q=trainer.sampler.bin_q, probabilities=trainer.sampler.probs,
                collector=trainer.collector, replay=trainer.replay, replay_ptr=trainer.replay_ptr,
                counters=dict(outer=trainer.outer_step, completed=trainer.completed_updates,
                    total_collector_steps=trainer.total_collector_steps, col_accumulator=trainer.col_accumulator),
                training_data=trainer.training_data, target_data=trainer.target_data,
                critic_targets=trainer.critic_targets,
                window=dict(loss=window_result[0], delta_gradient=window_result[1], clip_statistics=window_result[2]),
                rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()))
            if metrics is not None:
                state['metrics'] = {key: value for key, value in metrics.items()
                                    if key not in ('iteration_s', 'memory', 'engine_diagnostics')}
            if window_inputs is not None:
                state['window_inputs'] = dict(zip(
                    ('start', 'reset', 'action_noise', 'proprio_noise', 'goal_noise', 'delta', 'probe'), window_inputs))
            return flatten(state)

        def compare_snapshots(phase, actual):
            failures = {}
            exact_initial = ('actor.', 'critic.', 'ema.', 'actor_opt.', 'critic_opt.', 'ema_count', 'cnorm.',
                             'bin_q', 'probabilities', 'collector.', 'replay.', 'replay_ptr', 'counters.')
            with np.load(root/reference_mode/f'update_{phase:02d}.npz', allow_pickle=False) as expected:
                require(f'update{phase}.snapshot_keys', set(expected.files) == set(actual),
                        **{reference_mode: sorted(expected.files), 'graph': sorted(actual)})
                for name, graph_value in actual.items():
                    reference_value = expected[name]
                    same = reference_value.shape == graph_value.shape and reference_value.dtype == graph_value.dtype
                    finite = bool(np.isfinite(reference_value).all() and np.isfinite(graph_value).all())
                    exact = reference_value.dtype.kind in 'biu' or name.startswith(('rng.',
                        'window_inputs.action_noise', 'window_inputs.proprio_noise', 'window_inputs.goal_noise',
                        'window_inputs.delta', 'window_inputs.probe')) or (
                        phase == 0 and name.startswith(exact_initial))
                    if same and finite:
                        difference = graph_value.astype(np.float64)-reference_value.astype(np.float64)
                        mismatch = (graph_value != reference_value) if exact else (
                            np.abs(difference) > ATOL+RTOL*np.abs(reference_value))
                        equal = reference_value.tobytes() == graph_value.tobytes() if exact else not mismatch.any()
                        max_abs = float(np.abs(difference).max()) if difference.size else 0.
                        ratio = float((np.abs(difference)/(ATOL+RTOL*np.abs(reference_value))).max()) if difference.size and not exact else None
                    else:
                        equal, max_abs, ratio, mismatch = False, None, None, None
                    if not check(f'update{phase}.{name}', equal, shape=list(graph_value.shape),
                                 dtype=str(graph_value.dtype), finite=finite, exact=exact,
                                 max_abs_difference=max_abs, max_tolerance_ratio=ratio,
                                 differing_elements=int(mismatch.sum()) if mismatch is not None else None):
                        if same:
                            failures[name] = graph_value.astype(np.float64)-reference_value.astype(np.float64)
            if failures:
                np.savez(root/f'failed_differences_update_{phase:02d}.npz', **failures)

        def graph_counts(trainer, mode, phase):
            if mode == reference_mode:
                require(f'{mode}.update{phase}.no_graphs', not trainer.graphs)
                return
            expected = dict(window=phase+1, collector=int(phase/0.4), regression=phase*8,
                            critic=phase*20, statistics=phase)
            require(f'{mode}.update{phase}.graph_names', set(trainer.graphs) == set(GRAPH_NAMES))
            for name, count in expected.items():
                graph_record = trainer.graphs[name].record
                check(f'{mode}.update{phase}.graph.{name}', graph_record['captured'] and
                      graph_record['replays'] == count,
                      captured=graph_record['captured'], replays=graph_record['replays'], expected_replays=count)

        def replay_history_checks(mode, phase, values, previous):
            """Check copies within each trajectory, independently of backend parity.

            The previous collector snapshot supplies the predecessor even when
            the ring wraps. Refuse to inspect overwritten intermediate states.
            """
            total = int(values['counters.total_collector_steps'])
            history, episode = values['collector.hist'], values['collector.ep']
            pointer = int(values['replay_ptr'][0])
            current = dict(total=total, history=history.copy(), episode=episode.copy(), pointer=pointer)
            if previous is None:
                require(mode+'.history.initial_collector_steps', total == 0, actual=total)
                return current  # Constructor warmup/capture is not a training transition.
            prefix = f'{mode}.update{phase}.history'
            pool, ages = values['replay.hist'], values['replay.ep']
            count, batch = total-previous['total'], len(episode)
            require(prefix+'.retained_transitions', batch > 0 and len(pool) % batch == 0 and
                    0 <= count <= len(pool)//batch and history.shape[1:] == (10, 93) and
                    pool.shape[1:] == history.shape[1:] and pool.dtype == history.dtype == np.float32,
                    new_collector_steps=count, replay_rows=len(pool), collector_batch=batch)
            expected_pointer = (previous['pointer']+count*batch) % len(pool)
            require(prefix+'.write_pointer', pointer == expected_pointer,
                    actual=pointer, expected=expected_pointer)
            old_history, old_episode = previous['history'], previous['episode']
            for offset in range(count):
                step = previous['total']+offset+1
                positions = (previous['pointer']+offset*batch+np.arange(batch)) % len(pool)
                new_history, new_episode = pool[positions], ages[positions]
                continuing = new_episode == old_episode+1
                reset = new_episode == 0
                step_name = prefix+f'.step{step}'
                check(step_name+'.episode_boundary', (continuing | reset).all(),
                      continuing_count=int(continuing.sum()), reset_count=int(reset.sum()),
                      replay_rows=positions.tolist())
                # uint32 compares the original FP32 storage, including signed zero.
                # Every copied history value has an exact predecessor; no float
                # tolerance is appropriate for this structural copy contract.
                shifted = (new_history[:, :-1].view(np.uint32) != old_history[:, 1:].view(np.uint32))
                shifted &= continuing[:, None, None]
                repeated = (new_history.view(np.uint32) != new_history[:, :1].view(np.uint32))
                repeated &= reset[:, None, None]
                for name, mismatch in (('continuation_shift', shifted), ('reset_repeat', repeated)):
                    check(step_name+'.'+name, not mismatch.any(), exact=True,
                          differing_elements=int(mismatch.sum()),
                          affected_environments=np.flatnonzero(mismatch.any(axis=(1, 2))).tolist(),
                          continuing_count=int(continuing.sum()), reset_count=int(reset.sum()))
                old_history, old_episode = new_history, new_episode
            check(prefix+'.collector_matches_last_replay',
                  history.tobytes() == old_history.tobytes() and np.array_equal(episode, old_episode), exact=True)
            return current

        class ObserveReferenceWindow:
            """Record the actual return without changing inputs, outputs or math."""
            def __init__(self, function):
                self.function, self.result, self.inputs = function, None, None
            def __call__(self, *inputs):
                self.inputs = inputs
                self.result = self.function(*inputs)
                return self.result

        for mode in modes:
            mode_root = root/mode
            mode_root.mkdir()
            overrides = ['--config', str(args.config.resolve()), '--out', str(mode_root),
                '--initial-checkpoint', str(args.initial_checkpoint.resolve()),
                '--matmul-precision', 'highest' if args.matmul_precision == 'fp32' else 'nets_tf32',
                '--execution', mode, '--seed', str(SEED), '--num-envs', str(BATCH), '--collect-envs', str(BATCH),
                '--horizon', str(HORIZON), '--replay-steps', str(REPLAY_STEPS),
                '--regression-batch', str(REGRESSION_BATCH), '--iters', str(UPDATES)]
            resolved, _ = resolve_params(overrides)
            config = build_parser().parse_args(resolved)
            validate_args(config)
            require(mode+'.fixed_recipe', config.utd == .4 and config.critic_epochs == 5 and
                config.critic_minibatches == 4 and config.action_target[:2] == [8., 1.] and
                config.lr_warmup_steps == 100 and config.lr_decay_steps == 16000 and
                config.residual_action and config.action_rate_space == cfg.action_rate_space and
                config.solver_iterations == cfg.solver_iterations and config.clips == ['all'])
            write(mode_root/'resolved_config.json', {key: str(value) if isinstance(value, Path) else value
                                                     for key, value in vars(config).items()})
            # The seed reset is after all non-training initialization and before
            # every constructor, including identical random collector/reset draws.
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            actor = nets.tree_from_jax(initial['actor'], 'cuda')
            critic = nets.tree_from_jax(initial['critic'], 'cuda')
            trainer = TorchTrainer(env, actor, critic, config)
            torch.cuda.synchronize()
            diagnostics = tf.read_diagnostics(reset=True)
            require(mode+'.setup_native_diagnostics', all(diagnostics.get(key, -1) == 0 for key in BAD_DIAGNOSTICS),
                    diagnostics=diagnostics)
            record['modes'][mode] = dict(setup=json.loads(json.dumps(trainer.setup_record)), updates=[])
            write(mode_root/'setup.json', trainer.setup_record)
            observer = ObserveReferenceWindow(trainer.window) if mode == reference_mode else None
            if observer is not None:
                trainer.window = observer
            previous_history = None
            for phase in range(UPDATES+1):
                metrics = None
                if phase:
                    collector_steps_before = trainer.total_collector_steps
                    metrics = trainer.update()
                    collector_steps = trainer.total_collector_steps-collector_steps_before
                    metrics.update(window_terminations=int(trainer.training_data['term'].sum()),
                        window_episode_ends=int(trainer.training_data['done'].sum()), window_samples=BATCH*HORIZON,
                        collector_steps=collector_steps,
                        collector_terminations=round(metrics['collector_termination_rate']*collector_steps*BATCH))
                    record['modes'][mode]['updates'].append(metrics)
                window_result = (trainer.window_outputs if phase == 0 else observer.result) if mode == reference_mode else trainer.graphs['window'].outputs
                require(f'{mode}.update{phase}.window_observed', window_result is not None)
                graph_counts(trainer, mode, phase)
                window_inputs = None if phase == 0 else (
                    observer.inputs if mode == reference_mode else trainer.graphs['window']._inputs)
                values = snapshot(trainer, window_result, metrics, window_inputs)
                for name, value in values.items():
                    check(f'{mode}.update{phase}.finite.{name}', np.isfinite(value).all())
                np.savez(mode_root/f'update_{phase:02d}.npz', **values)
                previous_history = replay_history_checks(mode, phase, values, previous_history)
                if mode == 'compile-graph':
                    compare_snapshots(phase, values)
                write(mode_root/f'execution_{phase:02d}.json', dict(
                    graphs={name: graph.record for name, graph in trainer.graphs.items()}, metrics=metrics,
                    numpy_rng_state=[np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                     *np.random.get_state()[2:]], python_rng_state=random.getstate()))
                save()
                del values
            del trainer, actor, critic, observer, window_result, window_inputs, previous_history
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        record['status'] = 'failed' if record['failures'] else 'passed'
    except BaseException as error:
        record['status'] = 'failed'
        record['exception'] = repr(error)
        raise
    finally:
        record['elapsed_s'] = time.monotonic()-started
        save()
        if private_tmp:
            shutil.rmtree(private_tmp, ignore_errors=True)
    if record['failures']:
        raise SystemExit(reference_mode+'/graph update comparison failed; inspect result.json and retained arrays')


if __name__ == '__main__':
    main()
