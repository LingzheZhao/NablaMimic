"""Frozen N=4096, H=64 Torch compiled/CUDA-graph precision comparison.

Run only in a provisioned pinned GPU environment. One invocation keeps one live
checkpoint identity. It makes no optimizer update and does not resume replay.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time


BATCH, HORIZON, SEED = 4096, 64, 0
RTOL, ATOL = 1e-4, 1e-5
PRECISIONS = ('fp32', 'tf32')
INVALID = (
    'fwd_unsupported_worlds', 'fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds',
    'nonfinite_rows', 'nonfinite_qpos_rows', 'nonfinite_qvel_rows', 'nonfinite_ctrl_rows',
    'nonfinite_cotangent_rows', 'replay_mismatch_worlds',
)
ACTION_FIELDS = ('policy_action', 'action', 'loss_action_gradient', 'clipped_ascent_gradient', 'target')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def compare(reference, actual):
    """Bounded-memory reduction, with the old margins and exact discrete fields."""
    import numpy as np
    same = reference.shape == actual.shape and reference.dtype == actual.dtype
    result = dict(shape=list(actual.shape), dtype=str(actual.dtype), same_layout=same)
    if not same:
        return {**result, 'passed': False, 'finite': False}
    exact = reference.dtype.kind in 'biu'
    a = reference.reshape((1,) + reference.shape) if reference.ndim == 0 else reference
    b = actual.reshape((1,) + actual.shape) if actual.ndim == 0 else actual
    finite, outside, different, signs = True, 0, 0, 0
    aa = bb = ab = dd = maximum = 0.0
    first = None
    for row in range(len(a)):
        x, y = np.asarray(a[row], dtype=np.float64), np.asarray(b[row], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        finite &= bool(valid.all())
        delta = y - x
        mismatch = (~valid) | ((x != y) if exact else (np.abs(delta) > ATOL + RTOL * np.abs(x)))
        if first is None and bool(mismatch.any()):
            index = tuple(int(i) for i in np.argwhere(mismatch)[0]) if mismatch.ndim else ()
            first = ([row] + list(index)) if reference.ndim else []
        outside += int(np.count_nonzero(mismatch))
        different += int(np.count_nonzero(x != y))
        signs += int(np.count_nonzero((x * y) < 0))
        if valid.all():
            maximum = max(maximum, float(np.abs(delta).max(initial=0)))
            aa += float(np.sum(x * x)); bb += float(np.sum(y * y))
            ab += float(np.sum(x * y)); dd += float(np.sum(delta * delta))
    norm_a, norm_b = aa ** .5, bb ** .5
    result.update(passed=finite and outside == 0, finite=finite, exact_required=exact,
        differing_values=different, outside_tolerance=outside, first_outside_tolerance=first,
        max_abs_difference=maximum if finite else None, reference_norm=norm_a if finite else None,
        actual_norm=norm_b if finite else None,
        relative_l2=(dd / aa) ** .5 if finite and aa else None,
        cosine=ab / (norm_a * norm_b) if finite and norm_a and norm_b else None,
        sign_changes=signs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--engine-revision', required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', choices=('live',), default='live')
    parser.add_argument('--expected-updates', type=int)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--expected-gpu-model', default='NVIDIA CMP 170HX')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, source, engine = args.out.resolve(), args.source_root.resolve(), args.engine_root.resolve()
    for protected in (source, engine, args.reference_cache.resolve(), args.checkpoint.resolve().parent):
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            parser.error('Output must be separate from every frozen source/input root')
    if (args.expected_updates is not None and args.expected_updates < 0) or not args.checkpoint.is_file():
        parser.error('A current checkpoint and nonnegative expected update count are required')
    if os.environ.get('DK_ALLOW_UNSUPPORTED') or os.environ.get('DK_MAXCON'):
        parser.error('Unsupported-physics and capacity overrides are forbidden')
    if os.environ.get('TRITON_DEFAULT_FP_FUSION') not in (None, '1'):
        parser.error('Default FMA contraction is required')
    root.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        source_root=str(source), source_revision=args.source_revision, engine_root=str(engine),
        engine_revision=args.engine_revision, checkpoint=str(args.checkpoint.resolve()), policy='live',
        expected_updates=args.expected_updates, conditions={}, comparisons={}, failures=[],
        protocol=dict(batch=BATCH, horizon=HORIZON, seed=SEED, precisions=list(PRECISIONS),
            execution='compiled body/encoder/targets plus explicit CUDA Graph window',
            rtol=RTOL, atol=ATOL, integer_boolean_exact=True, all_nine_invalid_guards=list(INVALID),
            optimizer_updates=0, checkpoint_weights_frozen=True, default_fma=True,
            inputs='Once-materialized randomized reset starts/ages and reset states; explicit shared noise.',
            scope='Controlled reset-start window; not a replay-restored training continuation.',
            target_scope='Production compiled action_targets; no regression or optimizer update.',
            memory_rule='One precision graph at a time; sequential host exports and mmap comparisons.',
            automatic_fallback=False, thresholds_never_relaxed=True))
    started, private_tmp = time.monotonic(), None
    write(root / 'result.json', record)
    try:
        private_tmp = tempfile.mkdtemp(prefix='nabla-window-precision-', dir='/tmp')
        os.environ.update(TMPDIR=private_tmp, JAX_PLATFORMS='cpu',
                          MOTION_CACHE_DIR=str(args.reference_cache.resolve()))
        tempfile.tempdir = private_tmp
        for key, name in (('DK_CACHE_DIR', 'native'), ('G1_FAST_GEN_DIR', 'generator'),
                          ('CUDA_CACHE_PATH', 'cuda'), ('JAX_COMPILATION_CACHE_DIR', 'jax')):
            path = root / 'cache' / name
            path.mkdir(parents=True)
            os.environ[key] = str(path)
        os.environ.pop('DK_TORCH_ARTIFACT', None)
        sys.path[:0] = [str(source / 'src'), str(engine / 'src')]
        import numpy as np
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step, tape
        from nablamimic import torch_nets as nets
        from nablamimic.env import EnvConfig, BM_BODIES
        from nablamimic.motion import all_clip_names, load_library
        from nablamimic.params import resolve_params
        from nablamimic.torch_env import TorchTrackingEnv
        from nablamimic.torch_execution import compile_fullgraph, CudaGraphCall
        from nablamimic.torch_train import build_parser, make_control_body, make_window, ResetSampler

        assert Path(nets.__file__).resolve().is_relative_to(source)
        assert Path(tf.__file__).resolve().is_relative_to(engine)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert torch.cuda.get_device_name(0) == args.expected_gpu_model
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch._dynamo.config.suppress_errors = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        names = all_clip_names()
        assert len(names) == 40 and len(list(args.reference_cache.glob('*.npz'))) == 40
        config = args.config or source / 'configs/ours_ema.args'
        resolved, _ = resolve_params(['--config', str(config), '--out', str(root / 'unused-training-output')])
        recipe = build_parser().parse_args(resolved)
        assert recipe.num_envs == BATCH and recipe.horizon == HORIZON
        assert recipe.action_target[:2] == [8., 1.] and recipe.enc_hoist == 'on'
        assert recipe.anchor_to_decoder and not recipe.actor_obs_norm and not recipe.mask_after_done
        assert recipe.no_remat and recipe.policy_ema and recipe.physics == 'engine'
        assert recipe.residual_action and recipe.action_rate_space == 'setpoint'
        assert recipe.solver_iterations == 10 and recipe.clip_end_boot == 'next' and not recipe.v_bound
        record['resolved_recipe'] = vars(recipe)
        cfg = EnvConfig(solver_iterations=recipe.solver_iterations,
                        residual_action=recipe.residual_action, action_rate_space=recipe.action_rate_space)
        env = TorchTrackingEnv(load_library(names, bodies=BM_BODIES), cfg)
        info = tf.init(model=env.m, foot_pairs=env.foot_pairs)
        assert step.NEFC == 416 and info['maxcon'] == 96
        assert info['rec_words'] == tape.rec_words(env.m) == 1504
        assert info['kinematics_error'] is None
        tf.set_diagnostics(True)
        joint_names = []
        for i in range(env.nu):
            joint = int(env.m.actuator_trnid[i, 0])
            assert env.m.jnt_qposadr[joint] == 7 + i and env.m.jnt_dofadr[joint] == 6 + i
            joint_names.append(env.m.joint(joint).name)
        record.update(gpu=torch.cuda.get_device_name(0), joint_names=joint_names,
            highlighted_action_indices=[21, 27, 28],
            versions=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__),
            engine=dict(nefc=step.NEFC, maxcon=info['maxcon'], rec_words=info['rec_words']))
        with args.checkpoint.open('rb') as handle:
            checkpoint = pickle.load(handle)
        count = int(checkpoint['completed_updates'])
        assert count == int(checkpoint['ema_updates']) == int(checkpoint['iter'])
        assert args.expected_updates is None or count == args.expected_updates
        actor = nets.tree_from_jax(checkpoint['actor'], 'cuda')
        critic = nets.tree_from_jax(checkpoint['critic'], 'cuda')
        cnorm = nets.tree_from_jax(checkpoint['cnorm'], 'cuda')
        assert 'anchor_dec' in actor and actor['anchor_dec'].shape == (6,) and 'obs_norm' not in actor
        assert tuple(layer[0].shape[1] for layer in actor['enc'][:-1]) == nets.ACTOR_HIDDEN
        assert tuple(layer[0].shape[1] for layer in actor['dec'][:-1]) == nets.ACTOR_HIDDEN
        assert actor['enc'][-1][0].shape[1] == nets.ACTOR_LATENT and actor['dec'][-1][0].shape[1] == 29
        record['checkpoint_metadata'] = dict(completed_updates=count, ema_updates=count, wall_s=checkpoint['wall_s'])
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        def save_array(folder, name, tensor):
            value = tensor.detach()
            if value.dtype == torch.bfloat16:
                value = value.float()
            array = value.cpu().numpy()
            np.save(folder / (name + '.npy'), array, allow_pickle=False)
            finite = bool(np.isfinite(array).all())
            if not finite:
                record['failures'].append(str(folder.relative_to(root)) + '/' + name + ':nonfinite')
            return finite

        def save_tree(folder, prefix, tree):
            if isinstance(tree, dict):
                for key in sorted(tree):
                    save_tree(folder, prefix + key + '.', tree[key])
            elif isinstance(tree, (list, tuple)):
                for index, value in enumerate(tree):
                    save_tree(folder, prefix + str(index) + '.', value)
            else:
                save_array(folder, prefix.rstrip('.'), tree)

        with torch.no_grad():
            sampler = ResetSampler(env, recipe.max_episode_steps, recipe.gamma, recipe.uniform_over)
            start_state = sampler.sample(BATCH)
            start_state['ep'].copy_(torch.randint(recipe.max_episode_steps, (BATCH,), device='cuda', dtype=torch.int32))
            reset_state = sampler.sample(BATCH)
            action_noise = recipe.act_noise * torch.randn((HORIZON, BATCH, env.nu), device='cuda')
            observation_noise = env.sample_observation_noise(HORIZON * BATCH)
            proprio_noise = torch.stack((observation_noise['proprio'].reshape(HORIZON, BATCH, -1),
                                         2 * torch.rand((HORIZON, BATCH, 93), device='cuda') - 1))
            goal_noise = observation_noise['goal'].reshape(HORIZON, BATCH, -1)
        delta = torch.zeros_like(action_noise, requires_grad=True)
        probe = torch.zeros(2, device='cuda', requires_grad=True)
        fixed = (start_state, reset_state, action_noise, proprio_noise, goal_noise, delta, probe)
        inputs_folder = root / 'inputs'
        inputs_folder.mkdir()
        save_tree(inputs_folder, '', dict(start=start_state, reset=reset_state, action_noise=action_noise,
            proprio_noise=proprio_noise, goal_noise=goal_noise, delta=delta, probe=probe))
        save_array(inputs_folder, 'rng_cpu', torch.get_rng_state())
        save_array(inputs_folder, 'rng_cuda', torch.cuda.get_rng_state())
        del sampler, observation_noise
        # Register the native custom autograd type before Dynamo compilation.
        registration = env.step({key: value[:1] for key, value in start_state.items()},
            torch.zeros((1, env.nu), device='cuda', requires_grad=True), torch.zeros((1, 93), device='cuda'))
        del registration
        torch.cuda.synchronize()
        record['input_native_diagnostics'] = tf.read_diagnostics(reset=True)
        assert all(record['input_native_diagnostics'].get(key, -1) == 0 for key in INVALID)
        assert not record['failures']
        write(root / 'result.json', record)

        def run_condition(precision):
            folder = root / precision
            folder.mkdir()
            began = time.monotonic()
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
            torch.backends.cudnn.allow_tf32 = False
            torch._dynamo.reset()
            for key, name in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton')):
                path = root / 'cache' / precision / name
                path.mkdir(parents=True)
                os.environ[key] = str(path)
            torch.cuda.reset_peak_memory_stats()
            entry = dict(status='running', allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                         matmul_precision=torch.get_float32_matmul_precision(), cudnn_tf32=False)
            record['conditions'][precision] = entry
            write(root / 'result.json', record)
            rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()

            def encode(parameters, frames, reset_frames):
                offsets = torch.arange(HORIZON, device='cuda')[:, None]
                def sequence(frame):
                    indices = torch.minimum(frame[None] + offsets, env.ref_last[frame][None])
                    return nets.actor_encode(parameters, env.goal_ref(indices.reshape(-1))).reshape(HORIZON, BATCH, -1)
                return sequence(frames), sequence(reset_frames)

            body = compile_fullgraph(make_control_body(env, recipe.gamma, recipe.max_episode_steps,
                recipe.act_cot_clip, tuple(recipe.term_penalty), enc_hoist=True))
            encoder = compile_fullgraph(encode)
            window = make_window(env, actor, critic, cnorm, HORIZON, recipe.gamma, body, encoder)
            def restore_rng():
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state(rng_cuda)
            graph = CudaGraphCall(window, fixed, restore=restore_rng, compile_function=False)
            entry['setup_native_diagnostics'] = tf.read_diagnostics(reset=True)
            if any(entry['setup_native_diagnostics'].get(key, -1) != 0 for key in INVALID):
                record['failures'].append(precision + ':setup_native_invalid')
                write(root / 'result.json', record)
                raise RuntimeError('Native invalid state during capture setup')
            output = graph(*fixed)
            torch.cuda.synchronize()
            entry['window_rng_unchanged'] = bool(torch.equal(rng_cpu, torch.get_rng_state()) and
                                                 torch.equal(rng_cuda, torch.cuda.get_rng_state()))
            entry['native_diagnostics'] = tf.read_diagnostics(reset=True)
            if any(entry['native_diagnostics'].get(key, -1) != 0 for key in INVALID):
                record['failures'].append(precision + ':native_invalid')
            save_array(folder, 'loss', output[0])
            save_array(folder, 'loss_action_gradient', output[1])
            save_array(folder, 'in_graph_clip_statistics', output[2])
            rows = output[3]
            save_array(folder, 'policy_action', rows['a_pi'])
            save_array(folder, 'action', rows['a_pi'] + action_noise)
            for field in ('term', 'done', 'frame', 'r', 'v2'):
                save_array(folder, field, rows[field])
            for field, value in rows['terms'].items():
                save_array(folder, 'reward_term.' + field, value)
            outer_step = torch.tensor(count, device='cuda', dtype=torch.int32)
            def action_targets(action, gradient):
                return nets.action_targets(action, gradient, env.action_scale, recipe.action_target[2],
                    recipe.act_cot_clip, outer_step=outer_step, decay_steps=recipe.lr_decay_steps,
                    delta_floor=recipe.action_target_delta_floor)
            target_function = compile_fullgraph(action_targets)
            target = target_function(rows['a_pi'], output[1])
            for name, key in (('target', 'targets'), ('target_weights', 'weights'),
                              ('clipped_ascent_gradient', 'gradients'), ('target_clip_scale', 'scales'),
                              ('target_eta', 'eta'), ('target_delta', 'step_delta'),
                              ('target_clip_statistics', 'at_clip'), ('target_profile', 'at_prof')):
                save_array(folder, name, target[key])
            save_array(folder, 'target_live', target['weights'] > 0)
            save_array(folder, 'target_clipped', (target['weights'] > 0) & (target['scales'] < 1))
            torch.cuda.synchronize()
            entry['rng_unchanged'] = bool(torch.equal(rng_cpu, torch.get_rng_state()) and
                                          torch.equal(rng_cuda, torch.cuda.get_rng_state()))
            entry['graph'] = graph.record
            assert graph.record['captured'] and graph.record['replays'] == 1
            if not entry['rng_unchanged']:
                record['failures'].append(precision + ':unexpected_random_draw')
            # This is a read-only window: ensure no closed-over weights/statistics changed.
            unchanged = True
            for current, original in ((actor, checkpoint['actor']), (critic, checkpoint['critic']), (cnorm, checkpoint['cnorm'])):
                for left, right in zip(nets.tree_leaves(current), nets.tree_leaves(original), strict=True):
                    unchanged &= bool(np.array_equal(left.detach().cpu().numpy(), np.asarray(right)))
            entry['weights_and_normalizer_unchanged'] = unchanged
            if not unchanged:
                record['failures'].append(precision + ':frozen_parameters_changed')
            entry.update(status='completed', elapsed_s=time.monotonic() - began,
                memory=dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved()))
            write(root / 'result.json', record)
            # No tensor or callable from this graph escapes this function.

        for precision in PRECISIONS:
            run_condition(precision)
            torch.cuda.synchronize()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            record['conditions'][precision]['memory_after_cleanup'] = dict(
                allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
            write(root / 'result.json', record)

        fields = sorted(path.stem for path in (root / 'fp32').glob('*.npy'))
        assert fields == sorted(path.stem for path in (root / 'tf32').glob('*.npy'))
        earliest = None
        for name in fields:
            a = np.load(root / 'fp32' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            b = np.load(root / 'tf32' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            item = compare(a, b)
            if a.ndim and a.shape[0] == HORIZON:
                item['per_step'] = [compare(a[t], b[t]) for t in range(HORIZON)]
                first = item['first_outside_tolerance']
                if first is not None and (earliest is None or first[0] < earliest['step']):
                    earliest = dict(step=first[0], field=name, index=first)
            if name in ACTION_FIELDS:
                assert a.shape == (HORIZON, BATCH, 29)
                item['per_joint'] = [dict(index=j, name=joint_names[j], **compare(a[:, :, j], b[:, :, j]))
                                     for j in range(29)]
            record['comparisons'][name] = item
            if not item['passed']:
                record['failures'].append('precision_parity:' + name)
            del a, b
            write(root / 'result.json', record)
        record['first_step_difference'] = earliest
        support_fields = ('term', 'done', 'frame')
        support_differences = [dict(field=name, index=record['comparisons'][name]['first_outside_tolerance'])
                              for name in support_fields
                              if record['comparisons'][name]['first_outside_tolerance'] is not None]
        record['support'] = dict(exact=all(record['comparisons'][name]['passed'] for name in support_fields),
            fields=list(support_fields),
            first_difference=min(support_differences, key=lambda item: item['index']) if support_differences else None)
        record['parity_passed'] = all(item['passed'] for item in record['comparisons'].values())
        record['status'] = 'passed' if not record['failures'] else 'failed'
        record['scope'] = 'Frozen full-shape Torch window sensitivity; not a learning or final-fidelity causal result.'
    except BaseException as error:
        record.update(status='failed', exception=repr(error))
        for entry in record['conditions'].values():
            if entry.get('status') == 'running':
                entry.update(status='failed', exception=repr(error))
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - started
        write(root / 'result.json', record)
        if private_tmp:
            shutil.rmtree(private_tmp, ignore_errors=True)
    if record['status'] != 'passed':
        raise SystemExit('Strict precision comparison failed; retained outputs and margins are unchanged')


if __name__ == '__main__':
    main()
