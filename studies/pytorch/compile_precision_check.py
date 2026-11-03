"""Fixed-weight Torch precision/compilation diagnostics in a provisioned GPU job.

Six conditions share exact inputs. No optimizer, adaptive sampler or long rollout
can feed a difference back into the next condition. Numerical failures remain
failures at the existing margins; completion is not parity acceptance.
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


RTOL, ATOL, HORIZON, SEED = 1e-4, 1e-5, 4, 0
MODES = ('eager', 'compile', 'compile-graph')
PRECISIONS = ('fp32', 'tf32')


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def compare_arrays(reference, actual):
    """Strict existing margins plus descriptive directional metrics, never a new gate."""
    import numpy as np
    same = reference.shape == actual.shape and reference.dtype == actual.dtype
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    result = dict(shape=list(actual.shape), dtype=str(actual.dtype), same_layout=same, finite=finite)
    if not same or not finite:
        return {**result, 'passed': False}
    a, b = reference.astype(np.float64).reshape(-1), actual.astype(np.float64).reshape(-1)
    difference = b - a
    exact = reference.dtype.kind in 'biu'
    mismatch = b != a if exact else np.abs(difference) > ATOL + RTOL * np.abs(a)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    result.update(passed=not bool(mismatch.any()), exact_required=exact,
        bitwise_equal=reference.tobytes() == actual.tobytes(),
        differing_bits_values=int(np.count_nonzero(reference != actual)),
        outside_tolerance=int(np.count_nonzero(mismatch)),
        max_abs_difference=float(np.abs(difference).max(initial=0)),
        reference_norm=norm_a, actual_norm=norm_b,
        relative_l2=float(np.linalg.norm(difference)) / norm_a if norm_a else None,
        cosine=float(np.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else None,
        sign_changes=int(np.count_nonzero((a * b) < 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--engine-revision', required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', choices=('ema', 'live'), default='ema')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, source, engine = args.out.resolve(), args.source_root.resolve(), args.engine_root.resolve()
    for protected in (source, engine, args.reference_cache.resolve(), args.checkpoint.resolve().parent):
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            parser.error('Output must be separate from all frozen code/input roots')
    if os.environ.get('DK_ALLOW_UNSUPPORTED') or os.environ.get('DK_MAXCON'):
        parser.error('No unsupported-physics or capacity overrides allowed')
    if os.environ.get('TRITON_DEFAULT_FP_FUSION') not in (None, '1'):
        parser.error('This frozen condition requires default FMA contraction')
    root.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(source), source_revision=args.source_revision,
        engine_root=str(engine), engine_revision=args.engine_revision,
        checkpoint=str(args.checkpoint.resolve()), policy=args.policy, argv=sys.argv,
        protocol=dict(rtol=RTOL, atol=ATOL, seed=SEED, horizon=HORIZON, batch=40,
            modes=list(MODES), precisions=list(PRECISIONS), finite_required=True,
            integer_boolean_exact=True, frozen_weights=True, optimizer_updates=0,
            input_rule='Two fixed states per each of forty clips; shared explicit random draws.',
            comparisons='Within each precision: eager/compile and compile/graph; across precision: same mode.',
            storage='Full FP32 policy inputs and separate BF16 roundtrip of exactly the same values.',
            orientation='Negative mean body_ori reward through one native control step, action inputs fixed.',
            diagnostic_only=True, automatic_fallback=False), conditions={}, comparisons={}, failures=[])
    started, private_tmp = time.monotonic(), None
    write(root / 'result.json', record)
    try:
        private_tmp = tempfile.mkdtemp(prefix='nabla-precision-', dir='/tmp')
        os.environ.update(TMPDIR=private_tmp, JAX_PLATFORMS='cpu',
                          MOTION_CACHE_DIR=str(args.reference_cache.resolve()))
        tempfile.tempdir = private_tmp
        for key, folder in (('DK_CACHE_DIR', 'native'), ('G1_FAST_GEN_DIR', 'generator'),
                            ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'),
                            ('CUDA_CACHE_PATH', 'cuda'), ('JAX_COMPILATION_CACHE_DIR', 'jax')):
            path = root / 'cache' / folder
            path.mkdir(parents=True)
            os.environ[key] = str(path)
        os.environ.pop('DK_TORCH_ARTIFACT', None)
        sys.path[:0] = [str(source / 'src'), str(engine / 'src'), str(source / 'studies/pytorch')]
        import numpy as np
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step, tape
        from nablamimic import torch_nets as nets
        from nablamimic.env import EnvConfig, BM_BODIES, build_model
        from nablamimic.motion import all_clip_names, load_library
        from nablamimic.params import resolve_params
        from nablamimic.torch_env import TorchTrackingEnv, abs_real, quat_angle_sq
        from nablamimic.torch_execution import compile_fullgraph, CudaGraphCall
        from nablamimic.torch_train import build_parser, make_control_body, make_window
        from engine_check import BAD_DIAGNOSTICS, host_inputs

        assert Path(nets.__file__).resolve().is_relative_to(source)
        assert Path(tf.__file__).resolve().is_relative_to(engine)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert torch.cuda.get_device_name(0) == 'NVIDIA CMP 170HX'
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch._dynamo.config.suppress_errors = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        names = all_clip_names()
        assert len(names) == 40
        cfg = EnvConfig(solver_iterations=10, residual_action=True, action_rate_space='setpoint')
        _, rows = host_inputs(args.reference_cache, build_model(cfg)[0])
        env = TorchTrackingEnv(load_library(names, bodies=BM_BODIES), cfg)
        info = tf.init(model=env.m, foot_pairs=env.foot_pairs)
        assert step.NEFC == 416 and info['maxcon'] == 96
        assert info['rec_words'] == tape.rec_words(env.m) == 1504
        tf.set_diagnostics(True)
        record.update(versions=dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__),
            gpu=torch.cuda.get_device_name(0), reference_rows=rows,
            engine=dict(origin=tf.__file__, nefc=step.NEFC, maxcon=info['maxcon'], rec_words=info['rec_words']),
            joint_names=[env.m.joint(i).name for i in range(1, env.m.njnt)],
            highlighted_action_indices=[21, 27, 28])
        with args.checkpoint.open('rb') as handle:
            checkpoint = pickle.load(handle)
        count = int(checkpoint['completed_updates'])
        assert count == int(checkpoint['ema_updates']) == int(checkpoint['iter'])
        actor = nets.tree_from_jax(checkpoint['actor_ema' if args.policy == 'ema' else 'actor'], 'cuda')
        critic = nets.tree_from_jax(checkpoint['critic'], 'cuda')
        cnorm = nets.tree_from_jax(checkpoint['cnorm'], 'cuda')
        assert 'anchor_dec' in actor and 'obs_norm' not in actor
        trainable = nets.actor_trainable(actor)
        record['checkpoint_metadata'] = dict(completed_updates=count, ema_updates=count,
                                             wall_s=checkpoint['wall_s'])
        resolved, _ = resolve_params(['--config', str(source / 'configs/ours_ema.args'),
                                     '--out', str(root / 'resolved-recipe')])
        recipe = build_parser().parse_args(resolved)
        record['recipe'] = {key: getattr(recipe, key) for key in
            ('gamma', 'max_episode_steps', 'act_cot_clip', 'term_penalty', 'act_noise',
             'action_target', 'lr_decay_steps', 'action_target_delta_floor')}
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        def host(tree, prefix=''):
            if isinstance(tree, dict):
                return {name: value for key in sorted(tree)
                        for name, value in host(tree[key], prefix + key + '.').items()}
            if isinstance(tree, (tuple, list)):
                return {name: value for i, child in enumerate(tree)
                        for name, value in host(child, prefix + str(i) + '.').items()}
            value = tree.detach()
            if value.dtype == torch.bfloat16:
                value = value.float()
            return {prefix.rstrip('.'): value.cpu().numpy().copy()}

        def retain(path, tree):
            arrays = host(tree)
            if not all(np.isfinite(value).all() for value in arrays.values()):
                record['failures'].append(str(path.relative_to(root)) + ':nonfinite')
            np.savez(path, **arrays)
            return arrays

        # Materialize every random input once, before compiling or capturing.
        cases = []
        for case in range(2):
            frames = torch.as_tensor(env.starts + np.minimum(25 + case * 50, env.lengths - HORIZON - 1),
                                     device='cuda', dtype=torch.int32)
            reset_frames = torch.as_tensor(env.starts + np.minimum(100 + case * 50, env.lengths - 2),
                                           device='cuda', dtype=torch.int32)
            assert bool((frames >= torch.as_tensor(env.starts, device='cuda')).all())
            with torch.no_grad():
                start_noise, reset_noise = env.sample_reset_noise(40), env.sample_reset_noise(40)
                start, reset = env.reset(frames, start_noise), env.reset(reset_frames, reset_noise)
                action_noise = recipe.act_noise * torch.randn((HORIZON, 40, env.nu), device='cuda')
                observation_noise = env.sample_observation_noise(HORIZON * 40)
                proprio_noise = torch.stack((observation_noise['proprio'].reshape(HORIZON, 40, -1),
                                             2 * torch.rand((HORIZON, 40, 93), device='cuda') - 1))
                goal_noise = observation_noise['goal'].reshape(HORIZON, 40, -1)
                history, goal = env.policy_obs(start, goal_noise[0])
                action = nets.actor_apply(actor, history, goal) + action_noise[0]
                cotangent = torch.randn_like(action)
            delta = torch.zeros_like(action_noise, requires_grad=True)
            probe = torch.zeros(2, device='cuda', requires_grad=True)
            window_inputs = (start, reset, action_noise, proprio_noise, goal_noise, delta, probe)
            cases.append(dict(window=window_inputs, history=history, goal=goal,
                              action=action.detach().clone().requires_grad_(), cotangent=cotangent))
            retain(root / f'inputs_{case}.npz', dict(start=start, reset=reset, start_noise=start_noise,
                reset_noise=reset_noise, action_noise=action_noise, proprio_noise=proprio_noise,
                goal_noise=goal_noise, history=history, goal=goal, action=action, cotangent=cotangent))
        record['fixed_inputs'] = [str(root / f'inputs_{case}.npz') for case in range(2)]
        write(root / 'result.json', record)
        # Register lazy Engine autograd before Dynamo sees the first native call.
        env.step(cases[0]['window'][0], cases[0]['action'], cases[0]['window'][3][0, 0])

        def actor_vjp(parameters, history, goal, cotangent):
            def function(p, h, g):
                return nets.actor_apply(nets.with_actor_parameters(actor, p), h, g)
            output, backward = torch.func.vjp(function, parameters, history, goal)
            parameter_grad, history_grad, goal_grad = backward(cotangent)
            return dict(action=output, parameter_vjp=parameter_grad, history_vjp=history_grad, goal_vjp=goal_grad)

        def orientation_primitives(real, quaternion, reference):
            absolute, abs_backward = torch.func.vjp(abs_real, real)
            abs_gradient, = abs_backward(torch.ones_like(real))
            angle, angle_backward = torch.func.vjp(quat_angle_sq, quaternion, reference)
            quaternion_gradient, reference_gradient = angle_backward(torch.ones_like(angle))
            return dict(absolute=absolute, abs_gradient=abs_gradient, angle_sq=angle,
                        quaternion_gradient=quaternion_gradient, reference_gradient=reference_gradient)

        real = torch.tensor([-1., -1e-6, -0., 0., 1e-6, 1.], device='cuda')
        angles = torch.tensor([0., 1e-7, 1e-4, .3, 1.5707963267948966, 3.141592653589793], device='cuda')
        quaternion = torch.stack((torch.cos(angles / 2), torch.sin(angles / 2),
                                  torch.zeros_like(angles), torch.zeros_like(angles)), dim=-1)
        quaternion[-1, 0] = 0.  # Exact pi boundary, rather than rounded cos(pi/2).
        reference = torch.zeros_like(quaternion)
        reference[:, 0] = 1.
        primitive_inputs = (real, quaternion, reference)
        retain(root / 'primitive_inputs.npz', dict(real=real, quaternion=quaternion, reference=reference))

        def encode(parameters, frames, reset_frames):
            offsets = torch.arange(HORIZON, device='cuda')[:, None]
            def sequence(frame):
                indices = torch.minimum(frame[None] + offsets, env.ref_last[frame][None])
                return nets.actor_encode(parameters, env.goal_ref(indices.reshape(-1))).reshape(HORIZON, 40, -1)
            return sequence(frames), sequence(reset_frames)

        def orientation_forward(state, action, noise):
            stepped = env.step(state, action, noise)
            _, terms = env.reward(stepped, action, state['prev_a'], prev=state)
            return -terms['body_ori'].mean(), terms['body_ori'], stepped['qpos'], stepped['xquat'], env.terminated(stepped)

        def targets(action, gradient):
            return nets.action_targets(action, gradient, env.action_scale, recipe.action_target[2],
                recipe.act_cot_clip, outer_step=torch.tensor(count, dtype=torch.int32, device='cuda'),
                decay_steps=recipe.lr_decay_steps, delta_floor=recipe.action_target_delta_floor)

        for precision in PRECISIONS:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
            torch.backends.cudnn.allow_tf32 = False
            torch._dynamo.reset()
            for mode in MODES:
                condition = precision + '-' + mode
                folder = root / condition
                folder.mkdir()
                condition_started = time.monotonic()
                entry = dict(status='running', matmul_precision=torch.get_float32_matmul_precision(),
                    allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_tf32=False, graphs={})
                record['conditions'][condition] = entry
                write(root / 'result.json', record)
                body = make_control_body(env, recipe.gamma, recipe.max_episode_steps, recipe.act_cot_clip,
                                         tuple(recipe.term_penalty), enc_hoist=True)
                network, encoder, orientation, target, primitive = (
                    actor_vjp, encode, orientation_forward, targets, orientation_primitives)
                if mode != 'eager':
                    network, encoder, body, orientation, target, primitive = map(compile_fullgraph,
                        (network, encoder, body, orientation, target, primitive))
                window = make_window(env, actor, critic, cnorm, HORIZON, recipe.gamma, body, encoder)

                def orientation_gradient(state, action, noise):
                    loss, reward, qpos, quat, term = orientation(state, action, noise)
                    gradient, = torch.autograd.grad(loss, (action,))
                    clipped, statistics = nets.clip_cotangent(gradient, recipe.act_cot_clip)
                    norms = torch.linalg.vector_norm(gradient, dim=-1)
                    threshold = recipe.act_cot_clip * nets.masked_nanmedian(norms, norms > 0)
                    return dict(loss=loss, reward=reward, qpos=qpos, xquat=quat, terminated=term,
                                action_gradient=gradient, clipped_gradient=clipped, clip_statistics=statistics,
                                clipped_rows=(norms > 0) & (norms > threshold))

                operations = dict(network=network, window=window, orientation=orientation_gradient,
                                  orientation_primitives=primitive)
                examples = dict(network=(trainable, cases[0]['history'], cases[0]['goal'], cases[0]['cotangent']),
                    window=cases[0]['window'],
                    orientation=(cases[0]['window'][0], cases[0]['action'], cases[0]['window'][3][0, 0]),
                    orientation_primitives=primitive_inputs)
                if mode == 'compile-graph':
                    operations = {name: CudaGraphCall(function, examples[name], restore=lambda: None,
                                                     compile_function=False)
                                  for name, function in operations.items()}
                setup_diagnostics = tf.read_diagnostics(reset=True)
                entry['setup_native_diagnostics'] = setup_diagnostics
                if any(setup_diagnostics.get(key, -1) != 0 for key in BAD_DIAGNOSTICS):
                    record['failures'].append(condition + ':setup_native_invalid')
                primitive_arrays = retain(folder / 'orientation_primitives.npz',
                                          operations['orientation_primitives'](*primitive_inputs))
                entry['real_abs_zero_vjp_contract'] = bool(np.array_equal(
                    primitive_arrays['abs_gradient'], np.asarray([-1, -1, 1, 1, 1, 1], dtype=np.float32)))
                for case, fixed in enumerate(cases):
                    network_inputs = (trainable, fixed['history'], fixed['goal'], fixed['cotangent'])
                    retain(folder / f'network_{case}.npz', operations['network'](*network_inputs))
                    # Same function and cotangent, but the actual regression-storage roundtrip is isolated.
                    bf_history = fixed['history'].to(torch.bfloat16).float()
                    bf_goal = fixed['goal'].to(torch.bfloat16).float()
                    retain(folder / f'network_bf16_{case}.npz',
                           operations['network'](trainable, bf_history, bf_goal, fixed['cotangent']))
                    native = operations['window'](*fixed['window'])
                    retain(folder / f'window_{case}.npz', dict(loss=native[0], action_gradient=native[1],
                        clip_statistics=native[2], outputs=native[3], targets=target(native[3]['a_pi'], native[1])))
                    retain(folder / f'orientation_{case}.npz', operations['orientation'](
                        fixed['window'][0], fixed['action'], fixed['window'][3][0, 0]))
                    del native
                diagnostics = tf.read_diagnostics(reset=True)
                entry['native_diagnostics'] = diagnostics
                if any(diagnostics.get(key, -1) != 0 for key in BAD_DIAGNOSTICS):
                    record['failures'].append(condition + ':native_invalid')
                if mode == 'compile-graph':
                    entry['graphs'] = {name: operation.record for name, operation in operations.items()}
                    expected_replays = dict(network=4, window=2, orientation=2, orientation_primitives=1)
                    assert all(graph['captured'] and graph['replays'] == expected_replays[name]
                               for name, graph in entry['graphs'].items())
                entry.update(status='completed', elapsed_s=time.monotonic() - condition_started)
                write(root / 'result.json', record)
                del operations, window, body, network, encoder, orientation, target, primitive, orientation_gradient
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        pairs = [(f'{p}-eager', f'{p}-compile') for p in PRECISIONS]
        pairs += [(f'{p}-compile', f'{p}-compile-graph') for p in PRECISIONS]
        pairs += [('fp32-' + mode, 'tf32-' + mode) for mode in MODES]
        for baseline, candidate in pairs:
            comparison = {}
            for path in sorted((root / baseline).glob('*.npz')):
                with np.load(path, allow_pickle=False) as expected, np.load(root / candidate / path.name,
                                                                          allow_pickle=False) as actual:
                    assert set(expected.files) == set(actual.files)
                    fields = {key: compare_arrays(expected[key], actual[key]) for key in expected.files}
                    # Preserve action-column evidence for every joint, not only the nominated wrists.
                    for key in expected.files:
                        if 'gradient' in key and expected[key].ndim >= 2 and expected[key].shape[-1] == 29:
                            fields[key]['per_action'] = [compare_arrays(expected[key][..., j], actual[key][..., j])
                                                        for j in range(29)]
                    comparison[path.name] = fields
            record['comparisons'][baseline + '__' + candidate] = comparison
            write(root / 'result.json', record)
        record['storage_comparisons'] = {}
        for condition in record['conditions']:
            for case in range(2):
                with np.load(root / condition / f'network_{case}.npz', allow_pickle=False) as reference, \
                        np.load(root / condition / f'network_bf16_{case}.npz', allow_pickle=False) as stored:
                    record['storage_comparisons'][condition + f':case{case}'] = {
                        key: compare_arrays(reference[key], stored[key]) for key in reference.files}
        record['parity_passed'] = all(field['passed'] for pair in record['comparisons'].values()
                                      for fields in pair.values() for field in fields.values())
        record['status'] = 'completed' if not record['failures'] else 'failed'
        record['scope'] = 'Conditional compiler/precision diagnosis; not evidence for an end-to-end learning gap cause.'
    except BaseException as error:
        record.update(status='failed', exception=repr(error))
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - started
        write(root / 'result.json', record)
        if private_tmp:
            shutil.rmtree(private_tmp, ignore_errors=True)
    if record['status'] != 'completed':
        raise SystemExit('Runtime/finite/native guard failed; preserve and inspect partial outputs')


if __name__ == '__main__':
    main()
