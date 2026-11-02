"""Paired short-window NablaEngine training mathematics, JAX versus Torch.

Run only in a provisioned CUDA environment. This checks the real differentiable
control window with shared weights, native physical states and reference tables.
It does not establish an entire optimizer update, randomized training equivalence,
compiler/graph correctness, long-horizon tracking, or training speed. The separate
engine check owns bitwise native-interface and compiled CUDA-graph validation.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

from engine_contract import (add_engine_capacity_arguments, expected_engine_capacity,
                             capacity_validator_command, capacity_validator_environment)

from engine_check import BAD_DIAGNOSTICS, OPTIONS, host_inputs


RTOL, ATOL = 1e-4, 1e-5
BATCH, HORIZON, GAMMA, SEED = 8, 4, 0.99, 20261103
CASES = ('continuation', 'timeout', 'clip_end', 'termination')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--engine-root', type=Path, required=True)
    parser.add_argument('--capacity-validator', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--engine-base-revision', required=True)
    parser.add_argument('--engine-capacity-revision', required=True)
    parser.add_argument('--jax-cache-seed', type=Path, default=os.environ.get('NABLA_ENGINE_CACHE_SEED'))
    parser.add_argument('--expected-gpu-model', default='NVIDIA CMP 170HX')
    add_engine_capacity_arguments(parser)
    args = parser.parse_args()
    capacity = expected_engine_capacity(args)
    if os.environ.get('DK_ALLOW_UNSUPPORTED'):
        parser.error('Unsupported-physics bypass is forbidden')
    if os.environ.get('JAX_PLATFORMS') in ('cpu', 'interpreter'):
        parser.error('The paired oracle requires JAX and Torch on the provisioned GPU')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    record = dict(expected_engine_capacity=capacity,status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  argv=sys.argv, python=sys.executable, checks=[], failures=[], cases={},
                  source_revision=args.source_revision, engine_base_revision=args.engine_base_revision,
                  engine_capacity_revision=args.engine_capacity_revision,
                  scope=__doc__, protocol=dict(batch=BATCH, horizon=HORIZON, gamma=GAMMA,
                  seed=SEED, cases=list(CASES), rtol=RTOL, atol=ATOL, finite_required=True,
                  integer_boolean_exact=True, precision='FP32, TF32 disabled in both frameworks',
                  remat=False, mask_after_done=False, action_noise=0, act_clip=3,
                  term_penalty=[2, .08, .10], record_pre=True, record_obs=True, enc_hoist=True,
                  engine_options=OPTIONS, default_max_episode_steps=500, timeout_max_episode_steps=2,
                  termination_root_z_offset=.5, native_state_source='Torch NablaEngine kinematics',
                  reset_inputs='One detached shared reset state reused on every done within each window',
                  gradient_inputs='delta[H,N,29] and probe[2]; shared frozen actor/critic weights',
                  execution='Torch eager make_control_body/make_window; JAX jitted make_rollout',
                  untested=['full paired optimizer outer update', 'randomized observations/resets',
                            'compiled training window', 'training CUDA graph', 'long-horizon fidelity']),
                  initial_checkpoint=str(args.initial_checkpoint) if args.initial_checkpoint else None,
                  reference_cache=str(args.reference_cache.resolve()))
    started = time.monotonic()
    private_tmp = None

    def save():
        write(root/'result.json', record)

    def check(name, passed, **fields):
        record['checks'].append(dict(name=name, passed=bool(passed), **fields))
        if not passed:
            record['failures'].append(name)
        save()
        return bool(passed)

    def require(name, passed, **fields):
        if not check(name, passed, **fields):
            raise AssertionError(name)

    save()  # Protocol and margins are written before loading or executing either backend.
    try:
        private_tmp = tempfile.mkdtemp(prefix='nabla-training-check-', dir='/tmp')
        os.environ['TMPDIR'] = private_tmp
        tempfile.tempdir = private_tmp
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_CLIENT_MEM_FRACTION'] = '.45'
        os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
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
        (root/'capacity-validator.log').write_text(validator.stdout + validator.stderr)
        require('corrected_source_capacity_validator', validator.returncode == 0)

        import numpy as np
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step as engine_step, tape as engine_tape
        from nablamimic.env import BM_BODIES, EnvConfig, State, TrackingEnv, Xi, build_model
        from nablamimic.motion import all_clip_names, load_library
        from nablamimic.torch_env import TorchTrackingEnv
        from nablamimic import torch_nets as tn
        from nablamimic.torch_train import make_control_body, make_window

        require('engine_source_location', Path(tf.__file__).resolve().is_relative_to(args.engine_root.resolve()),
                module=str(Path(tf.__file__).resolve()))
        require('one_visible_gpu', torch.cuda.is_available() and torch.cuda.device_count() == 1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.45, 0)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        require('expected_gpu', torch.cuda.get_device_name(0) == args.expected_gpu_model,
                actual=torch.cuda.get_device_name(0))
        cfg = EnvConfig(dr=False, obs_noise=False, reset_noise=False, solver_iterations=10,
                        residual_action=True, action_rate_space='setpoint')
        record['env_config'] = asdict(cfg)
        names = all_clip_names()
        require('forty_current_clips', len(names) == 40)
        # Validate every existing cache before load_library can attempt a rebuild.
        _, validated_rows = host_inputs(args.reference_cache, build_model(cfg)[0])
        library = load_library(names, bodies=BM_BODIES)
        tenv = TorchTrackingEnv(library, cfg)
        record['reference_rows'] = validated_rows
        info = tf.init(model=tenv.m, foot_pairs=tenv.foot_pairs)
        tf.set_diagnostics(True)
        # Torch artifact metadata omits nefc; the corrected native source owns it.
        require('corrected_engine_capacity', engine_step.NEFC == capacity['constraint_rows'] and info['maxcon'] == capacity['max_contacts'] and
                info['rec_words'] == engine_tape.rec_words(tenv.m) == capacity['record_words'] and
                engine_tape.layout(tenv.m)['zone'][1] == capacity['zone_words'] and info['kinematics_error'] is None,
                rows=int(engine_step.NEFC), contacts=int(info['maxcon']),
                record_words=int(info['rec_words']), zone_words=int(engine_tape.layout(tenv.m)['zone'][1]))

        import jax
        import jax.numpy as jnp
        from nablaengine import jax_ffi as jf
        from nablamimic import nets as jn, sim
        from nablamimic.train import make_rollout

        require('jax_gpu_backend', jax.default_backend() == 'gpu', actual=jax.default_backend())
        jn.set_matmul_precision('highest')
        jenv = TrackingEnv(library, cfg)
        # Both FFIs use the exact same MjModel object; MJX is a shape/template holder.
        jenv.m = tenv.m
        jenv.warp = sim.make_engine(jenv, {BATCH}, clamp_kv=0, normal=3, clamp_soft=None)
        require('identical_packed_model_constants',
                tf._STATE['consts'].dtype == jf._STATE['consts'].dtype and
                tf._STATE['consts'].shape == jf._STATE['consts'].shape and
                tf._STATE['consts'].tobytes() == jf._STATE['consts'].tobytes())
        record['versions'] = dict(torch=torch.__version__, torch_cuda=torch.version.cuda, jax=jax.__version__)
        record['engine'] = {key: info.get(key) for key in
                            ('version', 'arch', 'nq', 'nv', 'nu', 'nbody', 'maxcon', 'rec_words')}
        record['engine']['nefc'] = int(engine_step.NEFC)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().to(torch.float32).cpu().numpy() if value.dtype == torch.bfloat16 else value.detach().cpu().numpy()
            value = np.asarray(value)
            return value.astype(np.float32) if str(value.dtype) == 'bfloat16' else value

        def compare(name, a, b):
            a, b = cpu(a), cpu(b)
            same = a.shape == b.shape
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            exact = a.dtype.kind in 'biu' or b.dtype.kind in 'biu'
            equal = same and finite and (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=RTOL, atol=ATOL))
            difference = np.abs(a.astype(np.float64)-b.astype(np.float64)) if same and finite else None
            check(name, equal, shape_jax=list(a.shape), shape_torch=list(b.shape), finite=finite,
                  exact=exact, max_abs_difference=float(difference.max()) if difference is not None and a.size else None,
                  max_tolerance_ratio=float((difference/(ATOL+RTOL*np.abs(b))).max()) if difference is not None and a.size and not exact else None)

        def flat(tree, prefix=''):
            if isinstance(tree, dict):
                return {name: value for key in sorted(tree) for name, value in flat(tree[key], prefix+key+'.').items()}
            return {prefix.rstrip('.'): cpu(tree)}

        def compare_tree(name, left, right):
            left, right = flat(left), flat(right)
            if check(name+'.keys', left.keys() == right.keys(), jax=sorted(left), torch=sorted(right)):
                for key in left:
                    compare(name+'.'+key, left[key], right[key])

        def diagnostics(name):
            td, jd = tf.read_diagnostics(reset=True), jf.read_diagnostics(reset=True)
            check(name, all(key in values and values[key] == 0 for values in (td, jd) for key in BAD_DIAGNOSTICS),
                  torch={k: int(v) for k, v in td.items()}, jax={k: int(v) for k, v in jd.items()})

        for key in tenv.TABLES:
            compare('shared_reference.'+key, jenv.tables()[key], tenv.tables()[key])
        if args.initial_checkpoint:
            with args.initial_checkpoint.open('rb') as handle:
                initial = pickle.load(handle)
            require('initial_checkpoint_zero_updates', initial['completed_updates'] == 0 and initial['ema_updates'] == 0)
            ja, jc = jax.tree.map(jnp.asarray, initial['actor']), jax.tree.map(jnp.asarray, initial['critic'])
            norm = jax.tree.map(jnp.asarray, initial['cnorm'])
        else:
            _, ka, kc = jax.random.split(jax.random.PRNGKey(SEED), 3)
            ja = jn.actor_init(ka, jenv, anchor_to_decoder=6)
            jc = jn.mlp_init(kc, [jenv.critic_dim, *jn.CRITIC_HIDDEN, 1])
            norm = (jnp.zeros(jenv.critic_dim), jnp.ones(jenv.critic_dim))
        require('current_actor_contract', 'anchor_dec' in ja and ja['anchor_dec'].shape == (6,) and 'obs_norm' not in ja)
        ta, tc, tnorm = (tn.tree_from_jax(value, 'cuda') for value in (ja, jc, norm))
        with (root/'shared_parameters.pkl').open('wb') as handle:
            pickle.dump(jax.tree.map(np.asarray, dict(actor=ja, critic=jc, cnorm=norm)), handle)
        # Static MJX leaves are never consumed by the native step; only these seven
        # actual physical fields and explicit environment fields are used at runtime.
        template = jax.tree.map(lambda x: jnp.broadcast_to(x, (BATCH, *x.shape)), jenv.d0)

        def to_jax_state(state):
            v = {key: jnp.asarray(cpu(value)) for key, value in state.items()}
            dx = template.replace(qpos=v['qpos'], qvel=v['qvel'], qacc_warmstart=v['warm'],
                                  xpos=v['xpos'], xquat=v['xquat'], cvel=v['cvel'], subtree_com=v['subtree_com'])
            return State(dx, v['frame'], v['ep'], v['hist'], v['prev_a'],
                         Xi(v['friction'], v['enc_bias'], v['com']), jax.random.split(jax.random.PRNGKey(SEED), BATCH))

        def encode(actor, frames, reset_frames):
            offsets = torch.arange(HORIZON, device=tenv.device)[:, None]
            def sequence(frame):
                f = torch.minimum(frame[None]+offsets, tenv.ref_last[frame][None])
                return tn.actor_encode(actor, tenv.goal_ref(f.reshape(-1))).reshape(HORIZON, BATCH, -1)
            return sequence(frames), sequence(reset_frames)

        # Eight evenly spaced identities; the full reference tables remain present.
        clip_indices = np.linspace(0, len(names)-1, BATCH, dtype=np.int64)
        normal_frames = np.asarray(library.starts[clip_indices], np.int32)
        reset_frames = np.asarray(library.starts[(clip_indices+1) % len(names)], np.int32)
        for case in CASES:
            case_started = time.monotonic()
            case_root = root/case
            case_root.mkdir()
            frames = np.asarray(library.last[normal_frames]-1, np.int32) if case == 'clip_end' else normal_frames
            max_ep = 2 if case == 'timeout' else 500
            with torch.no_grad():
                start = tenv.reset(torch.as_tensor(frames, device='cuda'))
                reset = tenv.reset(torch.as_tensor(reset_frames, device='cuda'))
                if case == 'termination':
                    # Lift rather than penetrate the floor; guarantee the declared
                    # height termination without generating unsupported deep contacts.
                    start['qpos'][:, 2].add_(.5)
                    start = tenv.restore(tenv.compact(start))
                    start['hist'] = tenv.proprio(start)[:, None, :].repeat(1, 10, 1)
            js, jr = to_jax_state(start), to_jax_state(reset)
            record['cases'][case] = dict(frames=frames.tolist(), reset_frames=reset_frames.tolist(),
                                        clips=[names[i] for i in clip_indices], max_episode_steps=max_ep)
            save()
            np.savez(case_root/'shared_start.npz', **flat(start))
            np.savez(case_root/'shared_reset.npz', **flat(reset))
            vf = sim.vmapped(jenv, jenv.tables())
            jhist, jgoal = vf['policy_obs'](js, js.key)
            thist, tgoal = tenv.policy_obs(start)
            for name, jvalue, tvalue in (('history', jhist, thist), ('goal', jgoal, tgoal),
                                       ('critic_obs', vf['critic_obs'](js), tenv.critic_obs(start)),
                                       ('policy_action', jn.actor_apply(ja, jhist, jgoal), tn.actor_apply(ta, thist, tgoal))):
                compare(case+'.initial.'+name, jvalue, tvalue)
            rollout = make_rollout(jenv, HORIZON, GAMMA, max_ep, remat=False, act_clip=3, act_noise=0,
                                   mask_after_done=False, term_penalty=(2, .08, .10), record_pre=True,
                                   record_obs=True, enc_hoist=True)
            jfn = jax.jit(jax.value_and_grad(rollout, argnums=(8, 7), has_aux=True))
            (jloss, (_, jout)), (jgrad, jprobe) = jax.block_until_ready(jfn(
                ja, jc, norm, js, jr, jax.random.PRNGKey(SEED), jenv.tables(),
                jnp.zeros(2), jnp.zeros((HORIZON, BATCH, tenv.nu))))
            body = make_control_body(tenv, GAMMA, max_ep, 3, (2, .08, .10), enc_hoist=True)
            window = make_window(tenv, ta, tc, tnorm, HORIZON, GAMMA, body, encode)
            action_noise = torch.zeros((HORIZON, BATCH, tenv.nu), device='cuda')
            delta = torch.zeros_like(action_noise, requires_grad=True)
            probe = torch.zeros(2, device='cuda', requires_grad=True)
            tloss, tgrad, tprobe, tout = window(start, reset, action_noise,
                torch.zeros((2, HORIZON, BATCH, 93), device='cuda'),
                torch.zeros((HORIZON, BATCH, 6), device='cuda'), delta, probe)
            torch.cuda.synchronize()
            for name, jvalue, tvalue in (('loss', jloss, tloss), ('delta_gradient', jgrad, tgrad),
                                       ('clip_probe', jprobe/HORIZON, tprobe)):
                compare(case+'.'+name, jvalue, tvalue)
            compare_tree(case+'.outputs', jout, tout)
            compare(case+'.td_lambda', jn.td_lambda(jout['r'], jout['term'], jout['done'], jout['v2'], GAMMA, .95),
                    tn.td_lambda(tout['r'], tout['term'], tout['done'], tout['v2'], GAMMA, .95))
            for backend, loss, grad, stats, outs in (('jax', jloss, jgrad, jprobe/HORIZON, jout),
                                                    ('torch', tloss, tgrad, tprobe, tout)):
                np.savez(case_root/(backend+'_window.npz'), loss=cpu(loss), delta_gradient=cpu(grad),
                         clip_probe=cpu(stats), **flat(outs))
            done, term = cpu(jout['done']), cpu(jout['term'])
            if case == 'timeout':
                check(case+'.boundary_exercised', bool(done[1].all() and done[3].all() and not term.any()))
            elif case == 'clip_end':
                check(case+'.boundary_exercised', bool(done[0].all() and not term[0].any()))
            elif case == 'termination':
                check(case+'.boundary_exercised', bool(term[0].all()))
            record['cases'][case].update(seconds=time.monotonic()-case_started,
                done_count=int(done.sum()), termination_count=int(term.sum()))
            diagnostics(case+'.native_diagnostics')
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
        raise SystemExit('Failed paired training checks: '+', '.join(record['failures']))


if __name__ == '__main__':
    main()
