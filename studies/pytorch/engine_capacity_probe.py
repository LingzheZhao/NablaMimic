"""Independent old/new capacity comparison and actual-contact stress qualification.

Run each phase in a separate provisioned process, with the declared Engine source
on PYTHONPATH. A process never loads both native libraries. The prepare phase is
CPU-only; native/stress phases require the provisioned GPU. No failed input is
removed from saved outputs. Artificial deep contacts test capacity, not tracking.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


INPUTS = ('qpos', 'qvel', 'warm', 'ctrl', 'friction', 'com')
OUTPUTS = ('qpos1', 'qvel1', 'warm1', 'info', 'tape', 'qpos_n', 'xpos', 'xquat', 'cvel', 'subtree_com')
COTANGENTS = ('qpos1', 'qvel1', 'qpos_n', 'xpos', 'xquat', 'cvel', 'subtree_com')
BAD = ('fwd_unsupported_worlds', 'fwd_contact_overflow_worlds', 'fwd_row_overflow_worlds',
       'nonfinite_rows', 'nonfinite_cotangent_rows', 'replay_mismatch_worlds')
RTOL, ATOL = 1e-4, 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'native', 'compare', 'stress'), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine-capacity-revision', required=True)
    parser.add_argument('--engine-root', type=Path)
    parser.add_argument('--reference-cache', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--expected-contacts', type=int, choices=(64, 96), default=96)
    parser.add_argument('--expected-rows', type=int, choices=(288, 416), default=416)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--stress-profile', choices=('original', 'expanded'), default='original',
                        help='Separate fixed stress conditions; expanded adds outside-limit random hinges and '
                             'a foot-overlap family. Prior condition results are never changed.')
    args = parser.parse_args()
    if os.environ.get('DK_ALLOW_UNSUPPORTED') or os.environ.get('DK_MAXCON'):
        parser.error('Qualification requires declared defaults without unsupported/capacity overrides')
    sys.path.insert(0, str(args.source_root.resolve() / 'studies' / 'pytorch'))
    from engine_capacity_check import load_model, write
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.phase == 'stress' else 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if args.reference_cache:
        os.environ['MOTION_CACHE_DIR'] = str(args.reference_cache.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['DK_CACHE_DIR'] = str(args.out / 'native-cache')
    result = dict(status='running', phase=args.phase, engine_capacity_revision=args.engine_capacity_revision,
                  source_revision=args.source_revision,
                  source_root=str(args.source_root.resolve()), started_utc=datetime.now(timezone.utc).isoformat(),
                  checks=[], failures=[], protocol=dict(rtol=RTOL, atol=ATOL, finite_required=True,
                  discrete_contact_order_exact=True,
                  same_build_ffi_and_graph_scope='Separate mandatory engine_capacity_interfaces.py stage',
                  baseline_substeps=4, stress_substeps=1, iterations=10, clamp_kv=0, normal=3,
                  stress_profile=args.stress_profile,
                  stress_pose_count=1345 if args.stress_profile == 'expanded' else 160,
                  stress_height_count=32, stress_seed=20261103,
                  stress_chunk_worlds=1024 if args.stress_profile == 'expanded' else None,
                  stress_height_range=[-2., .9] if args.stress_profile == 'expanded' else [-1., .9],
                  stress_joint_definition='1024 uniform[-pi,pi] hinge poses plus321 opposing hip/ankle-roll poses'
                      if args.stress_profile == 'expanded' else '80 reference poses plus80 uniform-within-limit poses',
                  natural_stress_min_valid_above64=1, natural_stress_min_raw_over96=1,
                  natural_stress_exact96_required=args.stress_profile == 'expanded',
                  stress_selection='First valid state of each distinct contact count65..96; all pool rows retained',
                  scope='Capacity qualification; does not establish96contacts sufficient for every training state'))
    started = time.monotonic()

    def save():
        write(args.out / 'result.json', result)

    def check(name, passed, **fields):
        result['checks'].append(dict(name=name, passed=bool(passed), **fields))
        if not passed:
            result['failures'].append(name)
        save()

    def require(name, passed, **fields):
        check(name, passed, **fields)
        if not passed:
            raise AssertionError(name)

    save()
    try:
        import numpy as np

        def compare(name, a, b, exact=False):
            a, b = np.asarray(a), np.asarray(b)
            same = a.shape == b.shape and a.dtype == b.dtype
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            ok = same and finite and (a.tobytes() == b.tobytes() if exact else np.allclose(a, b, rtol=RTOL, atol=ATOL))
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64)) if same and finite else None
            check(name, ok, exact=exact, finite=finite, shape_baseline=list(a.shape), shape_candidate=list(b.shape),
                  max_abs_difference=float(difference.max()) if difference is not None and difference.size else None)

        if args.phase == 'compare':
            require('comparison_paths', args.baseline is not None and args.candidate is not None)
            base_meta = json.loads((args.baseline / 'result.json').read_text())
            new_meta = json.loads((args.candidate / 'result.json').read_text())
            result['baseline_engine_revision'] = base_meta['engine_capacity_revision']
            require('both_native_stages_passed', base_meta['status'] == new_meta['status'] == 'passed')
            require('candidate_revision', new_meta['engine_capacity_revision'] == args.engine_capacity_revision)
            with np.load(args.baseline / 'native.npz') as old, np.load(args.candidate / 'native.npz') as new:
                for key in (*INPUTS, *('cot_' + name for name in COTANGENTS), 'model_constants'):
                    compare('identical_' + key, old[key], new[key], exact=True)
                for key in (*COTANGENTS, 'warm1', 'grad_qpos', 'grad_qvel', 'grad_ctrl',
                            'decoded_qpos', 'decoded_qvel', 'decoded_qacc', 'decoded_qacc_adv',
                            'decoded_L_H', 'decoded_L_Q'):
                    compare('continuous_' + key, old[key], new[key])
                for key in ('ncon', 'nefc', 'nlim', 'clamp_dof', 'unsupported'):
                    compare('discrete_' + key, old['decoded_' + key], new['decoded_' + key], exact=True)
                ncon, nefc = old['decoded_ncon'], old['decoded_nefc']
                for index in np.ndindex(ncon.shape):
                    label = '_'.join(map(str, index))
                    count, rows = int(ncon[index]), int(nefc[index])
                    for key, length in (('pair', count), ('slot', count), ('zone', rows)):
                        compare('active_' + key + '_' + label, old['decoded_' + key][index][:length],
                                new['decoded_' + key][index][:length], exact=True)
                result['solver_iteration_counts_identical'] = bool(np.array_equal(old['info'][:, 0], new['info'][:, 0]))
            result['status'] = 'failed' if result['failures'] else 'passed'
            if result['failures']:
                raise SystemExit(1)
            return

        model, feet = load_model()
        result['model'] = dict(nq=model.nq, nv=model.nv, nu=model.nu, nbody=model.nbody, pairs=model.npair)
        require('current_model', (model.nq, model.nv, model.nu, model.nbody, model.npair) == (36, 35, 29, 31, 502))
        if args.phase == 'prepare':
            require('reference_cache_given', args.reference_cache is not None)
            spec = importlib.util.spec_from_file_location('donor_native_check', args.source_root / 'studies/pytorch/engine_check.py')
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            arrays, rows = helper.host_inputs(args.reference_cache, model)
            initial = dict(zip(INPUTS, arrays))
            random = np.random.default_rng(910)
            shapes = ((80, 36), (80, 35), (80, 36), (80, 31, 3), (80, 31, 4), (80, 31, 6), (80, 31, 3))
            initial.update({'cot_' + key: random.standard_normal(shape).astype(np.float32) * .01
                            for key, shape in zip(COTANGENTS, shapes)})
            require('eighty_reference_inputs', len(initial['qpos']) == 80)
            np.savez(args.out / 'valid_inputs.npz', **initial)
            random = np.random.default_rng(20261103)
            if args.stress_profile == 'original':
                poses = np.concatenate([initial['qpos'], initial['qpos'].copy()])
                limits = np.asarray([model.jnt_range[model.actuator_trnid[i, 0]] for i in range(model.nu)])
                poses[80:, 7:] = random.uniform(limits[:, 0], limits[:, 1], (80, model.nu))
                quat = random.standard_normal((80, 4))
                poses[80:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
                heights = np.linspace(-1.0, .9, 32, dtype=np.float32)
                families = np.concatenate([np.zeros(80, np.int32), np.ones(80, np.int32)])
                family_names = ['reference', 'random_within_joint_limits']
            else:
                poses = np.zeros((1345, model.nq), np.float32)
                poses[:, 3] = 1.
                poses[:1024, 7:] = random.uniform(-np.pi, np.pi, (1024, model.nu))
                quat = random.standard_normal((1024, 4))
                poses[:1024, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
                angles = np.linspace(-.8, .8, 321, dtype=np.float32)
                for name, sign in (('left_hip_roll_joint', -1), ('right_hip_roll_joint', 1),
                                   ('left_ankle_roll_joint', 1), ('right_ankle_roll_joint', -1)):
                    joint = model.joint(name).id
                    poses[1024:, model.jnt_qposadr[joint]] = sign * angles
                heights = np.linspace(-2.0, .9, 32, dtype=np.float32)
                families = np.concatenate([np.zeros(1024, np.int32), np.ones(321, np.int32)])
                family_names = ['random_hinges_minus_pi_to_pi', 'opposing_hip_ankle_roll_grid']
            qpos = np.repeat(poses, len(heights), axis=0)
            qpos[:, 2] = np.tile(heights, len(poses))
            count = len(qpos)
            stress = dict(qpos=qpos, qvel=np.zeros((count, 35), np.float32), warm=np.zeros((count, 35), np.float32),
                          ctrl=np.clip(qpos[:, 7:], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]).astype(np.float32),
                          friction=np.ones(count, np.float32), com=np.zeros((count, 3), np.float32))
            np.savez(args.out / 'stress_inputs.npz', **stress)
            np.savez(args.out / 'stress_labels.npz', pose=np.repeat(np.arange(len(poses)), len(heights)),
                     family=np.repeat(families, len(heights)), height=np.tile(heights, len(poses)))
            result.update(reference_rows=rows, stress_worlds=count, stress_heights=heights.tolist(),
                          stress_families=family_names, status='passed')
            return

        require('native_input_paths', args.inputs is not None and args.engine_root is not None)
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step, tape
        require('engine_source_location', Path(tf.__file__).resolve().is_relative_to(args.engine_root.resolve()))
        require('one_cuda_device', torch.cuda.is_available() and torch.cuda.device_count() == 1)
        torch.cuda.set_device(0)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require('declared_capacities', step.maxcon(model) == args.expected_contacts and step.NEFC == args.expected_rows)
        info = tf.init(model=model, foot_pairs=feet)
        result['native'] = {key: info[key] for key in ('arch', 'rec_words', 'maxcon', 'version', 'nq', 'nv', 'nu', 'nbody')}
        tf.set_diagnostics(True)
        host = dict(np.load(args.inputs, allow_pickle=False))
        if args.phase == 'stress':
            prepare = json.loads((args.inputs.parent / 'result.json').read_text())
            require('stress_input_protocol_identity', prepare['status'] == 'passed' and
                    prepare['source_revision'] == args.source_revision and
                    prepare['engine_capacity_revision'] == args.engine_capacity_revision and
                    prepare['protocol']['stress_profile'] == args.stress_profile and
                    len(host['qpos']) == result['protocol']['stress_pose_count'] * 32)
        tensor = lambda value: torch.from_numpy(np.ascontiguousarray(value)).to('cuda')
        cpu = lambda value: value.detach().cpu().numpy()
        inputs = [tensor(host[key]) for key in INPUTS]
        substeps = 1 if args.phase == 'stress' else 4
        if args.phase == 'stress' and args.stress_profile == 'expanded':
            pieces = {key: [] for key in OUTPUTS}
            chunk_diagnostics = []
            for start in range(0, len(host['qpos']), 1024):
                stop = min(start + 1024, len(host['qpos']))
                forward = tf.forward(*(value[start:stop] for value in inputs),
                                     iterations=10, substeps=1, tape=True, kinematics=True)
                torch.cuda.synchronize()
                for key, value in zip(OUTPUTS, forward):
                    pieces[key].append(cpu(value))
                chunk_diagnostics.append(dict(start=start, stop=stop, counters=tf.read_diagnostics(reset=True)))
            values = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
            result['pool_chunk_diagnostics'] = chunk_diagnostics
            result['pool_diagnostics_sum'] = {key: sum(chunk['counters'][key] for chunk in chunk_diagnostics)
                                              for key in tf.DIAG}
        else:
            forward = tf.forward(*inputs, iterations=10, substeps=substeps, tape=True, kinematics=True)
            torch.cuda.synchronize()
            values = dict(zip(OUTPUTS, map(cpu, forward)))
        decoded = tape.decode(values['tape'].reshape(-1, substeps, info['rec_words']), model)
        values.update({'decoded_' + key: value for key, value in decoded.items()})
        values.update({key: host[key] for key in INPUTS})
        values['model_constants'] = tf._STATE['consts'].copy()
        np.savez(args.out / ('pool.npz' if args.phase == 'stress' else 'forward.npz'), **values)
        if args.phase == 'stress':
            require('candidate_stress_capacity', args.expected_contacts == 96 and args.expected_rows == 416)
            raw, flags = values['info'][:, 1], values['info'][:, 3]
            result['raw_contact_histogram'] = {str(int(k)): int(v) for k, v in zip(*np.unique(raw, return_counts=True))}
            check('contact_overflow_exact_trigger', np.array_equal((flags & 4) != 0, raw > 96))
            check('no_unexpected_unsupported_bits', np.all((flags & ~4) == 0))
            check('tape_contact_count_is_capped', np.array_equal(decoded['ncon'][:, 0], np.minimum(raw, 96)))
            require('natural_overflow_guard_exercised', np.any(raw > 96), worlds=int((raw > 96).sum()), raw_max=int(raw.max()))
            selected = [int(np.flatnonzero((raw == n) & (flags == 0))[0]) for n in range(65, 97)
                        if np.any((raw == n) & (flags == 0))]
            result['selected_valid_worlds'] = selected
            result['selected_valid_contact_counts'] = raw[selected].tolist()
            require('natural_above64_exercised', bool(selected))
            if args.stress_profile == 'expanded':
                require('natural_valid96_exercised', np.any((raw == 96) & (flags == 0)))
            # Intentional overflow rows are retained above, then excluded only
            # from the separate valid-world derivative/interface condition.
            host = {key: np.ascontiguousarray(host[key][selected]) for key in INPUTS}
            inputs = [tensor(host[key]) for key in INPUTS]
            tf.read_diagnostics(reset=True)
            forward = tf.forward(*inputs, iterations=10, substeps=1, tape=True, kinematics=True)
            values = dict(zip(OUTPUTS, map(cpu, forward)))
            decoded = tape.decode(values['tape'].reshape(-1, 1, info['rec_words']), model)
            values.update({'decoded_' + key: value for key, value in decoded.items()})
            values.update(host)
            random = np.random.default_rng(910)
            for name, output in zip(COTANGENTS, (forward[0], forward[1], *forward[5:])):
                host['cot_' + name] = random.standard_normal(tuple(output.shape)).astype(np.float32) * .01
        for name in COTANGENTS + ('warm1',):
            check('finite_' + name, np.isfinite(values[name]).all())
        cots = [tensor(host['cot_' + name]) for name in COTANGENTS]
        backward = tf.backward(forward[4], inputs[4], inputs[5], cots[0], cots[1], substeps=substeps,
                               clamp_kv=0, normal=3, kinematics=(forward[0], forward[1], *cots[2:]))
        torch.cuda.synchronize()
        for key, value in zip(('grad_qpos', 'grad_qvel', 'grad_ctrl', 'backward_info'), backward):
            values[key] = cpu(value)
            check('finite_' + key, np.isfinite(values[key]).all())
        values.update({'cot_' + name: host['cot_' + name] for name in COTANGENTS})
        diagnostic = tf.read_diagnostics(reset=True)
        check('valid_native_diagnostics', all(diagnostic[key] == 0 for key in BAD), diagnostics=diagnostic)
        values.setdefault('model_constants', tf._STATE['consts'].copy())
        np.savez(args.out / ('selected.npz' if args.phase == 'stress' else 'native.npz'), **values)
        result['status'] = 'failed' if result['failures'] else 'passed'
    except BaseException as error:
        result.update(status='failed', exception=repr(error))
        raise
    finally:
        result['elapsed_s'] = time.monotonic() - started
        save()
    if result['failures']:
        raise SystemExit('Capacity qualification failed; inspect retained result and arrays')


if __name__ == '__main__':
    main()
