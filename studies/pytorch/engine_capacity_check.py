"""CPU qualification of the declared 96-contact/416-row native Engine layout.

Static model and fresh tape fixtures only: no simulation or GPU initialization.
This is one prerequisite; it does not qualify contact dynamics or gradients.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_model():
    import mujoco
    import numpy as np
    from nablamimic.env import EnvConfig, build_model
    model, _ = build_model(EnvConfig(solver_iterations=10, residual_action=True))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    feet = np.asarray([p for p in range(model.npair)
                       if '_foot' in model.pair(p).name and model.pair(p).name.endswith('_floor')], np.int32)
    return model, feet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine-capacity-revision', required=True)
    args = parser.parse_args()
    if os.environ.get('DK_ALLOW_UNSUPPORTED') or os.environ.get('DK_MAXCON'):
        parser.error('Qualification requires the declared defaults without unsupported/capacity overrides')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(status='running', engine_capacity_revision=args.engine_capacity_revision,
                  checks=[], failures=[], scope=__doc__)

    def check(name, passed, **fields):
        result['checks'].append(dict(name=name, passed=bool(passed), **fields))
        if not passed:
            result['failures'].append(name)

    write(args.out/'result.json', result)
    try:
        import numpy as np
        from nablaengine.g1_engine import step, tape
        from nablaengine.model import capability
        model, feet = load_model()
        result['model'] = dict(nq=model.nq, nv=model.nv, nu=model.nu, nbody=model.nbody, pairs=model.npair,
                               collision_geoms=len(step.collision_geoms(model)), foot_pairs=feet.tolist(),
                               max_support=step.max_support(model))
        check('current_model', (model.nq, model.nv, model.nu, model.nbody, model.npair, len(feet)) == (36, 35, 29, 31, 502, 14))
        check('default_capacities', step.MAX_CONTACTS == 96 and step.NEFC == 416 and step.maxcon(model) == 96)
        check('small_model_default_retained', step.maxcon(SimpleNamespace(npair=64)) == 32)
        for value in (1, 31, 32, 63, 64, 65, 95, 96):
            check(f'valid_capacity_{value}', step.validate_maxcon(value) == value)
        for value in (-1, 0, 97, 127, 128, 1.5, '96', None):
            rejected = False
            try:
                step.validate_maxcon(value)
            except ValueError:
                rejected = True
            check(f'invalid_capacity_{value!r}', rejected)
        for value in (0, 97):
            rejected = False
            try:
                tape.layout(model, max_contacts=value)
            except ValueError:
                rejected = True
            check(f'tape_rejects_capacity_{value}', rejected)
        report = capability.check(model, feet, scope='engine')
        result['capability'] = report.to_dict()
        check('capability_accepts_current_model', report.ok())
        layout = tape.layout(model)
        result['tape_layout'] = {key: list(value) for key, value in layout.items()}
        check('tape_layout_96_416', tape.rec_words(model) == 1504 and layout['zone'][1] == 13 and
              layout['con'][1] == 48 and layout['L_H'][0] == 224 and layout['L_Q'][0] == 864)
        records = np.zeros((8, tape.rec_words(model)), np.float32)
        words = records.view(np.uint32)
        counts = np.asarray([0, 31, 32, 63, 64, 65, 95, 96], np.uint32)
        rows = np.asarray([0, 127, 128, 255, 256, 287, 415, 416], np.uint32)
        words[:, layout['hdr'][0]] = counts | (rows << 7) | (np.uint32(29) << 16) | (np.uint32(255) << 24)
        zone_indices = [0, 31, 32, 63, 64, 95, 96, 287, 288, 415]
        for bit in zone_indices:
            words[:, layout['zone'][0]+bit//32] |= np.uint32(1) << np.uint32(bit % 32)
        expected_pairs = np.arange(96, dtype=np.uint16)*5 % model.npair
        expected_pairs[-1] = 501
        expected_slots = np.arange(96, dtype=np.uint16) % 4
        packed = expected_pairs | (expected_slots << 9)
        for i in range(len(records)):
            field = words[i, layout['con'][0]:layout['con'][0]+layout['con'][1]].view(np.uint16)
            field[:] = np.uint16(0xffff)
            field[:counts[i]] = packed[:counts[i]]
        decoded = tape.decode(records, model)
        check('header_boundaries_roundtrip', np.array_equal(decoded['ncon'], counts) and
              np.array_equal(decoded['nefc'], rows) and np.all(decoded['nlim'] == 29) and np.all(decoded['niter'] == 255))
        expected_zones = np.zeros((8, 416), bool)
        expected_zones[:, zone_indices] = True
        check('all_thirteen_zone_words_roundtrip', np.array_equal(decoded['zone'], expected_zones))
        for i, count in enumerate(counts):
            check(f'contact_table_count_{count}', np.array_equal(decoded['pair'][i, :count], expected_pairs[:count]) and
                  np.array_equal(decoded['slot'][i, :count], expected_slots[:count]) and
                  np.all(decoded['pair'][i, count:] == -1) and np.all(decoded['slot'][i, count:] == -1))
        np.savez(args.out/'fresh_tape_fixtures.npz', tape=records, ncon=counts, nefc=rows,
                 expected_pairs=expected_pairs, expected_slots=expected_slots, expected_zones=expected_zones)
        header = step.step_header(model)
        (args.out/'generated_step_header.txt').write_text(header)
        check('generated_capacity_constants', '#define MAXCON 96' in header and '#define NEFC 416' in header and
              '#define CONTACT_TILES' in header)
        # These source guards supplement, not replace, the independent device
        # Jacobian/Hessian fixture required by the overall qualification protocol.
        hessian = step.STEP_KERNEL.split('void hessian(', 1)[1].split('// contact', 1)[0]
        check('contact_masks_are_tiled', 'act[CONTACT_TILES]' in hessian and 'world' in hessian and
              'unsigned long long act' not in hessian)
        check('overflow_guard_retained', 's.ncon > MAXCON' in step.STEP_KERNEL and
              's.unsupported |= 4' in step.STEP_KERNEL and 'c >= MAXCON' in step.STEP_DEVICE)
        result['status'] = 'failed' if result['failures'] else 'passed'
    except BaseException as error:
        result.update(status='failed', exception=repr(error))
        raise
    finally:
        write(args.out/'result.json', result)
    if result['failures']:
        raise SystemExit('CPU capacity qualification failed: '+', '.join(result['failures']))


if __name__ == '__main__':
    main()
