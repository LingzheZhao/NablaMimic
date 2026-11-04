"""Portable forward-only checks of the COM range-sum backport.

Run on a provisioned CUDA device. No model, simulator, checkpoint, scheduler or
private-harness dependency is needed. The synthetic zero-mass cases establish
only the declared forward COM formula, not zero-mass adjoint support.
"""
import argparse
import json
from pathlib import Path

RTOL, ATOL = 1e-5, 1e-6


def require(value, message):
    if not value:
        raise ValueError(message)


def descendants(parent):
    """Independent ancestor walks; no reliance on uploaded subtree lengths."""
    require(parent and parent[0] == 0, 'World body must be its own parent')
    groups = [[] for _ in parent]
    for child in range(len(parent)):
        seen, node = set(), child
        while True:
            require(node not in seen and 0 <= node < len(parent), 'Invalid/cyclic body tree')
            seen.add(node)
            groups[node].append(child)
            if node == 0:
                break
            require(parent[node] < node, 'Parent must precede child')
            node = parent[node]
    require(all(group == list(range(body, body+len(group))) for body, group in enumerate(groups)),
            'Body numbering must have contiguous DFS subtrees')
    return groups


def fp64_references(mass, xipos, stored_mass, parent):
    """Two independent sums from actual FP32 inputs; preserve the old mass branch."""
    import numpy as np
    require(mass.dtype == xipos.dtype == stored_mass.dtype == np.float32, 'Actual model/kinematic FP32 inputs required')
    require(mass.ndim == stored_mass.ndim == 2 and xipos.ndim == 3 and xipos.shape[2] == 3,
            'Expected mass [batch, body] and xipos [world, body, xyz]')
    require(mass.shape[1] == stored_mass.shape[1] == xipos.shape[1] == len(parent), 'Body axes differ')
    require(all(np.isfinite(value).all() for value in (mass, xipos, stored_mass)), 'Finite input arrays required')
    require(np.all(mass >= 0) and np.all(stored_mass >= 0), 'Nonnegative physical masses required')
    groups = descendants(parent)
    independent, stored = np.empty(xipos.shape, np.float64), np.empty(xipos.shape, np.float64)
    for world in range(xipos.shape[0]):
        for body, group in enumerate(groups):
            weights = mass[world % mass.shape[0], group].astype(np.float64)
            numerator = np.sum(weights[:, None]*xipos[world, group].astype(np.float64), axis=0, dtype=np.float64)
            total = float(np.sum(weights, dtype=np.float64))
            old_total = float(stored_mass[world % stored_mass.shape[0], body])
            independent[world, body] = numerator/total if total != 0 else numerator
            stored[world, body] = numerator/old_total if old_total != 0 else numerator
    return {'independent_mass_sum': independent, 'stored_subtree_mass': stored}


def reference_error(reference, actual):
    import numpy as np
    require(reference.dtype == np.float64 and actual.dtype == np.float32 and reference.shape == actual.shape,
            'Reference/actual layouts differ')
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    if not finite:
        return dict(passed=False, finite=False, rtol=RTOL, atol=ATOL)
    error = np.abs(actual.astype(np.float64)-reference)
    outside = error > ATOL+RTOL*np.abs(reference)
    where = np.argwhere(outside)
    return dict(passed=not bool(outside.any()), finite=True, rtol=RTOL, atol=ATOL,
                max_abs_error_m=float(error.max(initial=0)), outside_tolerance=int(outside.sum()),
                first_outside_tolerance=where[0].tolist() if len(where) else None,
                world_body_max_abs_error_m=float(error[:, 0].max(initial=0)),
                nonworld_max_abs_error_m=float(error[:, 1:].max(initial=0)))


def run_synthetic(out):
    import numpy as np
    import warp as wp
    from mujoco_warp._src import smooth
    out = Path(out)
    require(not out.exists(), 'Fresh synthetic output directory required')
    out.mkdir(parents=True)
    record = dict(status='running', cases=[], rtol=RTOL, atol=ATOL,
                  smooth_source=str(Path(smooth.__file__).resolve()), warp_version=wp.__version__,
                  scope='Forward COM kernels only. Fixed branching/multiple-root/zero/tiny-positive masses; no adjoint claim.')
    try:
        require(hasattr(smooth, '_subtree_com') and hasattr(smooth, '_subtree_com_world'), 'COM range-sum kernels required')
        device = wp.get_device('cuda:0')
        parent = [0, 0, 1, 1, 0, 4, 4, 0, 7]
        groups = descendants(parent)
        counts = np.asarray([len(group) for group in groups], dtype=np.int32)
        mass = np.asarray([[0, 3, 1, 2, 4, 2, 1, 0, 0]], np.float32)
        tiny = mass.copy(); tiny[0, 7:] = np.asarray([1e-20, 3e-20], np.float32)
        world_mass = mass.copy(); world_mass[0, 0] = np.float32(.5)
        # Every positive mass remains normal, avoiding an unrelated subnormal
        # or flush-to-zero experiment.
        all_tiny = np.asarray([[0, 3e-20, 1e-20, 2e-20, 4e-20, 2e-20, 1e-20, 1e-20, 3e-20]], np.float32)
        cases = (('branching_multiple_roots_zero_subtree', mass),
                 ('nonzero_world_own_mass', world_mass), ('tiny_positive_subtree', tiny),
                 ('all_zero_mass', np.zeros_like(mass)), ('tiny_positive_total_mass', all_tiny))
        xipos = np.asarray(np.arange(54).reshape(2, 9, 3)/17.0-1.0, np.float32)
        for name, masses in cases:
            folder = out/name; folder.mkdir()
            stored = np.asarray([[np.sum(masses[0, group].astype(np.float64)) for group in groups]], np.float32)
            references = fp64_references(masses, xipos, stored, parent)
            for key, value in dict(mass=masses, stored_mass=stored, xipos=xipos, parent=np.asarray(parent, np.int32),
                                   counts=counts, **references).items():
                np.save(folder/(key+'.npy'), value, allow_pickle=False)
            wm = wp.array(masses, dtype=float, device=device)
            ws = wp.array(stored, dtype=float, device=device)
            wc = wp.array(counts, dtype=int, device=device)
            wx = wp.array(xipos, dtype=wp.vec3, device=device)
            sentinel = wp.array(np.full(xipos.shape, 123.5, np.float32), dtype=wp.vec3, device=device)
            output = wp.empty((2, 9), dtype=wp.vec3, device=device)
            # Module loading must precede capture. This first forward is setup.
            def launch():
                wp.launch(smooth._subtree_com, dim=(2, 9), inputs=[wm, ws, wc, wx], outputs=[output])
                wp.launch(smooth._subtree_com_world, dim=2, inputs=[9, wm, ws, wc, wx, output], outputs=[output])
            wp.copy(output, sentinel); launch(); wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                launch()
            first, readings = None, []
            for index in range(16):
                wp.copy(output, sentinel); wp.capture_launch(capture.graph); wp.synchronize_device(device)
                actual = np.array(output.numpy(), copy=True)
                if first is None:
                    first = actual.copy()
                equal = actual.tobytes() == first.tobytes()
                errors = {key: reference_error(value, actual) for key, value in references.items()}
                zero = stored[0] == 0
                zero_raw = bool(np.all(actual[:, zero] == 0))
                readings.append(dict(repetition=index, bitwise_equal=equal, reference_errors=errors,
                                     zero_mass_raw_sum_preserved=zero_raw))
                np.save(folder/f'rep-{index:03d}.npy', actual, allow_pickle=False)
            unchanged = all(np.array_equal(array.numpy(), value) for array, value in
                            ((wm, masses), (ws, stored), (wc, counts), (wx, xipos)))
            passed = unchanged and all(row['bitwise_equal'] and row['zero_mass_raw_sum_preserved']
                                       and all(x['passed'] for x in row['reference_errors'].values()) for row in readings)
            record['cases'].append(dict(name=name, passed=passed, inputs_unchanged=unchanged, readings=readings))
        record['status'] = 'passed' if all(row['passed'] for row in record['cases']) else 'failed_checks'
    except BaseException as error:
        record.update(status='failed', exception=repr(error))
        raise
    finally:
        (out/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return 0 if run_synthetic(args.out)['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
