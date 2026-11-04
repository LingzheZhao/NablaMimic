"""Check saved three-child subtree sums against all six sequential FP32 orders.

Inputs are matching float32 NumPy arrays [world, body, xyz] immediately before
and after the declared parent accumulation. This CPU-only study reads arrays;
it never initializes a simulator or changes a kernel.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_bits(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(
        np.ascontiguousarray(left).view(np.uint32), np.ascontiguousarray(right).view(np.uint32))


def analyze_pair(before, after, parent, children):
    """Coordinate-wise membership, with a float32 rounding after every add."""
    require(before.dtype == after.dtype == np.dtype('float32')
            and before.shape == after.shape and before.ndim == 3
            and before.shape[0] > 0 and before.shape[2] == 3, 'Expected matching native float32 [world, body, 3] arrays')
    require(len(children) == 3 and len(set((parent, *children))) == 4
            and all(0 <= index < before.shape[1] for index in (parent, *children)),
            'Declare one parent and three distinct children within the saved body axis')
    require(np.isfinite(before).all() and np.isfinite(after).all(), 'Finite saved layer arrays required')
    unchanged = [index for index in range(before.shape[1]) if index != parent]
    other_bodies_equal = equal_bits(before[:, unchanged], after[:, unchanged])
    orders = list(itertools.permutations(children))
    candidates = []
    subnormal_intermediates = 0
    tiny = np.finfo(np.float32).tiny
    for order in orders:
        value = before[:, parent].copy()
        for child in order:
            with np.errstate(over='raise', invalid='raise'):
                value = np.add(value, before[:, child], dtype=np.float32)
            subnormal_intermediates += int(np.count_nonzero((value != 0) & (np.abs(value) < tiny)))
        candidates.append(value)
    candidates = np.stack(candidates)
    observed = after[:, parent]
    candidate_bits = candidates.view(np.uint32)
    observed_bits = np.ascontiguousarray(observed).view(np.uint32)
    matches = candidate_bits == observed_bits[None]
    coordinate_matches = matches.any(axis=0)
    same_order_for_xyz = matches.all(axis=2).any(axis=0)
    selected = before[:, [parent, *children]]
    subnormal_inputs = int(np.count_nonzero((selected != 0) & (np.abs(selected) < tiny)))
    coordinates = []
    for world in range(before.shape[0]):
        for axis in range(3):
            coordinates.append(dict(world=world, axis='xyz'[axis],
                initial_parent=float(before[world, parent, axis]),
                children=[float(before[world, child, axis]) for child in children],
                observed=float(observed[world, axis]), observed_bits=f'{int(observed_bits[world, axis]):08x}',
                candidate_values=[float(value) for value in candidates[:, world, axis]],
                candidate_bits=[f'{int(value):08x}' for value in candidate_bits[:, world, axis]],
                matching_order_indices=np.flatnonzero(matches[:, world, axis]).tolist()))
    return dict(shape=list(before.shape), parent=parent, children=list(children),
        orders=[list(order) for order in orders], coordinates=coordinates,
        matching_coordinates=int(coordinate_matches.sum()), total_coordinates=int(coordinate_matches.size),
        all_coordinates_match=bool(coordinate_matches.all()), other_bodies_bitwise_unchanged=other_bodies_equal,
        worlds_without_one_common_xyz_order=int(np.count_nonzero(~same_order_for_xyz)),
        subnormal_inputs=subnormal_inputs, subnormal_intermediate_occurrences=subnormal_intermediates,
        interpretation='Each coordinate may match a different child order. Membership neither identifies an actual GPU thread schedule nor attributes a full trajectory or return difference.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, nargs='+', required=True)
    parser.add_argument('--after', type=Path, nargs='+', required=True)
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--children', type=int, nargs=3, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    require(len(args.before) == len(args.after), 'Before/after lists must have equal length and paired order')
    inputs = [path.resolve() for path in (*args.before, *args.after)]
    out = args.out.resolve()
    require(not out.exists() and all(not path.is_relative_to(out) for path in inputs),
            'A fresh output directory separate from all inputs is required')
    require(all(path.is_file() for path in inputs), 'All declared layer snapshots must exist')
    out.mkdir(parents=True)
    record = dict(status='running', source_revision=args.source_revision, numpy_version=np.__version__,
        parent=args.parent, children=args.children, readings=[], all_inputs_bitwise_equal=True,
        scope='CPU order enumeration over saved layer arrays only; no simulation, GPU work or evaluator changes')
    try:
        # Reject a changed host rounding mode instead of silently interpreting it as FP32/RNE.
        anchor = np.float32(2**24)
        require(np.add(anchor, np.float32(1), dtype=np.float32) == anchor
                and np.add(anchor, np.float32(3), dtype=np.float32) == np.float32(2**24+4),
                'Expected round-to-nearest, ties-to-even float32 addition')
        initial = None
        observed_bits = []
        for index, (left, right) in enumerate(zip(args.before, args.after)):
            before = np.load(left, allow_pickle=False)
            after = np.load(right, allow_pickle=False)
            result = analyze_pair(before, after, args.parent, args.children)
            if initial is None:
                initial = before.copy()
            same = equal_bits(initial, before)
            record['all_inputs_bitwise_equal'] &= same
            observed_bits.append(after[:, args.parent].copy().view(np.uint32))
            record['readings'].append(dict(index=index, before=str(left.resolve()), after=str(right.resolve()),
                same_before_as_first=same, **result))
        record['observed_varying_coordinates'] = int(np.any(
            np.stack(observed_bits) != observed_bits[0][None], axis=0).sum())
        valid = record['all_inputs_bitwise_equal'] and all(
            row['all_coordinates_match'] and row['other_bodies_bitwise_unchanged']
            and row['subnormal_inputs'] == row['subnormal_intermediate_occurrences'] == 0
            for row in record['readings'])
        record['status'] = 'completed_matching' if valid else 'completed_with_mismatch'
        record['limitation'] = ('Subnormal inputs/intermediates require a separate GPU atomic-flush analysis. '
            'No float64 accumulation, shared XYZ permutation, schedule identification, or numerical amplification factor is inferred.')
    except BaseException as error:
        record.update(status='failed', exception=repr(error))
        raise
    finally:
        (out/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return 0 if record['status'] == 'completed_matching' else 1


if __name__ == '__main__':
    sys.exit(main())
