"""Compare saved serial/batched chains without rerunning physical execution."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--absolute-tolerance', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    serial_paths = sorted(args.serial.glob('*/trajectory.npz'))
    batch_paths = sorted(args.batch.glob('*/trajectory.npz'))
    if [path.parent.name for path in serial_paths] != [path.parent.name for path in batch_paths]:
        raise ValueError('Compared panels have different chain identities')
    for first_path, second_path in zip(serial_paths, batch_paths):
        condition = dict(chain=first_path.parent.name, arrays={})
        with np.load(first_path) as first, np.load(second_path) as second:
            if set(first.files) != set(second.files):
                raise ValueError('Compared chains have different recorded arrays')
            for key in first.files:
                a, b = first[key], second[key]
                if a.shape != b.shape:
                    condition['arrays'][key] = dict(matches=False, serial_shape=list(a.shape), batch_shape=list(b.shape))
                else:
                    error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.0
                    condition['arrays'][key] = dict(matches=bool(np.allclose(a, b, rtol=0, atol=args.absolute_tolerance)), max_absolute_error=error)
        condition['matches'] = all(item['matches'] for item in condition['arrays'].values())
        results.append(condition)
    (args.output/'result.json').write_text(json.dumps(dict(
        absolute_tolerance=args.absolute_tolerance, chains=results,
        matches=bool(results) and all(item['matches'] for item in results)), indent=2))


if __name__ == '__main__':
    main()
