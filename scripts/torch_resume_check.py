"""Provisioned-GPU comparison of four updates against two plus two updates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = args.training_args
    if common[:1] == ['--']:
        common = common[1:]
    if any(flag in common for flag in ('--out', '--resume', '--iters')):
        raise ValueError('Comparison owns output roots, resume source and update counts')
    commands = [
        ('continuous', ['--iters', '4']),
        ('first', ['--iters', '2']),
        ('continued', ['--iters', '4', '--resume', str(args.output/'first/training_state_it000002.pt')]),
    ]
    for name, options in commands:
        command = [sys.executable, '-m', 'nablamimic.torch_train', *common, *options,
                   '--full-state-at-end', '--out', str(args.output/name)]
        with (args.output/f'{name}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (args.output/'result.json').write_text(json.dumps(dict(stage=name, returncode=result.returncode)))
            raise SystemExit(result.returncode)
    first = torch.load(args.output/'continuous/training_state_it000004.pt', map_location='cpu', weights_only=False)
    second = torch.load(args.output/'continued/training_state_it000004.pt', map_location='cpu', weights_only=False)
    differences = []
    def compare(a, b, path):
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                differences.append(path)
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                differences.append(path)
        elif isinstance(a, dict):
            if set(a) != set(b):
                differences.append(path+'.keys')
            else:
                for key in a:
                    compare(a[key], b[key], path+'.'+key)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                differences.append(path+'.length')
            else:
                for index, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, path+f'[{index}]')
        elif a != b:
            differences.append(path)
    for key in first:
        if key not in ('wall_s', 'resolved_config'):
            compare(first[key], second[key], key)
    (args.output/'result.json').write_text(json.dumps(dict(
        matches=not differences, differences=differences,
        comparison='all full-state tensors, counters and RNG exactly equal; wall time and launch control configuration excluded'), indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
