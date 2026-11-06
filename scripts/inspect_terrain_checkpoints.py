"""CPU summaries of saved current-format policy, critic and normalization arrays."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def leaves(value, name=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, name+'.'+key)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from leaves(item, name+f'[{index}]')
    else:
        yield name, np.asarray(value)


def summary(array):
    array = np.asarray(array)
    finite = np.isfinite(array)
    values = array[finite].astype(np.float64)
    return dict(shape=list(array.shape), nonfinite=int((~finite).sum()),
                absolute_max=float(abs(values).max()) if values.size else None,
                rms=float(np.sqrt(np.mean(values**2))) if values.size else None,
                minimum=float(values.min()) if values.size else None,
                maximum=float(values.max()) if values.size else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    previous = None
    for path in args.checkpoint:
        with path.open('rb') as handle:
            checkpoint = pickle.load(handle)
        report = dict(path=str(path), completed_updates=checkpoint['completed_updates'], tensors={})
        for key in ('actor', 'actor_ema', 'critic', 'cnorm', 'bin_q'):
            for name, array in leaves(checkpoint[key], key):
                report['tensors'][name] = summary(array)
        mean, variance = checkpoint['cnorm']
        report['critic_variance'] = dict(nonpositive=int((variance <= 0).sum()),
            smallest_indices=np.argsort(variance)[:20].tolist(),
            largest_indices=np.argsort(variance)[-20:].tolist(),
            quantiles=np.quantile(variance, [0,.01,.5,.99,1]).tolist())
        if previous is not None:
            old = dict(leaves(previous))
            report['changes'] = {name: summary(array-old[name]) for name, array in leaves(
                {key: checkpoint[key] for key in ('actor','actor_ema','critic','cnorm')})}
        previous = {key: checkpoint[key] for key in ('actor','actor_ema','critic','cnorm')}
        reports.append(report)
    (args.output/'result.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
