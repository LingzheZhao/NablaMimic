"""CPU policy-history recurrence with fixed non-action inputs, without simulation.

The synthetic fixed inputs isolate feedback sensitivity. This diagnostic does
not estimate executed policy performance or identify the original rare state.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def mlp(layers, value):
    for weight, bias in layers[:-1]:
        value = value @ weight + bias
        negative = np.minimum(value, 0)
        value = np.maximum(value, 0) + np.expm1(negative)
    weight, bias = layers[-1]
    return value @ weight + bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    actor = checkpoint['actor']
    latent = np.tanh(mlp(actor['enc'], np.zeros((1, actor['enc'][0][0].shape[0]), np.float32)))
    decoder_width = actor['dec'][0][0].shape[0]
    history_width = decoder_width-latent.shape[1]-6
    traces = {}
    for mode in ('raw', 'source_action_limit20'):
        history = np.zeros((10, 93), np.float32)
        history[:, 64:] = np.random.default_rng(0).normal(0, .01, (10, 29))
        scan = np.zeros((1, history_width-930), np.float32)
        rows = []
        for step in range(args.steps):
            inputs = np.concatenate((latent, history.reshape(1, -1), scan, np.zeros((1, 6), np.float32)), -1)
            action = mlp(actor['dec'], inputs)[0]
            feedback = action if mode == 'raw' else np.clip(action, -20., 20.)
            rows.append(dict(step=step, raw_policy_action_absolute_max=float(abs(action).max()),
                             feedback_absolute_max=float(abs(feedback).max())))
            if not np.isfinite(action).all():
                break
            fresh = np.zeros(93, np.float32)
            fresh[64:] = feedback
            history = np.concatenate((history[1:], fresh[None]), 0)
        traces[mode] = rows
    (args.output/'result.json').write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint),
        fixed_other_inputs='zero proprioception, reference features and scan',
        source_action_limit=20., traces=traces), indent=2))


if __name__ == '__main__':
    main()
