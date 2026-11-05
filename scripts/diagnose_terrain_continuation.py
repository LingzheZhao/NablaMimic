"""Record named continuation updates with the frozen trainer imported by PYTHONPATH.

Use source58's runtime for source58 full-state continuation. This script only
observes already computed tensors and saves explicitly selected updates.
"""
import argparse
from pathlib import Path
import sys

import torch
from nablamimic import torch_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--updates', type=int, nargs='+', required=True)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.diagnostics.mkdir(parents=True, exist_ok=False)
    selected = set(args.updates)
    original_update = torch_train.TorchTrainer.update
    original_window = torch_train.TorchTrainer.window_inputs
    def window_inputs(trainer):
        inputs = original_window(trainer)
        if trainer.completed_updates+1 in selected:
            trainer.diagnostic_window = torch_train.clone_tree(inputs)
        return inputs
    def update(trainer):
        metrics = original_update(trainer)
        if trainer.completed_updates in selected:
            torch.save(dict(metrics=metrics, window_inputs=trainer.diagnostic_window,
                training_data=trainer.training_data, target_data=trainer.target_data,
                critic_targets=trainer.critic_targets, cnorm=trainer.cnorm,
                actor=trainer.actor, critic=trainer.critic,
                actor_optimizer=trainer.actor_opt, critic_optimizer=trainer.critic_opt),
                args.diagnostics/f'update_{trainer.completed_updates:06d}.pt')
            del trainer.diagnostic_window
        return metrics
    torch_train.TorchTrainer.window_inputs = window_inputs
    torch_train.TorchTrainer.update = update
    command = args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args
    torch_train.main(command)


if __name__ == '__main__':
    main()
