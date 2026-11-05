"""Decompose a newly sampled source training window without changing training rules."""
import argparse
import json
import pickle
from pathlib import Path

import torch


class IdentityCotangent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, action, probe):
        ctx.save_for_backward(probe)
        return action.clone()

    @staticmethod
    def backward(ctx, gradient):
        probe, = ctx.saved_tensors
        return gradient, torch.zeros_like(probe)


def gradient_reading(gradient, reference, action_scale):
    """Loss-action cotangent and its normalized ascent direction in setpoint units."""
    scaled = -gradient * action_scale
    reference_scaled = -reference * action_scale
    rms = scaled.square().mean().sqrt()
    denominator = scaled.norm() * reference_scaled.norm()
    cosine = (scaled * reference_scaled).sum() / denominator.clamp_min(1e-30)
    direction = scaled / rms.clamp_min(1e-30)
    return dict(loss_action_gradient_rms=float(gradient.square().mean().sqrt()),
                setpoint_gradient_rms=float(rms), cosine_with_full=float(cosine),
                normalized_setpoint_direction_rms=float(direction.square().mean().sqrt()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=('ema', 'live'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from nablamimic import torch_nets as nets
    from nablamimic.torch_train import TorchTrainer, build_environment

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = json.loads(args.config.read_text())
    config.update(execution='eager', num_envs=128, horizon=32, out=str(args.output))
    for name in ('terrain_model', 'terrain_model_contract'):
        config[name] = Path(config[name])
    (args.output/'resolved_config.json').write_text(json.dumps(
        {name:str(value) if isinstance(value,Path) else value for name,value in config.items()},indent=2))
    options = argparse.Namespace(**config)
    env, bindings = build_environment(options)
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
        raise ValueError('Checkpoint does not match the constructed observation and scene')
    actor = nets.tree_from_jax(checkpoint['actor_ema' if args.weights == 'ema' else 'actor'], env.device)
    critic = nets.tree_from_jax(checkpoint['critic'], env.device)
    trainer = TorchTrainer(env, actor, critic, options)
    nets.copy_tree_(trainer.cnorm, nets.tree_from_jax(checkpoint['cnorm'], env.device))
    sampled = trainer.window_inputs()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    reward = env.reward
    hazard = env.hazard_features
    critic_apply = nets.critic_apply
    clip = nets.clip_rows_to_median
    results = {}
    forwards = {}
    objectives = {}
    try:
        for condition in ('full_raw', 'reward', 'term_penalty', 'bootstrap', 'full_native_clip'):
            env.reward = reward
            env.hazard_features = hazard
            nets.critic_apply = critic_apply
            nets.clip_rows_to_median = clip
            if condition not in ('full_raw', 'full_native_clip', 'reward'):
                def detached_reward(*values, **keywords):
                    value, terms = reward(*values, **keywords)
                    return value.detach(), terms
                env.reward = detached_reward
            if condition not in ('full_raw', 'full_native_clip', 'term_penalty'):
                env.hazard_features = lambda *values: hazard(*values).detach()
            if condition not in ('full_raw', 'full_native_clip', 'bootstrap'):
                nets.critic_apply = lambda *values: critic_apply(*values).detach()
            if condition != 'full_native_clip':
                nets.clip_rows_to_median = lambda action, kappa, probe: IdentityCotangent.apply(action, probe)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            inputs = nets.tree_map(lambda value: value.detach().clone(), sampled)
            inputs[-2].requires_grad_()
            inputs[-1].requires_grad_()
            objective, gradient, statistics, records = trainer.window(*inputs)
            results[condition] = gradient
            forwards[condition] = records
            objectives[condition] = float(objective)
    finally:
        env.reward = reward
        env.hazard_features = hazard
        nets.critic_apply = critic_apply
        nets.clip_rows_to_median = clip
    full = results['full_raw']
    equality = {}
    for name, records in forwards.items():
        equality[name] = all(torch.equal(a, b) for a, b in zip(
            nets.tree_leaves(forwards['full_raw']), nets.tree_leaves(records)))
    residual = results['reward'] + results['term_penalty'] + results['bootstrap'] - full
    report = dict(checkpoint=str(args.checkpoint), weights=args.weights,
                  completed_updates=checkpoint['completed_updates'], seed=args.seed,
                  condition='New N128 H32 window sampled by the current trainer, not a historical training batch',
                  reset=bindings['reset'], forward_records_bitwise_equal=equality,
                  objectives=objectives,
                  gradients={name: gradient_reading(value, full, env.action_scale)
                             for name, value in results.items()},
                  component_sum_residual_rms=float(residual.square().mean().sqrt()),
                  component_sum_residual_maximum_absolute=float(residual.abs().max()),
                  component_sum_relative_l2=float(residual.norm()/full.norm().clamp_min(1e-30)))
    import numpy as np
    np.savez(args.output/'gradients.npz', **{name:value.cpu().numpy() for name,value in results.items()},
             action_scale=env.action_scale.cpu().numpy())
    (args.output/'result.json').write_text(json.dumps(report, indent=2))
    if not all(equality.values()):
        raise SystemExit('Gradient comparisons changed forward records')


if __name__ == '__main__':
    main()
