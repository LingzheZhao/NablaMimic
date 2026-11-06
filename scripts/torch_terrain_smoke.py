"""Finite CPU sensor/actor/critic/optimizer smoke, with analytic box geometry.

Native physics one-update checks use torch_train --iters 1 with a prepared scene.
"""
import argparse
import json
from pathlib import Path

import torch

from nablamimic import torch_nets as nets
from nablamimic.terrain_observation import ConvexTerrain, height_scan
from nablamimic.torch_train import build_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build_parser()
    normals = torch.cat((torch.eye(3), -torch.eye(3))).unsqueeze(0)
    terrain = ConvexTerrain(normals, -torch.ones(1, 6), torch.ones(1, 6, dtype=torch.bool))
    position = torch.tensor([[0., 0., 2.]], requires_grad=True)
    orientation = torch.tensor([[1., 0., 0., 0.]])
    offsets = torch.tensor([[0., 0.], [4., 0.]])
    scan = height_scan(terrain, position, orientation, offsets)
    torch.testing.assert_close(scan, torch.tensor([[2., 3.]]))
    derivative = torch.autograd.grad(scan[0, 0], position)[0]
    torch.testing.assert_close(derivative, torch.tensor([[0., 0., 1.]]))
    def layer(width, output):
        return (torch.randn(width, output)*0.01, torch.zeros(output))
    for displacement in (False, True):
        options = ['--out', str(args.output), '--future-stride', '5' if displacement else '1']
        if displacement:
            options.append('--reference-root-displacement')
        parsed = build_parser().parse_args(options)
        goal_width = 686 + 30*int(parsed.reference_root_displacement)
        critic_width = 902 + 30*int(parsed.reference_root_displacement)
        actor = dict(enc=[layer(goal_width-6, 32)], dec=[layer(32+1007+6, 29)], anchor_dec=torch.zeros(6))
        history = torch.zeros(2, 1007)
        goal = torch.zeros(2, goal_width)
        prediction = nets.actor_apply(actor, history, goal)
        assert prediction.shape == (2, 29)
        def loss(parameters):
            policy = nets.with_actor_parameters(actor, parameters)
            return nets.weighted_action_mse(nets.actor_apply(policy, history, goal), torch.ones(2, 29)*.01,
                                            torch.ones(2), torch.ones(29))
        parameters = nets.actor_trainable(actor)
        gradient, value = torch.func.grad_and_value(loss)(parameters)
        updated, optimizer, gradient_norm = nets.adam_step(parameters, gradient, nets.adam_init(parameters), 1.e-3)
        assert all(bool(torch.isfinite(item).all()) for item in nets.tree_leaves(updated))
        critic = [layer(critic_width, 1)]
        assert nets.critic_apply(critic, torch.zeros(2, critic_width)).shape == (2, 1)
    (args.output/'result.json').write_text(json.dumps(dict(sensor=True, pose_gradient=True,
        actor=True, critic=True, actor_regression_update=True, goal_widths=[686, 716], loss=float(value),
        scope='analytic sensor and numerical optimizer only'), indent=2))


if __name__ == '__main__':
    main()
