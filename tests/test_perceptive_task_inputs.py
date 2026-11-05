"""Generated tensor checks for the source perceptive actor interface."""
import unittest

import torch

from nablamimic.perceptive_task_inputs import (
    source_actor_apply, source_actor_init, source_depth_image,
    source_policy_observation, source_raw_camera_rays)
from nablamimic import torch_nets as nets


class PerceptiveTaskInputsTest(unittest.TestCase):
    def test_source_unlimited_position_targets_retain_action_effect_and_gradient(self):
        from nablamimic.source_perceptive_env import SourcePerceptiveEnv
        env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
        from types import SimpleNamespace
        env.cfg=SimpleNamespace(residual_action=False)
        env.q0=torch.tensor([.1,.2])
        env.action_scale=torch.tensor([.5,.25])
        env.ctrl_lo=torch.tensor([0.,-1.])
        env.ctrl_hi=torch.tensor([0.,1.])
        env.source_ctrl_limited=torch.tensor([False,True])
        action=torch.tensor([[2.,8.]],requires_grad=True)
        target=env.ctrl_of(action,torch.tensor([0]))
        torch.testing.assert_close(target,torch.tensor([[1.1,1.]]))
        gradient=torch.autograd.grad(target.sum(),action)[0]
        torch.testing.assert_close(gradient,torch.tensor([[.5,0.]]))

    def test_source_residual_zero_tracks_the_current_reference_without_changing_action_rate(self):
        from types import SimpleNamespace
        from nablamimic.source_perceptive_env import SourcePerceptiveEnv
        env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
        env.cfg=SimpleNamespace(residual_action=True)
        env.q0=torch.tensor([.1,.2])
        env.action_scale=torch.tensor([.5,.25])
        env.ctrl_lo=torch.zeros(2)
        env.ctrl_hi=torch.zeros(2)
        env.source_ctrl_limited=torch.zeros(2,dtype=torch.bool)
        env.ref_qpos=torch.zeros(2,9)
        env.ref_qpos[:,7:]=torch.tensor([[.7,-.3],[.9,.4]])
        frames=torch.tensor([0,1])
        zero=torch.zeros(2,2)
        torch.testing.assert_close(env.ctrl_of(zero,frames),env.ref_qpos[:,7:])
        torch.testing.assert_close(env.action_change(zero,zero),zero)
        self.assertEqual(env.action_contract['mode'],'current_reference_residual')
        env.cfg.residual_action=False
        torch.testing.assert_close(env.ctrl_of(zero,frames),env.q0.expand(2,-1))
        self.assertEqual(env.action_contract['mode'],'source_absolute')

    def test_source_actor_uses_history_before_nonnegative_depth_feature(self):
        from nablamimic.perceptive_task_inputs import SourceDepthEncoder
        encoder=SourceDepthEncoder()
        self.assertIsInstance(encoder.head[-1],torch.nn.ReLU)
        actor=source_actor_init('cpu')
        # Only the first history coordinate reaches output zero.
        for parameter in actor['enc'].values():
            parameter.zero_()
        actor['enc']['head.2.bias'].fill_(-1.)
        for weight,bias in actor['dec']:
            weight.zero_()
            bias.zero_()
        actor['dec'][0][0][670,0]=1.
        for weight,bias in actor['dec'][1:]:
            weight[0,0]=1.
        history=torch.zeros(1,1320)
        history[0,0]=2.02
        output=source_actor_apply(actor,history,torch.zeros(1,670))
        self.assertAlmostEqual(float(output[0,0]),2.,places=5)

    def test_camera_optical_distance_and_constant_image(self):
        rays = source_raw_camera_rays('cpu')
        torch.testing.assert_close(torch.linalg.vector_norm(rays, dim=-1), torch.ones(27,48))
        raw = 1./rays[..., 0]
        image = source_depth_image(raw[None], rays)
        torch.testing.assert_close(image, torch.full((1,1,18,32), .5))

    def test_source_empirical_normalizer_matches_population_moments(self):
        from nablamimic.perceptive_task_inputs import update_source_normalizer
        actor = source_actor_init('cpu')
        history = torch.arange(3*1320).reshape(3,1320).float()/1000
        goal = torch.arange(3*670).reshape(3,670).float()/1000
        update_source_normalizer(actor, history[:2], goal[:2])
        update_source_normalizer(actor, history[2:], goal[2:])
        packed = torch.cat((goal, history[:,744:], history[:,:744]), -1)
        torch.testing.assert_close(actor['source_norm']['mean'], packed.mean(0))
        torch.testing.assert_close(actor['source_norm']['variance'], packed.var(0,unbiased=False))
        self.assertEqual(int(actor['source_norm']['count']), 3)

    def test_source_depth_noise_is_finite_and_seed_repeatable(self):
        from nablamimic.source_depth_noise import SourceDepthNoise
        pipeline = SourceDepthNoise('cpu')
        meters = torch.ones(3,27,48,1)
        meters[0,:,:,0] = .05
        torch.manual_seed(3)
        first = pipeline.noised(meters)
        torch.manual_seed(3)
        second = pipeline.noised(meters)
        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (3,1,18,32))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(bool(((first>=0)&(first<=1)).all()))

    def test_source_image_input_stops_gradient_while_encoder_and_physical_inputs_train(self):
        torch.manual_seed(4)
        actor=source_actor_init('cpu')
        for parameter in nets.tree_leaves(nets.actor_trainable(actor)):
            parameter.requires_grad_()
        history=torch.rand(2,744,requires_grad=True)
        depth=torch.rand(2,576,requires_grad=True)
        goal=torch.rand(2,670,requires_grad=True)
        policy_history=torch.cat((history,depth),-1)
        output=source_actor_apply(actor,policy_history,goal)
        inputs=(depth,history,goal,actor['enc']['convolutions.0.weight'])
        image_gradient,history_gradient,goal_gradient,encoder_gradient=torch.autograd.grad(
            output.square().mean(),inputs,allow_unused=True)
        if image_gradient is not None:
            torch.testing.assert_close(image_gradient,torch.zeros_like(depth),rtol=0,atol=0)
        self.assertGreater(float(history_gradient.abs().sum()),0)
        self.assertGreater(float(goal_gradient.abs().sum()),0)
        self.assertGreater(float(encoder_gradient.abs().sum()),0)

    def test_term_major_history_and_actor_encoder_gradient(self):
        goal = torch.zeros(2,670)
        depth = torch.ones(2,1,18,32)*.5
        history = torch.arange(8*93).reshape(1,8,93).expand(2,-1,-1).float()
        packed = source_policy_observation(goal, depth, history)
        self.assertEqual(packed.shape, (2,1990))
        torch.testing.assert_close(packed[:,1246:1270], history[:,:,:3].flatten(1))
        actor = source_actor_init('cpu')
        policy_history = torch.cat((packed[:,1246:], depth.flatten(1)), -1)
        trainable = nets.actor_trainable(actor)
        def objective(parameters):
            return source_actor_apply(nets.with_actor_parameters(actor, parameters), policy_history, goal).square().mean()
        gradients = torch.func.grad(objective)(trainable)
        self.assertGreater(float(gradients['enc']['convolutions.0.weight'].abs().sum()), 0)
        self.assertEqual(nets.actor_apply(actor, policy_history, goal).shape, (2,29))
        average, count = nets.init_policy_ema(actor)
        _, next_count = nets.update_policy_ema(average, actor, count)
        self.assertEqual(int(next_count), 1)


if __name__ == '__main__':
    unittest.main()
