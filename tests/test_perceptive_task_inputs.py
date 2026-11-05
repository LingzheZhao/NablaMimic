"""Generated tensor checks for the source perceptive actor interface."""
import unittest

import torch

from nablamimic.perceptive_task_inputs import (
    source_actor_apply, source_actor_init, source_depth_image,
    source_policy_observation, source_raw_camera_rays)
from nablamimic import torch_nets as nets


class PerceptiveTaskInputsTest(unittest.TestCase):
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
