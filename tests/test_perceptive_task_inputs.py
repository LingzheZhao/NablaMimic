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
