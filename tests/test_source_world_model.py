"""Generated differentiable state-integration checks without simulation."""
import unittest
import torch
from nablamimic.source_world_model import SourceWorldModel, so3_exp, integrate_increments


class SourceWorldModelTest(unittest.TestCase):
    def test_zero_rotation_exponential_has_finite_correct_derivative(self):
        vector = torch.zeros(2, 3, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(so3_exp, (vector,)))

    def test_yaw_local_translation_and_root_local_velocity(self):
        rotation = so3_exp(torch.tensor([[0., 0., torch.pi/2]]))
        state = (torch.zeros(1, 3), rotation, torch.zeros(1, 3),
                 torch.zeros(1, 3), torch.zeros(1, 29), torch.zeros(1, 29))
        increment = torch.zeros(1, 70)
        increment[0, 0] = 1.
        increment[0, 6] = 2.
        result = integrate_increments(state, increment)
        torch.testing.assert_close(result[0], torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(result[2], torch.tensor([[0., 2., 0.]]), atol=1e-6, rtol=0)

    def test_predictor_has_action_gradient_after_nonzero_weight_update(self):
        torch.manual_seed(0)
        model = SourceWorldModel(scan_dim=4, physics_dim=2, latent_dim=8,
                                 contact_dim=14, hidden=(32,))
        state = (torch.zeros(2, 3), torch.eye(3).expand(2, 3, 3),
                 torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 29), torch.zeros(2, 29))
        action = torch.zeros(2, 29, requires_grad=True)
        with torch.no_grad():
            model.net[-1].weight.normal_(std=.01)
        for _ in range(32):
            state, _, _ = model(state, action, torch.zeros(2, 4), torch.zeros(2, 2), torch.zeros(2, 8))
        gradient, = torch.autograd.grad(state[0].square().sum(), action)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0.)


class SourceWorldModelConditionTest(unittest.TestCase):
    def test_render_and_latent_have_distinct_model_inputs(self):
        models = [SourceWorldModel(scan_dim=187, physics_dim=0, latent_dim=dimension,
                                   contact_dim=30, hidden=(32,32)) for dimension in (0,32)]
        self.assertEqual(models[1].input_mean.numel()-models[0].input_mean.numel(),32)
        self.assertEqual(models[0].net[-1].out_features,100)
        self.assertEqual(models[1].net[-1].out_features,132)

    def test_render_fit_does_not_consume_images_or_predict_latents(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).parents[1]/'scripts/train_source_world_model.py'
        spec = importlib.util.spec_from_file_location('world_model_trainer', path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        class Vision:
            def encode(self, observation, goal):
                return observation[:, :32], goal, observation[:, :744]
            def reconstruction_loss(self, latent, target):
                return latent.square().mean()
        class Context:
            def __call__(self, state):
                return torch.zeros(2,187), torch.empty(2,0)
            def render(self, state):
                raise AssertionError('Render physics fitting must not render')
        batch = {name: torch.zeros(2,9,width) for name,width in
                 (('torso_state',13),('qpos',36),('qvel',35),('policy_observation',1320),('goal',670))}
        batch['torso_state'][:,:,3] = 1.
        batch.update(depth_clean=torch.zeros(2,9,1,18,32), pd_target=torch.zeros(2,8,29),
                     substep_all_force=torch.zeros(2,8,4,31,3))
        model = SourceWorldModel(scan_dim=187,physics_dim=0,latent_dim=0,contact_dim=30,hidden=(32,))
        loss, _ = trainer.sequence_loss(model, Vision(), batch, 8, Context(), condition='render', contact_force_threshold_n=1.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
