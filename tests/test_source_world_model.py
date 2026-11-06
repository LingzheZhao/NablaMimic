"""Generated differentiable state-integration checks without simulation."""
import unittest
import torch
from nablamimic.source_world_model import (
    SourceWorldModel, so3_exp, integrate_increments, invariant_features)


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
        self.assertEqual(models[1].input_mean.shape, models[0].input_mean.shape)
        self.assertEqual(models[1].net[0].in_features-models[0].net[0].in_features,32)
        self.assertEqual(models[1].z_mean.shape, (32,))
        self.assertEqual(models[0].z_mean.shape, (0,))
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


class SourceWorldModelLatentStatisticsTest(unittest.TestCase):
    def test_statistics_follow_moving_encoder_without_encoder_gradients(self):
        model = SourceWorldModel(scan_dim=0, physics_dim=0, latent_dim=2,
                                 contact_dim=0, hidden=())
        encoder = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            encoder.weight.copy_(torch.eye(2))
        observations = torch.tensor([[[0., 0.], [1., 2.], [4., 6.]],
                                     [[2., 4.], [4., 8.], [8., 16.]]], requires_grad=True)
        valid = torch.tensor([[True, False], [True, True]])
        target_mean = torch.tensor([4.25, 8.])
        target_std = torch.tensor([8.25, 104./3]).sqrt()
        difference_rms = torch.tensor([7., 28.]).sqrt()
        model.update_latent_statistics(encoder(observations), valid, rate=.25)
        torch.testing.assert_close(model.z_mean, .25*target_mean)
        torch.testing.assert_close(model.z_std, .75+.25*target_std)
        torch.testing.assert_close(model.dz_std, .75+.25*difference_rms)
        previous = [value.clone() for value in (model.z_mean, model.z_std, model.dz_std)]
        with torch.no_grad():
            encoder.weight.mul_(3.)
        model.update_latent_statistics(encoder(observations), valid, rate=.25)
        for value, old, target in zip((model.z_mean, model.z_std, model.dz_std), previous,
                                     (target_mean, target_std, difference_rms)):
            torch.testing.assert_close(value, .75*old+.25*3*target)
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        self.assertIsNone(encoder.weight.grad)
        self.assertIsNone(observations.grad)
        torch.testing.assert_close(model.input_mean, torch.zeros_like(model.input_mean))
        torch.testing.assert_close(model.input_std, torch.ones_like(model.input_std))
        self.assertTrue({'z_mean', 'z_std', 'dz_std'} <= model.state_dict().keys())

    def test_singleton_and_zero_validity_keep_finite_scale_floors(self):
        model = SourceWorldModel(scan_dim=0, physics_dim=0, latent_dim=2,
                                 contact_dim=0, hidden=())
        model.update_latent_statistics(torch.ones(1, 2, 2), torch.zeros(1, 1, dtype=torch.bool), rate=1.)
        torch.testing.assert_close(model.z_mean, torch.ones(2))
        torch.testing.assert_close(model.z_std, torch.full((2,), 1e-3))
        torch.testing.assert_close(model.dz_std, torch.full((2,), 1e-3))

    def test_latent_uses_separate_input_scale_and_recursive_increment_scale(self):
        model = SourceWorldModel(scan_dim=0, physics_dim=0, latent_dim=2,
                                 contact_dim=0, hidden=())
        with torch.no_grad():
            model.input_mean.fill_(10.)
            model.input_std.fill_(2.)
            model.z_mean.copy_(torch.tensor([2., -1.]))
            model.z_std.copy_(torch.tensor([4., 2.]))
            model.dz_std.copy_(torch.tensor([3., 4.]))
            model.net[-1].weight[-2:, -2:].copy_(torch.eye(2))
        state = (torch.zeros(1, 3), torch.eye(3)[None], torch.zeros(1, 3),
                 torch.zeros(1, 3), torch.zeros(1, 29), torch.zeros(1, 29))
        action = torch.zeros(1, 29)
        empty = torch.empty(1, 0)
        latent = torch.tensor([[6., 3.]], requires_grad=True)
        captured = []
        handle = model.net.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        following_state, _, first = model(state, action, empty, empty, latent)
        _, _, second = model(following_state, action, empty, empty, first)
        handle.remove()
        physical = torch.cat((invariant_features(state), action), -1)
        torch.testing.assert_close(captured[0][:, :-2], (physical-10.)/2., rtol=0, atol=0)
        torch.testing.assert_close(captured[0][:, -2:], torch.tensor([[1., 2.]]), rtol=0, atol=0)
        torch.testing.assert_close(first, torch.tensor([[9., 11.]]), rtol=0, atol=0)
        torch.testing.assert_close(second, torch.tensor([[14.25, 35.]]), rtol=0, atol=0)
        gradient, = torch.autograd.grad(second.sum(), latent)
        torch.testing.assert_close(gradient, torch.tensor([[3.0625, 9.]]), rtol=0, atol=0)

    def test_render_matches_original_physical_formula_with_nonzero_predictions(self):
        torch.manual_seed(13)
        model = SourceWorldModel(scan_dim=3, physics_dim=1, latent_dim=0,
                                 contact_dim=4, hidden=(16,))
        with torch.no_grad():
            model.net[-1].weight.normal_(std=.02)
            model.net[-1].bias.normal_(std=.01)
            model.input_mean.normal_(std=.2)
            model.input_std.uniform_(.5, 2.)
            model.increment_scale.uniform_(.05, .2)
        state = (torch.randn(2, 3), so3_exp(torch.randn(2, 3)*.2), torch.randn(2, 3),
                 torch.randn(2, 3), torch.randn(2, 29), torch.randn(2, 29))
        action, scan, physics = torch.randn(2, 29), torch.randn(2, 3), torch.randn(2, 1)
        latent = torch.empty(2, 0)
        original_features = torch.cat((invariant_features(state), action, scan, physics, latent), -1)
        output = model.net((original_features-model.input_mean)/model.input_std)
        increment, contacts, expected_latent = output.split((70, 4, 0), -1)
        expected_state = integrate_increments(state, increment*model.increment_scale)
        model.update_latent_statistics(torch.empty(2, 3, 0), torch.ones(2, 2, dtype=torch.bool))
        actual_state, actual_contacts, actual_latent = model(state, action, scan, physics, latent)
        for actual, expected in zip(actual_state, expected_state):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_contacts, contacts, rtol=0, atol=0)
        torch.testing.assert_close(actual_latent, expected_latent, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
