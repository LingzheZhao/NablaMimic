"""Generated CPU checks for source vision values, ownership and gradient paths."""

import unittest

import torch
from torch import nn

from nablamimic import torch_nets as nets
from nablamimic.perceptive_task_inputs import source_actor_apply, source_actor_init
from nablamimic.source_world_model_vision import SourceWorldModelVision


class SourceWorldModelVisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.actor = source_actor_init('cpu')
        self.actor['source_norm']['mean'].copy_(torch.linspace(-.3, .4, 1990))
        self.actor['source_norm']['variance'].copy_(torch.linspace(0., 2., 1990))
        self.actor['source_norm']['count'].fill_(123)
        self.actor['enc']['head.2.bias'].fill_(.3)
        self.vision = SourceWorldModelVision(self.actor)
        self.inputs = torch.rand(2, 1320)
        self.goal = torch.rand(2, 670)

    def test_normalization_split_and_source_actor_values_agree(self):
        feature, goal, history = self.vision.encode(self.inputs, self.goal)
        packed = torch.cat((self.goal, self.inputs[:, 744:], self.inputs[:, :744]), -1)
        norm = self.actor['source_norm']
        expected = (packed - norm['mean']) / (torch.sqrt(norm['variance']) + .01)
        torch.testing.assert_close(goal, expected[:, :670], rtol=0, atol=0)
        torch.testing.assert_close(history, expected[:, 1246:], rtol=0, atol=0)
        expected_depth = expected[:, 670:1246].reshape(2, 1, 18, 32)
        torch.testing.assert_close(feature, self.vision.encoder(expected_depth), rtol=0, atol=0)
        torch.testing.assert_close(feature, self.vision.encode_depth(
            self.inputs[:, 744:].reshape(2, 1, 18, 32)), rtol=0, atol=0)
        self.assertEqual(feature.shape, (2, 32))
        self.assertTrue(bool((feature >= 0).all()))
        expected_action = source_actor_apply(self.actor, self.inputs, self.goal)
        torch.testing.assert_close(self.vision.actor_from_features(feature, goal, history),
                                   expected_action, rtol=0, atol=0)
        torch.testing.assert_close(self.vision(self.inputs, self.goal), expected_action, rtol=0, atol=0)

    def test_depth_encoding_uses_only_the_depth_normalizer_slice(self):
        depth = self.inputs[:, 744:].reshape(2, 1, 18, 32)
        expected = self.vision.encode_depth(depth)
        with torch.no_grad():
            self.vision.normalizer_mean[:670].add_(9.)
            self.vision.normalizer_mean[1246:].sub_(7.)
        torch.testing.assert_close(self.vision.encode_depth(depth), expected, rtol=0, atol=0)

    def test_actor_updates_detach_feature_and_preserve_physical_input_gradients(self):
        inputs = self.inputs.clone().requires_grad_()
        goal = self.goal.clone().requires_grad_()
        feature, normalized_goal, history = self.vision.encode(inputs, goal)
        action = self.vision.actor_from_features(feature, normalized_goal, history)
        parameters = (feature, self.vision.encoder.convolutions[0].weight,
                      self.vision.actor_layers[0][0], inputs, goal)
        gradients = torch.autograd.grad(action.square().mean(), parameters, allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertGreater(float(gradients[2].abs().sum()), 0.)
        self.assertGreater(float(gradients[3][:, :744].abs().sum()), 0.)
        torch.testing.assert_close(gradients[3][:, 744:], torch.zeros(2, 576), rtol=0, atol=0)
        self.assertGreater(float(gradients[4].abs().sum()), 0.)

    def test_clean_reconstruction_trains_encoder_and_decoder(self):
        clean = torch.rand(2, 1, 18, 32, requires_grad=True)
        feature = self.vision.encode(self.inputs, self.goal)[0]
        projected = self.vision.depth_decoder[:3](feature)
        self.assertEqual(projected.shape, (2, 32, 5, 8))
        first_upsample = self.vision.depth_decoder[3:5](projected)
        self.assertEqual(first_upsample.shape, (2, 32, 9, 16))
        self.assertEqual(self.vision.depth_decoder[5:7](first_upsample).shape, (2, 16, 18, 32))
        self.assertEqual(self.vision.depth_decoder[3].kernel_size, (3, 3))
        self.assertEqual(self.vision.depth_decoder[3].output_padding, (0, 1))
        self.assertEqual(self.vision.depth_decoder[5].kernel_size, (3, 3))
        self.assertEqual(self.vision.depth_decoder[5].output_padding, (1, 1))
        reconstructed = self.vision.reconstruct(feature)
        self.assertEqual(reconstructed.shape, clean.shape)
        self.assertTrue(bool(((reconstructed >= 0) & (reconstructed <= 1)).all()))
        loss = self.vision.reconstruction_loss(feature, clean)
        torch.testing.assert_close(loss, (reconstructed - clean.detach()).abs().mean())
        gradients = torch.autograd.grad(loss, (self.vision.encoder.convolutions[0].weight,
            self.vision.depth_decoder[0].weight, self.vision.depth_decoder[3].weight,
            self.vision.depth_decoder[5].weight, self.vision.depth_decoder[7].weight,
            self.vision.actor_layers[0][0], clean), allow_unused=True)
        for gradient in gradients[:5]:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertIsNone(gradients[5])
        self.assertIsNone(gradients[6])

    def test_latent_prediction_target_does_not_train_encoder(self):
        predictor = nn.Linear(32, 32)
        target = self.vision.prediction_target(self.inputs[:, 744:].reshape(2, 1, 18, 32).requires_grad_())
        torch.testing.assert_close(target, self.vision.encode(self.inputs, self.goal)[0])
        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad_fn)
        loss = (predictor(torch.ones(2, 32)) - target).square().mean()
        predictor_gradient, encoder_gradient = torch.autograd.grad(loss,
            (predictor.weight, self.vision.encoder.convolutions[0].weight), allow_unused=True)
        self.assertGreater(float(predictor_gradient.abs().sum()), 0.)
        self.assertIsNone(encoder_gradient)

    def test_actor_roundtrip_copies_every_value_and_retains_functional_layout(self):
        exported = self.vision.export_actor()
        self.assertEqual(set(exported), set(self.actor))
        self.assertIsInstance(exported['dec'], list)
        self.assertTrue(all(isinstance(layer, tuple) for layer in exported['dec']))
        for expected, actual in zip(nets.tree_leaves(self.actor), nets.tree_leaves(exported)):
            self.assertEqual(expected.dtype, actual.dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if expected.numel():
                self.assertNotEqual(expected.data_ptr(), actual.data_ptr())
        with torch.no_grad():
            self.vision.actor_layers[-1][1].add_(.2)
            self.vision.encoder.head[2].bias.add_(.1)
        exported = self.vision.export_actor()
        torch.testing.assert_close(source_actor_apply(exported, self.inputs, self.goal),
                                   self.vision(self.inputs, self.goal), rtol=0, atol=0)
        expected_mean = self.vision.normalizer_mean.clone()
        exported['source_norm']['mean'].zero_()
        self.actor['source_norm']['mean'].fill_(8.)
        self.actor['enc']['convolutions.0.weight'].zero_()
        torch.testing.assert_close(self.vision.normalizer_mean, expected_mean, rtol=0, atol=0)
        self.assertGreater(float(self.vision.encoder.convolutions[0].weight.detach().abs().sum()), 0.)

    def test_inputs_use_current_source_shapes(self):
        with self.assertRaisesRegex(ValueError, 'Source inputs'):
            self.vision.encode(self.inputs[:, :-1], self.goal)
        with self.assertRaisesRegex(ValueError, 'Clean source depth'):
            feature = self.vision.encode(self.inputs, self.goal)[0]
            self.vision.reconstruction_loss(feature, torch.zeros(2, 18, 32))
        with self.assertRaisesRegex(ValueError, 'Source processed depth'):
            self.vision.encode_depth(torch.zeros(2, 18, 32))


if __name__ == '__main__':
    unittest.main()
