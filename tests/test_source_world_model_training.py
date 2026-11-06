"""Generated checks for teacher-action adaptation and heldout gradient isolation."""
import importlib.util
from pathlib import Path
import unittest

import torch

from nablamimic.perceptive_task_inputs import source_actor_init
from nablamimic.source_world_model import SourceWorldModel
from nablamimic.source_world_model_vision import SourceWorldModelVision

path = Path(__file__).parents[1]/'scripts/train_source_world_model.py'
spec = importlib.util.spec_from_file_location('source_world_model_trainer', path)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class SourceWorldModelTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.vision = SourceWorldModelVision(source_actor_init('cpu'))
        self.batch = {name:torch.zeros(2,33,width) for name,width in
            (('torso_state',13),('qpos',36),('qvel',35),('policy_observation',1320),('goal',670))}
        self.batch['torso_state'][:,:,3] = 1.
        self.batch['policy_observation'].uniform_()
        self.batch.update(depth_clean=torch.zeros(2,33,1,18,32),
            depth_noisy=torch.rand(2,33,1,18,32), transition_valid=torch.ones(2,32,dtype=torch.bool),
            substep_all_force=torch.zeros(2,32,4,31,3),
            pd_target=torch.zeros(2,32,29), actor_action=torch.ones(2,32,29,requires_grad=True))

    def test_teacher_regression_updates_head_without_encoder_or_target_gradients(self):
        loss = trainer.actor_regression_loss(self.vision,self.batch,8)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.vision.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.vision.depth_decoder.parameters()))
        gradients = [p.grad for p in self.vision.actor_layers.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(sum(float(g.square().sum()) for g in gradients),0.)
        self.assertIsNone(self.batch['actor_action'].grad)

    def test_heldout_measurement_keeps_all_parameters_gradient_free(self):
        class Context:
            def __call__(self,state):
                return torch.zeros(2,187),torch.empty(2,0)
        model = SourceWorldModel(scan_dim=187,physics_dim=0,latent_dim=32,contact_dim=30,hidden=(32,))
        result = trainer.measure(model,self.vision,self.batch,Context(),'latent',
                                 dict(forecast_horizons=[1,8,32],fit_horizon=8,latent_variance_floor=1e-6))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.vision.parameters()))
        for horizon in ('1','8','32'):
            self.assertIn('current_encoder_latent_persistence_mse',result[horizon])

    def test_both_condition_fits_use_physical_statistics_and_backward(self):
        class Context:
            def __call__(self,state):
                return torch.zeros(2,187),torch.empty(2,0)
            def render(self,state):
                raise AssertionError('Model fitting must not render')
        study = dict(fit_horizon=8,latent_statistics_rate=.01)
        for condition, dimension in (('render',0),('latent',32)):
            model = SourceWorldModel(scan_dim=187,physics_dim=0,latent_dim=dimension,contact_dim=30,hidden=(32,))
            trainer.initialize_training_statistics(model,self.vision,self.batch,Context(),condition,study)
            self.assertEqual(model.input_mean.numel(),284)
            loss,_ = trainer.sequence_loss(model,self.vision,self.batch,8,Context(),condition=condition,
                contact_force_threshold_n=1.,latent_statistics_rate=.01,latent_variance_floor=1e-6)
            loss = loss + trainer.actor_regression_loss(self.vision,self.batch,8)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.vision.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
