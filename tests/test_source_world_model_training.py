"""Generated checks for teacher-action adaptation and heldout gradient isolation."""
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nablamimic.perceptive_task_inputs import source_actor_init
from nablamimic.source_world_model import SourceWorldModel, so3_exp
from nablamimic.source_world_model_vision import SourceWorldModelVision
from nablamimic.source_world_model_data import (
    SCHEMA, STATE_SHAPES, TRANSITION_SHAPES, split_windows, validity_masks)

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
        self.batch['qpos'][:,:,3] = 1.
        self.batch['policy_observation'].uniform_()
        self.batch.update(depth_clean=torch.zeros(2,33,1,18,32),
            depth_noisy=self.batch['policy_observation'][:,:,744:].reshape(2,33,1,18,32),
            frame=torch.arange(33).expand(2,-1).clone(), transition_valid=torch.ones(2,32,dtype=torch.bool),
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
            dict(forecast_horizons=[1,8,32],fit_horizon=8,latent_variance_floor=1e-6,
                 reading_phase_boundaries=[0.,.25,.5,.75,1.]),
            control_dt_s=.02, window_metadata=self.reading_metadata())
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

    def reading_metadata(self):
        self.batch['frame'][1] = torch.arange(150,183)
        dataset = SimpleNamespace(window_index=np.asarray([[0,0,32],[1,50,82]]),
            episode_metadata=[dict(seed=2,clip_index=i,reference_clip=f'motion_{i:02d}') for i in range(2)])
        env = SimpleNamespace(starts=torch.tensor([0,100]),lengths=torch.tensor([101,101]))
        return trainer.window_reading_metadata(dataset,[0,1],self.batch,env,[0.,.25,.5,.75,1.])

    def test_constant_velocity_uses_world_translation_body_rotation_and_joint_velocity(self):
        rotation = so3_exp(torch.tensor([[torch.pi/2,0.,0.]],dtype=torch.double))
        state = (torch.tensor([[1.,2.,3.]],dtype=torch.double),rotation,
            torch.tensor([[.5,-1.,2.]],dtype=torch.double),torch.tensor([[0.,0.,1.]],dtype=torch.double),
            torch.full((1,29),.1,dtype=torch.double),torch.full((1,29),.4,dtype=torch.double))
        predicted = trainer.constant_velocity_prediction(state,.02,5)
        torch.testing.assert_close(predicted[0],torch.tensor([[1.05,1.9,3.2]],dtype=torch.double))
        torch.testing.assert_close(predicted[1],rotation@so3_exp(torch.tensor([[0.,.1,0.]],dtype=torch.double)))
        torch.testing.assert_close(predicted[4],torch.full((1,29),.14,dtype=torch.double))
        for index in (2,3,5):
            torch.testing.assert_close(predicted[index],state[index],rtol=0,atol=0)

    def test_forecasts_baselines_and_strata_retain_identical_window_supports(self):
        class Context:
            def __call__(self,state):
                return torch.zeros(2,187),torch.empty(2,0)
        class ConstantVelocityModel(torch.nn.Module):
            def forward(self,state,action,scan,physics,latent):
                return trainer.constant_velocity_prediction(state,.02,1),torch.zeros(2,30),latent
        times = torch.arange(33)*.02
        self.batch['torso_state'][:,:,:3] = times[None,:,None]*torch.tensor([1.,2.,0.])
        self.batch['torso_state'][:,:,7:10] = torch.tensor([1.,2.,0.])
        self.batch['qpos'][:,:,7:] = times[None,:,None]*.4
        self.batch['qvel'][:,:,6:] = .4
        metadata = self.reading_metadata()
        readings = trainer.measure(ConstantVelocityModel(),self.vision,self.batch,Context(),'latent',
            dict(forecast_horizons=[1,8,32],fit_horizon=8,latent_variance_floor=1e-6,
                 reading_phase_boundaries=[0.,.25,.5,.75,1.]),control_dt_s=.02,window_metadata=metadata)
        for horizon in (1,8,32):
            values = readings[str(horizon)]
            self.assertLess(values['baselines']['constant_velocity']['position_mean_m'],1e-6)
            self.assertAlmostEqual(values['baselines']['persistence']['position_mean_m'],
                                   5**.5*horizon*.02,places=6)
            self.assertAlmostEqual(values['target_motion']['root_path_speed_m_s'],5**.5,places=5)
            self.assertEqual(values['support']['states_per_window'],horizon+1)
            self.assertEqual(values['stratified']['motion']['0']['window_indices'],[0])
            self.assertEqual(values['stratified']['phase']['2']['window_indices'],[1])
            self.assertEqual(values['stratified']['phase']['1']['count'],0)
            self.assertIsNone(values['stratified']['phase']['1']['metrics'])
            self.assertEqual(values['window_readings']['metadata'],metadata)

    def test_resume_matches_uninterrupted_actual_optimizer_steps_and_all_rngs(self):
        class Context:
            def __call__(self,state):
                return torch.zeros(2,4),torch.empty(2,0)
        configuration = dict(vision='latent',seed=7,batch_size=2)
        binding = dict(dataset_artifact_id='generated-current-dataset',window_split=dict(split_type='episode',heldout_seeds=[2]))
        def components():
            torch.manual_seed(7)
            np.random.seed(8)
            random.seed(9)
            vision = SourceWorldModelVision(source_actor_init('cpu'))
            model = SourceWorldModel(scan_dim=4,physics_dim=0,latent_dim=32,contact_dim=30,hidden=(8,))
            optimizer = torch.optim.Adam(list(model.parameters())+list(vision.parameters()),lr=1e-4)
            sampler = np.random.default_rng(10)
            progress = dict(completed_updates=0,statistics_indices=[0],heldout_indices=[1],
                            update_indices=[],training_records=[])
            return model,vision,optimizer,sampler,progress
        def step(model,vision,optimizer,sampler,progress):
            indices = sampler.integers(2,size=2)
            batch = {name:value[indices].detach() for name,value in self.batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss,_ = trainer.sequence_loss(model,vision,batch,2,Context(),condition='latent',
                contact_force_threshold_n=1.,latent_statistics_rate=.01,latent_variance_floor=1e-6)
            loss = loss+trainer.actor_regression_loss(vision,batch,2)
            loss = loss*(torch.rand(())+np.random.random()+random.random())
            loss.backward()
            optimizer.step()
            progress['completed_updates'] += 1
            progress['update_indices'].append(indices.tolist())
            progress['training_records'].append(dict(loss=float(loss.detach())))
        full = components()
        for _ in range(4):
            step(*full)
        first = components()
        for _ in range(2):
            step(*first)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'fit.pt'
            trainer.save_fit_checkpoint(destination,*first[:4],configuration,binding,first[4])
            resumed = components()
            progress = trainer.restore_fit_checkpoint(destination,*resumed[:4],configuration,binding)
            resumed = (*resumed[:4],progress)
            for _ in range(2):
                step(*resumed)
            self.assertEqual(full[4],resumed[4])
            for original,continued in zip(full[:2],resumed[:2]):
                for name,value in original.state_dict().items():
                    torch.testing.assert_close(value,continued.state_dict()[name],rtol=0,atol=0)
            for original,continued in zip(full[2].state.values(),resumed[2].state.values()):
                for name,value in original.items():
                    torch.testing.assert_close(value,continued[name],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'data binding differs'):
                trainer.restore_fit_checkpoint(destination,*resumed[:4],configuration,dict(dataset_artifact_id='other'))
            with self.assertRaises(FileExistsError):
                trainer.save_fit_checkpoint(destination,*resumed[:4],configuration,binding,progress)

    def test_main_uses_manifest_episode_split_and_resumes_to_cumulative_target(self):
        """Exercise the fit CLI with real format/data and a tensor-only environment boundary."""
        from nablamimic import torch_nets
        class Context:
            scan_dim,physics_dim = 4,0
            def __init__(self,env,**options):
                self.metadata = dict(scan_dim=4,physics_dim=0,physics_steps=0)
            def __call__(self,state):
                return state[0].new_zeros((len(state[0]),4)),state[0].new_empty((len(state[0]),0))
        render_module = ModuleType('nablamimic.source_world_model_render')
        render_module.SourceWorldModelContext = Context
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'data'
            (dataset/'episodes').mkdir(parents=True)
            checkpoint = root/'actor.pkl'
            actor = source_actor_init('cpu')
            contract = dict(task='generated-source-task')
            checkpoint.write_bytes(pickle.dumps(dict(actor_ema=torch_nets.tree_map(lambda value:value.numpy(),actor),
                observation=contract,perception=contract,completed_updates=42)))
            episodes = []
            for seed in (0,1,2):
                for clip in range(2):
                    arrays = {name:np.zeros((41,*shape),dtype=np.float32) for name,shape in STATE_SHAPES.items()}
                    arrays.update({name:np.zeros((40,*shape),dtype=np.float32) for name,shape in TRANSITION_SHAPES.items()})
                    for name in ('tracking_terminated','clip_ended'):
                        arrays[name] = np.zeros(41,dtype=bool)
                    arrays['qpos'][:,3] = 1.
                    arrays['torso_state'][:,3] = 1.
                    arrays['frame'] = clip*1000+np.arange(41)
                    arrays['time_s'] = np.arange(41)*.02
                    arrays.update(validity_masks(arrays))
                    filename = f'episodes/episode_{len(episodes):02d}.npz'
                    np.savez(dataset/filename,**arrays)
                    episodes.append(dict(seed=seed,clip_index=clip,reference_clip=f'motion_{clip:02d}',
                                         file=filename,transitions=40))
            split = dict(split_type='episode',heldout_seeds=[2],window_controls=32)
            windows,bounds = split_windows(episodes,split)
            for episode,bound in zip(episodes,bounds):
                episode['split'] = bound
            manifest = dict(schema=SCHEMA,status='complete',artifact_id='generated-current-data',
                episode_seeds=[0,1,2],motion_count=2,episodes=episodes,window_split=split,
                clock=dict(control_dt_s=.02),teacher=dict(weights='ema',path=str(checkpoint),
                    completed_updates=42,observation=contract,perception=contract))
            (dataset/'manifest.json').write_text(json.dumps(manifest))
            np.savez(dataset/'windows.npz',**windows)
            config = root/'config.json'
            config.write_text(json.dumps(dict(terrain_model='generated.mjb',terrain_model_contract='generated.json',
                source_motion_start_height_offset=.1,source_ensure_link_below_zero_ground=True)))
            study = dict(model_hidden=[8],fit_horizon=2,forecast_horizons=[1,8,32],statistics_windows=2,
                contact_force_threshold_n=1.,heightmap_resolution_m=.1,scan_max_distance_m=3.,margin_m=2.,
                z_top_m=10.,actor_regression_weight=1.,latent_statistics_rate=.01,latent_variance_floor=1e-6,
                reading_phase_boundaries=[0.,.25,.5,.75,1.],checkpoint_every_updates=1)
            study_path = root/'study.yaml'
            study_path.write_text(trainer.yaml.safe_dump(study))
            env = SimpleNamespace(device=torch.device('cpu'),starts=torch.tensor([0,1000]),lengths=torch.tensor([41,41]),
                m=SimpleNamespace(nbody=31),observation_contract=contract,perception_contract=contract,camera_noise=None)
            bindings = dict(physical_model=dict(artifact_id='generated-physical-model'),engine=dict(artifact_version=5))
            def run(output,updates,resume=None):
                arguments = ['train_source_world_model.py','--study-config',str(study_path),'--vision','latent',
                    '--dataset',str(dataset),'--config',str(config),'--checkpoint',str(checkpoint),
                    '--output',str(output),'--device','cpu','--batch-size','2','--updates',str(updates)]
                if resume is not None:
                    arguments.extend(['--resume',str(resume)])
                with patch.object(sys,'argv',arguments),patch.object(sys,'path',[str(path.parent),*sys.path]),\
                     patch.dict(sys.modules,{'nablamimic.source_world_model_render':render_module}),\
                     patch('nablamimic.torch_train.build_environment',return_value=(env,bindings)):
                    trainer.main()
            run(root/'full',2)
            run(root/'first',1)
            run(root/'resumed',2,root/'first/latent.pt')
            full = torch.load(root/'full/latent.pt',weights_only=False)
            resumed = torch.load(root/'resumed/latent.pt',weights_only=False)
            self.assertEqual(full['completed_updates'],2)
            self.assertEqual(full['update_indices'],resumed['update_indices'])
            self.assertEqual(full['heldout_indices'],resumed['heldout_indices'])
            self.assertEqual(full['training_records'],resumed['training_records'])
            for name in ('model','vision'):
                for key,value in full[name].items():
                    torch.testing.assert_close(value,resumed[name][key],rtol=0,atol=0)
            protocol = json.loads((root/'resumed/resolved_protocol.json').read_text())
            self.assertEqual(protocol['dataset_split'],manifest['window_split'])
            result = json.loads((root/'resumed/result.json').read_text())
            self.assertEqual(result['starting_completed_updates'],1)
            self.assertEqual(result['completed_updates'],2)
            self.assertEqual(result['status'],'complete')


if __name__ == '__main__':
    unittest.main()
