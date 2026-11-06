"""Generated differentiable closed-loop feedback, without physics execution."""
from types import SimpleNamespace
import unittest
import torch
from torch import nn

from nablamimic.source_world_model import SourceWorldModel
from nablamimic.source_world_model_imagination import SourceWorldModelImagination


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(576,32)
        self.depth_decoder=nn.Linear(32,576)
        self.actor_layers=nn.Linear(744+3+32,29)
    def encode(self,observation,goal):
        return self.encoder(observation[:,744:]),goal,observation[:,:744]
    def actor_from_features(self,feature,goal,history):
        return self.actor_layers(torch.cat((history,goal,feature.detach()),-1))


class Env:
    body_ids=[0]
    torso_id=0
    anchor_k=0
    root_com_id=0
    action_scale=.25
    def __init__(self):
        self.frames=[]
        self.ref_body_pos=torch.zeros(64,1,3)
        self.ref_body_lin=torch.zeros(64,1,3)
        self.ref_qpos=torch.zeros(64,36)
        self.ref_qvel=torch.zeros(64,35)
        self.ref_body_quat=torch.zeros(64,1,4);self.ref_body_quat[:,:,0]=1.
    def engine_kinematics(self,q,v,com):
        position=q[:,:3,None].transpose(1,2)+q[:,7:10,None].transpose(1,2)
        velocity=torch.cat((v[:,3:6],v[:,:3]),-1)[:,None]
        return q,position,q[:,3:7,None].transpose(1,2),velocity,position
    def body_velocity(self,state):
        from nablamimic.torch_env import TorchTrackingEnv
        return TorchTrackingEnv.body_velocity(self,state)
    def source_goal(self,state):
        self.frames.append(state['frame'].detach().clone())
        return state['qpos'][:,:3]+state['xquat'][:,0,1:]
    def bound_action(self,action):return action.tanh()
    def ctrl_of(self,action,frame):return action*.25
    def termination_penalty(self,state,height,gravity):return state['qpos'][:,:3].square().sum(-1)
    def proprio(self,state):
        from nablamimic.torch_env import gravity_b
        return torch.cat((gravity_b(state['xquat'][:,0]),state['qvel'][:,3:6],state['qpos'][:,7:],
                          state['qvel'][:,6:],state['prev_a']),-1)


class Context:
    def __call__(self,state):return state[0].new_empty(2,0),state[0].new_empty(2,0)
    def render(self,state):return state[0].new_zeros(2,1,18,32)


class SourceWorldModelImaginationTest(unittest.TestCase):
    def test_feedback_changes_actions_and_keeps_frozen_parameter_gradients_absent(self):
        torch.manual_seed(5)
        for condition,dimension in (('render',0),('latent',32)):
            env=Env();vision=Vision()
            model=SourceWorldModel(scan_dim=0,physics_dim=0,latent_dim=dimension,contact_dim=1,hidden=(32,))
            with torch.no_grad():model.net[-1].weight.normal_(std=.01)
            objective={name+'_weight':1. for name in ('body_position','body_velocity','torso_orientation','joint_position','joint_velocity','termination','action_smoothness')}
            objective.update(teacher_action_weight=0.,height_margin_start_m=.15,gravity_margin_start=.5)
            imagined=SourceWorldModelImagination(env,Context(),model,vision,condition,objective)
            state=(torch.zeros(2,3),torch.eye(3).expand(2,3,3),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
            history=torch.zeros(2,8,93,requires_grad=True)
            args=(state,history,torch.zeros(2,29),torch.zeros(2,dtype=torch.int32),torch.zeros(2,3),4)
            loss,_,result=imagined.rollout(*args)
            loss.backward()
            self.assertTrue(torch.isfinite(history.grad).all())
            self.assertGreater(float(history.grad.abs().max()),0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in vision.encoder.parameters()))
            self.assertGreater(sum(float(p.grad.square().sum()) for p in vision.actor_layers.parameters()),0.)
            self.assertTrue(torch.equal(result['history'][:,-1,64:93],result['actions'][:,-1]))
            changed=list(args);changed[1]=history.detach()+.1
            _,_,perturbed=imagined.rollout(*changed)
            self.assertGreater(float((perturbed['actions']-result['actions']).detach().abs().max()),0.)
            self.assertEqual(int(result['frame'][0]),4)
            joint_state=list(state);joint_state[4]=state[4]+.1
            changed=list(args);changed[0]=tuple(joint_state)
            _,_,joint_response=imagined.rollout(*changed)
            self.assertGreater(float((joint_response['actions'][:,1:]-result['actions'][:,1:]).detach().abs().max()),0.)
            from nablamimic.source_world_model import so3_exp
            rotated=list(state);rotated[1]=so3_exp(torch.tensor([[.2,0.,0.],[.2,0.,0.]]))
            changed=list(args);changed[0]=tuple(rotated)
            _,_,rotation_response=imagined.rollout(*changed)
            self.assertGreater(float((rotation_response['actions']-result['actions']).detach().abs().max()),0.)


    def test_initial_observation_matches_source_history_packing_and_normalization(self):
        from nablamimic.source_world_model_imagination import policy_observation
        from nablamimic.source_world_model_vision import SourceWorldModelVision
        from nablamimic.perceptive_task_inputs import source_actor_init,source_actor_apply
        history=torch.randn(2,8,93);depth=torch.rand(2,1,18,32);goal=torch.randn(2,670)
        observation=policy_observation(history,depth)
        expected=torch.cat(tuple(history[:,:,start:end].flatten(1) for start,end in
            ((0,3),(3,6),(6,35),(35,64),(64,93)))+(depth.flatten(1),),-1)
        torch.testing.assert_close(observation,expected,rtol=0,atol=0)
        actor=source_actor_init('cpu');vision=SourceWorldModelVision(actor)
        torch.testing.assert_close(vision(observation,goal),source_actor_apply(actor,expected,goal),rtol=0,atol=0)

    def test_future_recorded_history_is_not_an_input(self):
        import importlib.util
        from pathlib import Path
        import sys
        import numpy as np
        path=Path(__file__).parents[1]/'scripts/refine_source_world_model_actor.py'
        sys.path.insert(0,str(path.parent))
        try:
            spec=importlib.util.spec_from_file_location('refiner',path)
            script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
        finally:
            sys.path.pop(0)
        row=dict(pd_target=np.zeros((32,29)),state_valid=np.ones(33,dtype=bool),
            torso_state=np.zeros((33,13)),qpos=np.zeros((33,36)),qvel=np.zeros((33,35)),
            history=np.zeros((33,8,93)),frame=np.arange(33))
        row['torso_state'][:,3]=1.
        initial=script.initial_batch([row],[0],'cpu',32)
        row['history'][1:]=1000.;row['torso_state'][1:]=1000.
        changed=script.initial_batch([row],[0],'cpu',32)
        for a,b in zip(initial[0],changed[0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(initial[1:],changed[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_body_origin_velocity_includes_nonzero_com_offset(self):
        class OffsetEnv(Env):
            def engine_kinematics(self,q,v,com):
                position=q.new_tensor([[[1.,0.,0.]]]).expand(2,1,3)
                quaternion=q.new_tensor([[[1.,0.,0.,0.]]]).expand(2,1,4)
                cvel=q.new_tensor([[[0.,0.,2.,1.,2.,3.]]]).expand(2,1,6)
                return q,position,quaternion,cvel,torch.zeros_like(position)
        env=OffsetEnv();env.ref_body_lin[:]=torch.tensor([1.,4.,3.])
        vision=Vision();model=SourceWorldModel(scan_dim=0,physics_dim=0,latent_dim=0,contact_dim=1,hidden=(32,))
        objective={name+'_weight':0. for name in ('body_position','body_velocity','torso_orientation','joint_position','joint_velocity','termination','action_smoothness')}
        objective.update(body_velocity_weight=1.,teacher_action_weight=0.,height_margin_start_m=.15,gravity_margin_start=.5)
        imagined=SourceWorldModelImagination(env,Context(),model,vision,'render',objective)
        state=(torch.zeros(2,3),torch.eye(3).expand(2,3,3),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
        loss,metrics,_=imagined.rollout(state,torch.zeros(2,8,93),torch.zeros(2,29),torch.zeros(2,dtype=torch.int32),torch.zeros(2,3),1)
        self.assertEqual(float(metrics['body_velocity']),0.)
        self.assertEqual(float(loss.detach()),0.)
        # Directly consuming cvel linear would instead give mean-square error 4/3.
        self.assertAlmostEqual(float((torch.tensor([1.,2.,3.])-env.ref_body_lin[0,0]).square().mean()),4/3,places=6)

    def test_teacher_regularization_is_in_pd_radians_and_teacher_stays_frozen(self):
        class Teacher(Vision):
            def forward(self,observation,goal):return observation.new_zeros((2,29))
        env=Env();vision=Vision();teacher=Teacher()
        saved={name:value.detach().clone() for name,value in teacher.state_dict().items()}
        model=SourceWorldModel(scan_dim=0,physics_dim=0,latent_dim=0,contact_dim=1,hidden=(32,))
        objective={name+'_weight':0. for name in ('body_position','body_velocity','torso_orientation','joint_position','joint_velocity','termination','action_smoothness')}
        objective.update(teacher_action_weight=1.,height_margin_start_m=.15,gravity_margin_start=.5)
        imagined=SourceWorldModelImagination(env,Context(),model,vision,'render',objective,teacher=teacher)
        state=(torch.zeros(2,3),torch.eye(3).expand(2,3,3),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,29),torch.zeros(2,29))
        loss,metrics,_=imagined.rollout(state,torch.zeros(2,8,93),torch.zeros(2,29),torch.zeros(2,dtype=torch.int32),torch.zeros(2,3),1)
        torch.testing.assert_close(metrics['teacher_action'],metrics['teacher_action_normalized']*.25**2)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        for name,value in teacher.state_dict().items():torch.testing.assert_close(value,saved[name],rtol=0,atol=0)

    def test_fit_binding_rejects_old_format_and_changed_scientific_inputs(self):
        import copy
        import importlib.util
        from pathlib import Path
        import sys
        path=Path(__file__).parents[1]/'scripts/refine_source_world_model_actor.py'
        sys.path.insert(0,str(path.parent))
        try:
            spec=importlib.util.spec_from_file_location('refine_fit_binding',path)
            script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
        finally:
            sys.path.pop(0)
        env=SimpleNamespace(observation_contract={'layout':'source'},perception_contract={'scene':'six'})
        context=SimpleNamespace(metadata={'scan_dim':187})
        study={'model_hidden':[256,256],'checkpoint_every_updates':100}
        teacher_checkpoint={'completed_updates':2000}
        manifest={'clock':{'control_dt_s':.02},'artifact_id':'generated-data-v1',
            'teacher':dict(weights='ema',completed_updates=2000,path='/tmp/generated-teacher.pkl',
                observation=env.observation_contract,perception=env.perception_contract)}
        dataset=SimpleNamespace(manifest=manifest)
        bindings={'physical_model':{'artifact_id':'generated-source-model'}}
        binding=dict(dataset='/tmp/generated-dataset',dataset_manifest=manifest,
            checkpoint='/tmp/generated-teacher.pkl',checkpoint_updates=2000,
            observation=env.observation_contract,perception=env.perception_contract,
            physical_model=bindings['physical_model'],context=context.metadata,
            torch_version=str(torch.__version__),control_dt_s=.02)
        checkpoint=dict(format='nablamimic_source_world_model_fit_v1',condition='latent',model={},vision={},optimizer={},
            resolved_fit_configuration={'vision':'latent','study':{'model_hidden':[256,256]}},data_binding=binding)
        def check(value):return script.validate_fit_inputs(value,dataset,env,bindings,context,study,
            Path('/tmp/generated-dataset'),Path('/tmp/generated-teacher.pkl'),teacher_checkpoint)
        self.assertEqual(check(checkpoint),'latent')
        for field in ('format','study','scene','data','condition'):
            changed=copy.deepcopy(checkpoint)
            if field=='format':changed['format']='old'
            if field=='study':changed['resolved_fit_configuration']['study']['model_hidden']=[32]
            if field=='scene':changed['data_binding']['physical_model']['artifact_id']='different'
            if field=='data':changed['data_binding']['dataset_manifest']['artifact_id']='different'
            if field=='condition':changed['condition']='render'
            with self.assertRaises(ValueError):check(changed)


if __name__=='__main__':unittest.main()
