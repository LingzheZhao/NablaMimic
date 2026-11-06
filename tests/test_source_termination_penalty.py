"""Generated source termination errors and native smooth penalty checks."""
import unittest

import torch

from test_torch_env import fixture
from nablamimic.source_perceptive_env import SourcePerceptiveEnv
from nablamimic.torch_env import gravity_b


def source_fixture(batch=1):
    template,_=fixture()
    state=template.reset(torch.zeros(batch,dtype=torch.int32))
    env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
    env.__dict__.update(template.__dict__)
    env.torso_id=7
    env.anchor_k=7
    env.termination_end_indices=[3,6,10,13]
    return env,state


class SourceTerminationPenaltyTest(unittest.TestCase):
    def test_ankle_height_alone_terminates_and_receives_corrective_gradient(self):
        env,state=source_fixture()
        ankle=6
        height=torch.tensor(.3,requires_grad=True)
        displacement=torch.zeros_like(state['xpos'])
        displacement[:,ankle,2]=height
        state['xpos']=state['xpos']+displacement
        self.assertTrue(bool(env.terminated(state)[0]))
        penalty=env.termination_penalty(state,.08,.1)
        derivative=torch.autograd.grad(penalty.sum(),height)[0]
        self.assertGreater(float(derivative),0)
        self.assertEqual(float((state['qpos'][:,2]-env.ref_qpos[0,2]).abs()),0.)

    def test_full_gravity_detects_difference_with_identical_z_components(self):
        env,state=source_fixture()
        angle=torch.tensor(.5,requires_grad=True)
        actual=torch.stack((torch.cos(angle/2),torch.sin(angle/2),angle*0,angle*0))
        reference=torch.tensor([torch.cos(torch.tensor(-.25)),torch.sin(torch.tensor(-.25)),0.,0.])
        state['xquat']=actual[None,None].expand(1,14,4)
        env.ref_body_quat[:,env.anchor_k]=reference
        torch.testing.assert_close(gravity_b(actual)[2],gravity_b(reference)[2])
        self.assertTrue(bool(env.terminated(state)[0]))
        gradient=torch.autograd.grad(env.termination_penalty(state,.08,.1).sum(),angle)[0]
        self.assertGreater(float(gradient),0)

    def test_shared_hard_errors_match_previous_source_conditions(self):
        torch.manual_seed(12)
        env,state=source_fixture(32)
        state['xpos']=state['xpos']+torch.randn_like(state['xpos'])*.15
        q=torch.randn_like(state['xquat'])
        state['xquat']=q/q.norm(dim=-1,keepdim=True)
        indices=env.termination_end_indices
        frame=state['frame']
        old=((state['xpos'][:,env.torso_id,2]-env.ref_body_pos[frame,env.anchor_k,2]).abs()>.25)|(
            (gravity_b(state['xquat'][:,env.torso_id])-gravity_b(env.ref_body_quat[frame,env.anchor_k])).norm(dim=-1)>.8)|(
            (state['xpos'][:,env.body_ids[indices],2]-env.ref_body_pos[frame][:,indices,2]).abs().amax(-1)>.25)
        torch.testing.assert_close(env.terminated(state),old)
        state['qvel'][0,0]=float('nan')
        self.assertTrue(bool(env.terminated(state)[0]))

    def test_heightscan_penalty_preserves_original_formula(self):
        env,_=fixture()
        state=env.reset(torch.tensor([0]))
        state['qpos']=state['qpos'].clone()
        state['qpos'][:,2]+=.12
        hazard=env.hazard_features(state)
        zero=torch.zeros(())
        original=((torch.maximum(env.cfg.term_height-hazard[:,0]-.08,zero)/(env.cfg.term_height-.08))**2+
                  (torch.maximum(env.cfg.term_tilt-hazard[:,2]-.1,zero)/(env.cfg.term_tilt-.1))**2)
        torch.testing.assert_close(env.termination_penalty(state,.08,.1),original,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
