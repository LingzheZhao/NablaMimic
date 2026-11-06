"""Generated source state checks for evaluation-only world pose offsets."""
import importlib.util
from pathlib import Path
import unittest

import torch

from test_source_termination_penalty import source_fixture
from nablamimic.torch_env import quat_mul

path=Path(__file__).parents[1]/'scripts/evaluate_source_perceptive.py'
spec=importlib.util.spec_from_file_location('source_evaluator',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceInitialPoseOffsetTest(unittest.TestCase):
    def test_zero_offset_returns_identical_state_without_fk(self):
        env,state=source_fixture()
        def unexpected(*args):
            raise AssertionError('Zero offset must not rebuild FK')
        env.engine_kinematics=unexpected
        self.assertIs(module.offset_initial_state(env,state,[0.,0.],0.),state)

    def test_world_yaw_wxyz_translation_and_all_caches_are_consistent(self):
        env,state=source_fixture()
        env.m.nbody=14
        angle=torch.tensor(.4)
        state['qpos'][:,3:7]=torch.tensor([angle.cos(),angle.sin(),0.,0.])
        state=env.restore(env.compact(state))
        original={name:value.clone() for name,value in state.items()}
        reference=env.ref_qpos.clone()
        shifted=module.offset_initial_state(env,state,[.1,-.2],.6)
        yaw=torch.tensor([torch.cos(torch.tensor(.3)),0.,0.,torch.sin(torch.tensor(.3))])
        expected=quat_mul(yaw,state['qpos'][:,3:7])
        torch.testing.assert_close(shifted['qpos'][:,3:7],expected)
        self.assertGreater(float((expected-quat_mul(state['qpos'][:,3:7],yaw)).abs().max()),.01)
        torch.testing.assert_close(shifted['qpos'][:,:2],state['qpos'][:,:2]+torch.tensor([.1,-.2]))
        torch.testing.assert_close(shifted['qvel'],state['qvel'],rtol=0,atol=0)
        _,positions,rotations,velocities,centers=env.engine_kinematics(shifted['qpos'],shifted['qvel'],shifted['com'])
        for name,expected_value in (('xpos',positions),('xquat',rotations),('cvel',velocities),('subtree_com',centers)):
            torch.testing.assert_close(shifted[name],expected_value)
        torch.testing.assert_close(shifted['hist'],env.proprio(shifted)[:,None].expand_as(state['hist']))
        torch.testing.assert_close(env.ref_qpos,reference,rtol=0,atol=0)
        for name,value in original.items():
            torch.testing.assert_close(state[name],value,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
