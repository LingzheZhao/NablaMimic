"""Generated training-provenance and explicit inference-camera boundary checks."""
import importlib.util
from pathlib import Path
import unittest


path=Path(__file__).parents[1]/'scripts/evaluate_source_perceptive.py'
spec=importlib.util.spec_from_file_location('source_camera_evaluator',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceEvaluationCameraInputTest(unittest.TestCase):
    def test_default_preserves_both_training_input_conditions(self):
        for mode in ('measured','constant-far'):
            config={'source_camera_input':mode}
            observation={'camera_input':mode,'physical_binding':'same'}
            result=module.resolve_camera_input(config,observation,None)
            self.assertEqual(result['camera_input'],mode)
            self.assertEqual(result['source_camera_input'],mode)
            self.assertEqual(result['training_camera_input'],mode)
            self.assertIsNone(result['inference_camera_input_override'])
            self.assertEqual(config,{'source_camera_input':mode})
            self.assertEqual(observation,{'camera_input':mode,'physical_binding':'same'})

    def test_only_explicit_override_restores_measured_depth_after_no_depth_training(self):
        config={'source_camera_input':'constant-far'}
        observation={'camera_input':'constant-far'}
        for mode in ('measured','constant-far','self-only'):
            result=module.resolve_camera_input(config,observation,mode)
            self.assertEqual(result['training_camera_input'],'constant-far')
            self.assertEqual(result['inference_camera_input_override'],mode)
            self.assertEqual(result['camera_input'],mode)
            self.assertEqual(result['source_camera_input'],'measured' if mode=='self-only' else mode)

    def test_override_does_not_allow_wrong_checkpoint_training_mode(self):
        for override in (None,'measured','constant-far'):
            with self.assertRaisesRegex(ValueError,'training camera input differs'):
                module.resolve_camera_input({'source_camera_input':'constant-far'},
                                            {'camera_input':'measured'},override)
        with self.assertRaises(KeyError):
            module.resolve_camera_input({}, {'camera_input':'measured'},None)
        with self.assertRaises(KeyError):
            module.resolve_camera_input({'source_camera_input':'measured'}, {},None)

    def test_constant_training_default_and_explicit_measured_override_use_actual_image(self):
        from dataclasses import dataclass,replace
        import torch
        from nablamimic.source_perceptive_env import SourcePerceptiveEnv
        @dataclass
        class Settings:
            source_camera_input: str='constant-far'
            source_depth_gradient: bool=False
        env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
        env.cfg=Settings()
        generated=[]
        measured=torch.arange(576).reshape(1,1,18,32).float()/576
        def pipeline(state):
            generated.append(True)
            return measured.clone()
        env._depth_image=pipeline
        condition=module.resolve_camera_input({'source_camera_input':'constant-far'},
                                             {'camera_input':'constant-far'},None)
        torch.testing.assert_close(env.depth_image({}),torch.ones_like(measured))
        self.assertEqual(len(generated),1)
        condition=module.resolve_camera_input({'source_camera_input':'constant-far'},
                                             {'camera_input':'constant-far'},'measured')
        env.cfg=replace(env.cfg,source_camera_input=condition['source_camera_input'])
        torch.testing.assert_close(env.depth_image({}),measured)
        self.assertEqual(len(generated),2)

    def test_environment_policy_observation_owns_actor_and_recorded_depth(self):
        from types import SimpleNamespace
        import torch
        from nablamimic.source_perceptive_env import SourcePerceptiveEnv
        env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
        env.cfg=SimpleNamespace(source_camera_input='constant-far',source_depth_gradient=False,obs_noise=False)
        measured=torch.linspace(0.,1.,576).reshape(1,1,18,32)
        env._depth_image=lambda state:measured.clone()
        goal=torch.arange(670).float()[None]
        env.source_goal=lambda state:goal
        history=torch.arange(8*93).reshape(1,8,93).float()
        original=history.clone()
        state={'hist':history}
        observation,actual_goal=env.policy_obs(state)
        expected_history=torch.cat([history[...,start:stop].flatten(1) for start,stop in
            ((0,3),(3,6),(6,35),(35,64),(64,93))],-1)
        torch.testing.assert_close(observation[:,:744],expected_history)
        torch.testing.assert_close(actual_goal,goal)
        torch.testing.assert_close(state['hist'],original,rtol=0,atol=0)
        actor_depth=observation[:,744:].reshape(1,1,18,32)
        recorded_depth=observation[0,744:]
        torch.testing.assert_close(actor_depth,torch.ones_like(measured))
        torch.testing.assert_close(actor_depth[0].flatten(),recorded_depth,rtol=0,atol=0)
        env.cfg.source_camera_input='measured'
        observation,_=env.policy_obs(state)
        torch.testing.assert_close(observation[:,744:],measured.flatten(1))


if __name__=='__main__':
    unittest.main()
