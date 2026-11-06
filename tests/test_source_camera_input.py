"""Generated sensor inputs check the training camera condition without physics."""
from types import SimpleNamespace
import unittest
import torch

from nablamimic.source_perceptive_env import SourcePerceptiveEnv
from nablamimic.perceptive_task_inputs import source_actor_init,source_actor_apply


def environment(condition):
    env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
    env.cfg=SimpleNamespace(source_camera_input=condition,source_depth_gradient=False,obs_noise=False)
    env.source_goal=lambda state:torch.zeros(2,670)
    # Stand-in for complete camera generation, including its random draws.
    env._depth_image=lambda state:torch.rand(2,1,18,32)
    return env


class SourceCameraInputTest(unittest.TestCase):
    def test_far_input_consumes_same_rng_and_hides_all_generated_depth(self):
        state={'hist':torch.randn(2,8,93)}
        torch.manual_seed(19)
        measured,_=environment('measured').policy_obs(state)
        measured_rng=torch.get_rng_state()
        torch.manual_seed(19)
        far,_=environment('constant-far').policy_obs(state)
        torch.testing.assert_close(torch.get_rng_state(),measured_rng,rtol=0,atol=0)
        torch.testing.assert_close(measured[:,:744],far[:,:744],rtol=0,atol=0)
        torch.testing.assert_close(far[:,744:],torch.ones(2,576),rtol=0,atol=0)
        self.assertFalse(torch.equal(measured[:,744:],far[:,744:]))
        env=environment('constant-far')
        env._depth_image=lambda state:torch.zeros(2,1,18,32)
        changed,_=env.policy_obs(state)
        torch.testing.assert_close(changed,far,rtol=0,atol=0)

    def test_far_input_preserves_network_and_encoder_parameter_learning(self):
        torch.manual_seed(7)
        actor=source_actor_init('cpu')
        for parameter in actor['enc'].values():parameter.requires_grad_(True)
        observation,goal=environment('constant-far').policy_obs({'hist':torch.randn(2,8,93)})
        source_actor_apply(actor,observation,goal).square().sum().backward()
        gradients=[p.grad for p in actor['enc'].values() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(sum(float(g.square().sum()) for g in gradients),0.)

    def test_critic_does_not_call_camera(self):
        env=environment('constant-far')
        env.proprio=lambda state:torch.zeros(2,93)
        env.anchor_rel=lambda state:torch.zeros(2,9)
        env._depth_image=lambda state:(_ for _ in ()).throw(AssertionError('Critic must not read depth'))
        state={'qvel':torch.zeros(2,35),'friction':torch.ones(2),'enc_bias':torch.zeros(2,29),'com':torch.zeros(2,3)}
        result=env.critic_obs(state)
        self.assertTrue(torch.isfinite(result).all())

    def test_observation_contract_binds_camera_condition(self):
        measured=environment('measured');far=environment('constant-far')
        for env in (measured,far):
            env.hist_dim=1320;env.critic_dim=810
            env.cfg.action_limit=20.;env.cfg.source_reset_profile='nominal'
            env.cfg.source_depth_noise=True;env.cfg.residual_action=True
        measured_contract=measured.observation_contract
        far_contract=far.observation_contract
        self.assertNotEqual(measured_contract,far_contract)
        self.assertEqual(measured_contract['camera_input'],'measured')
        self.assertEqual(far_contract['camera_input'],'constant-far')
        self.assertFalse(far_contract['critic_depth_input'])
        for name in measured_contract:
            if name not in ('camera_input','camera_input_definition'):
                self.assertEqual(measured_contract[name],far_contract[name])


if __name__=='__main__':unittest.main()
