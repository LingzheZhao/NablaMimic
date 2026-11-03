"""CPU behavioral contracts for decoder-only position feedback; no physics instantiated."""
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from nablamimic.env import TrackingEnv, EnvConfig, FUTURE, GOAL_FRAME_DIM, validate_feedback
from nablamimic import nets


class PositionFeedbackTest(unittest.TestCase):
    def fixture(self, enabled, noise=True):
        env = object.__new__(TrackingEnv)
        env.cfg = EnvConfig(decoder_position_feedback=enabled, obs_noise=noise)
        env.torso_id = env.anchor_k = 0
        env.ref_body_pos = jnp.array([[[2., 3., 1.]]])
        env.ref_body_quat = jnp.array([[[1., 0., 0., 0.]]])
        env.ref_last = jnp.array([0])
        env.goal_feat = jnp.arange(GOAL_FRAME_DIM, dtype=jnp.float32)[None]
        state = SimpleNamespace(frame=jnp.array(0), dx=SimpleNamespace(
            xpos=jnp.array([[0., 0., 1.]]), xquat=jnp.array([[1., 0., 0., 0.]])))
        return env, state

    def test_translation_changes_only_added_position_and_preserves_noise(self):
        for noise in (False, True):
            baseline, state = self.fixture(False, noise)
            feedback, _ = self.fixture(True, noise)
            key = jax.random.PRNGKey(7)
            before = feedback.goal(state, key)
            original = baseline.goal(state, key)
            n = FUTURE * GOAL_FRAME_DIM
            np.testing.assert_array_equal(before[:n], original[:n])
            np.testing.assert_array_equal(before[n+3:], original[n:])
            state.dx.xpos = state.dx.xpos + jnp.array([[.7, -.4, 0.]])
            after = feedback.goal(state, key)
            np.testing.assert_allclose(after[n:n+3] - before[n:n+3], [-.7, .4, 0.], atol=1e-6)
            np.testing.assert_array_equal(after[n+3:], before[n+3:])
            np.testing.assert_array_equal(baseline.goal(state, key), original)

    def test_hoisted_encoder_action_and_input_gradient_parity(self):
        for k in (6, 9):
            env = SimpleNamespace(goal_dim=17+k, hist_dim=4, nu=2)
            p = nets.actor_init(jax.random.PRNGKey(2), env, hidden=(8,), latent=3, anchor_to_decoder=k)
            hist = jnp.arange(4, dtype=jnp.float32) / 4
            goal = jnp.arange(17+k, dtype=jnp.float32) / 30
            direct = lambda g: nets.actor_apply(p, hist, g)
            hoisted = lambda g: nets.actor_decode(p, nets.actor_encode(p, g[:-k]), hist, g[-k:])
            np.testing.assert_allclose(direct(goal), hoisted(goal), atol=1e-7)
            np.testing.assert_allclose(jax.jacrev(direct)(goal), jax.jacrev(hoisted)(goal), atol=1e-7)
            self.assertGreater(float(jnp.linalg.norm(jax.jacrev(direct)(goal)[:, -k:])), 0.)

    def test_checkpoint_feedback_flag_and_both_actor_dimensions(self):
        for enabled in (False, True):
            k = 9 if enabled else 6
            cfg = dict(decoder_position_feedback=enabled)
            checkpoint = dict(decoder_position_feedback=enabled, actor={'anchor_dec': jnp.zeros(k)}, actor_ema={'anchor_dec': jnp.zeros(k)})
            validate_feedback(checkpoint, cfg)
            with self.assertRaises(ValueError):
                validate_feedback(checkpoint, dict(decoder_position_feedback=not enabled))
            for name in ('actor', 'actor_ema'):
                bad = dict(checkpoint, **{name: {'anchor_dec': jnp.zeros(15-k)}})
                with self.assertRaises(ValueError):
                    validate_feedback(bad, cfg)
        validate_feedback({'decoder_position_feedback': False, 'actor': {}},
                          {'decoder_position_feedback': False})
        with self.assertRaises(KeyError):
            validate_feedback({'actor': {}}, {'decoder_position_feedback': False})
        with self.assertRaises(ValueError):
            validate_feedback({'decoder_position_feedback': True, 'actor': {}}, {'decoder_position_feedback': True})

    def test_real_critic_reward_and_termination_are_flag_invariant(self):
        baseline, state = self.fixture(False)
        feedback, _ = self.fixture(True)
        for env in (baseline, feedback):
            env.nu = 29
            env.q0 = jnp.zeros(29)
            env.body_ids = jnp.array([0])
            env.root_com_id = 0
            env.ref_body_lin = jnp.zeros((1, 1, 3))
            env.ref_body_ang = jnp.zeros((1, 1, 3))
            env.soft_hi, env.soft_lo = jnp.ones(29), -jnp.ones(29)
            env.ref_qpos = jnp.concatenate([jnp.array([0., 0., 1., 1., 0., 0., 0.]), jnp.zeros(29)])[None]
        state.dx.qpos = baseline.ref_qpos[0]
        state.dx.qvel = jnp.arange(35, dtype=jnp.float32) / 100
        state.dx.cvel = jnp.arange(6, dtype=jnp.float32)[None] / 10
        state.dx.subtree_com = jnp.zeros((1, 3))
        state.xi = SimpleNamespace(friction=jnp.array(.8), enc_bias=jnp.zeros(29), com=jnp.zeros(3))
        state.prev_a = jnp.zeros(29)
        action = jnp.ones(29) * .1
        for height, expected_fall in ((1., False), (1.5, True)):
            state.dx.qpos = state.dx.qpos.at[2].set(height)
            old = baseline.critic_obs(state)
            new = feedback.critic_obs(state)
            self.assertEqual(old.shape, new.shape)
            np.testing.assert_array_equal(new, old)
            reward0, terms0 = baseline.reward(state, action, state.prev_a)
            reward1, terms1 = feedback.reward(state, action, state.prev_a)
            np.testing.assert_array_equal(reward1, reward0)
            for term in terms0:
                np.testing.assert_array_equal(terms1[term], terms0[term])
            self.assertEqual(bool(baseline.terminated(state)), expected_fall)
            self.assertEqual(bool(feedback.terminated(state)), expected_fall)

    def test_config_opt_in_uses_current_upstream_recipe(self):
        from pathlib import Path
        from nablamimic import train
        from nablamimic.params import read_params
        configs = Path(__file__).resolve().parents[1] / 'configs'
        def parse(name):
            return train.build_parser().parse_args(read_params(configs / name) + ['--out', 'unused'])
        self.assertFalse(EnvConfig().decoder_position_feedback)
        self.assertFalse(parse('ours.args').decoder_position_feedback)
        enabled = parse('ours_position_feedback.args')
        self.assertTrue(enabled.decoder_position_feedback)
        self.assertTrue(enabled.anchor_to_decoder)
        self.assertEqual(enabled.actor_ema, .999)
        self.assertTrue(enabled.actor_ema_warmup)
        self.assertFalse(enabled.mask_after_done)
        self.assertEqual(enabled.lr_warmup_steps, 100)
