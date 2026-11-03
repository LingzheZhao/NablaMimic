"""CPU behavioral contracts for position feedback routes; no physics instantiated."""
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from nablamimic.env import TrackingEnv, EnvConfig, FUTURE, GOAL_FRAME_DIM
from nablamimic import nets
from nablamimic.evaluate import validate_feedback


class PositionFeedbackTest(unittest.TestCase):
    def fixture(self, enabled, noise=True):
        env = object.__new__(TrackingEnv)
        env.cfg = EnvConfig(position_feedback=(enabled if isinstance(enabled, str) else "decoder" if enabled else "none"), obs_noise=noise)
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
            cfg = dict(position_feedback="decoder" if enabled else "none")
            checkpoint = dict(position_feedback="decoder" if enabled else "none", actor={'anchor_dec': jnp.zeros(k)}, actor_ema={'anchor_dec': jnp.zeros(k)})
            validate_feedback(checkpoint, cfg)
            with self.assertRaises(ValueError):
                validate_feedback(checkpoint, dict(position_feedback="none" if enabled else "decoder"))
            for name in ('actor', 'actor_ema'):
                bad = dict(checkpoint, **{name: {'anchor_dec': jnp.zeros(15-k)}})
                with self.assertRaises(ValueError):
                    validate_feedback(bad, cfg)
        validate_feedback({'actor': {}}, {})
        with self.assertRaises(ValueError):
            validate_feedback({'position_feedback': 'decoder', 'actor': {}}, {'position_feedback': 'decoder'})

    def test_real_critic_reward_and_termination_are_flag_invariant(self):
        baseline, state = self.fixture(False)
        feedback, _ = self.fixture("encoder")
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

    def test_config_opt_in_and_ema_preserves_nine_dimensional_marker(self):
        from pathlib import Path
        from nablamimic import train
        from nablamimic.params import read_params
        from nablamimic.policy_ema import init_policy_ema, update_policy_ema
        configs = Path(__file__).resolve().parents[1] / 'configs'
        def parse(name):
            return train.build_parser().parse_args(read_params(configs / name) + ['--out', 'unused'])
        self.assertEqual(EnvConfig().position_feedback, "none")
        self.assertEqual(parse('ours_ema.args').position_feedback, 'none')
        enabled = parse('ours_ema_position_feedback.args')
        self.assertEqual(enabled.position_feedback, 'decoder')
        self.assertTrue(enabled.anchor_to_decoder)
        p = nets.actor_init(jax.random.PRNGKey(2), SimpleNamespace(goal_dim=26, hist_dim=4, nu=2),
                            hidden=(8,), latent=3, anchor_to_decoder=9)
        ema, count = init_policy_ema(p)
        live = dict(p, anchor_dec=jnp.arange(9, dtype=jnp.float32))
        updated, _ = update_policy_ema(ema, live, count)
        np.testing.assert_array_equal(updated['anchor_dec'], live['anchor_dec'])

    def test_encoder_and_decoder_receive_identical_feedback_and_noise(self):
        for noise in (False, True):
            encoder, state = self.fixture("encoder", noise)
            decoder, _ = self.fixture("decoder", noise)
            key = jax.random.PRNGKey(13)
            np.testing.assert_array_equal(encoder.goal(state, key), decoder.goal(state, key))
            def position_goal(pos):
                dx = SimpleNamespace(xpos=pos[None], xquat=state.dx.xquat)
                return encoder.goal(SimpleNamespace(frame=state.frame, dx=dx), key)
            jac = jax.jacrev(position_goal)(state.dx.xpos[0])
            n = FUTURE * GOAL_FRAME_DIM
            np.testing.assert_allclose(jac[n:n+3], -np.eye(3), atol=1e-7)
            np.testing.assert_array_equal(jac[:n], 0.)
            np.testing.assert_array_equal(jac[n+3:], 0.)

    def test_position_routes_and_shared_initialization(self):
        env = SimpleNamespace(goal_dim=26, hist_dim=4, nu=2)
        key = jax.random.PRNGKey(9)
        encoder = nets.actor_init(key, env, hidden=(8, 5), latent=3, anchor_to_decoder=6)
        decoder = nets.actor_init(key, env, hidden=(8, 5), latent=3, anchor_to_decoder=9)
        for section in ("enc", "dec"):
            for a, b in zip(encoder[section][1:], decoder[section][1:]):
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)
        hist, goal = jnp.ones(4), jnp.arange(26, dtype=jnp.float32) / 30
        for actor, k in ((encoder, 6), (decoder, 9)):
            direct = lambda g: nets.actor_apply(actor, hist, g)
            split = lambda g: nets.actor_decode(actor, nets.actor_encode(actor, g[:-k]), hist, g[-k:])
            np.testing.assert_allclose(direct(goal), split(goal), atol=1e-7)
            gradient = jax.jacrev(direct)(goal)
            np.testing.assert_allclose(gradient, jax.jacrev(split)(goal), atol=1e-7)
            self.assertGreater(float(jnp.linalg.norm(gradient[:, -9:-6])), 0.)
        self.assertEqual(encoder['enc'][0][0].shape[0], decoder['enc'][0][0].shape[0] + 3)
        self.assertEqual(encoder['dec'][0][0].shape[0] + 3, decoder['dec'][0][0].shape[0])

    def test_checkpoint_encoder_route_and_live_ema_input_dimensions(self):
        from nablamimic.env import HIST, PROPRIO_DIM
        env = SimpleNamespace(goal_dim=FUTURE * GOAL_FRAME_DIM + 9, hist_dim=HIST * PROPRIO_DIM, nu=2)
        for route, k in (("encoder", 6), ("decoder", 9)):
            actor = nets.actor_init(jax.random.PRNGKey(0), env, hidden=(8,), latent=3, anchor_to_decoder=k)
            checkpoint = dict(position_feedback=route, actor=actor, actor_ema=actor)
            validate_feedback(checkpoint, dict(position_feedback=route))
            with self.assertRaises(ValueError):
                validate_feedback(checkpoint, dict(position_feedback="decoder" if route == "encoder" else "encoder"))
            for name in ("actor", "actor_ema"):
                wrong = dict(actor, enc=[(actor['enc'][0][0][:-3], actor['enc'][0][1]), *actor['enc'][1:]])
                with self.assertRaises(ValueError):
                    validate_feedback(dict(checkpoint, **{name: wrong}), dict(position_feedback=route))

    def test_encoder_hoist_is_rejected(self):
        from nablamimic import train
        env = SimpleNamespace(cfg=EnvConfig(position_feedback="encoder"))
        rollout = train.make_rollout(env, 2, gamma=.99, max_ep=100, remat=False, enc_hoist=True)
        from unittest.mock import patch
        with patch.object(train, 'vmapped', return_value=dict.fromkeys(
                ('policy_obs', 'critic_obs', 'step', 'reward', 'terminated', 'clip_ended'))):
            with self.assertRaisesRegex(ValueError, 'Cannot hoist'):
                rollout({}, None, None, SimpleNamespace(frame=jnp.array([0])), None, None, None)
