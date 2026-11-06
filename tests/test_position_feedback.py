"""CPU behavioral contracts for decoder-only position feedback; no physics instantiated."""
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from nablamimic.env import (TrackingEnv, EnvConfig, FUTURE, GOAL_FRAME_DIM,
                           position_feedback_enabled, validate_feedback)
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

    def test_native_actor_payloads_use_explicit_modes_without_checkpoint_flags(self):
        for enabled in (False, True):
            k = 9 if enabled else 6
            cfg = dict(obs_mode="anchor_pos_ori" if enabled else "anchor_ori", actor_latent=3,
                       anchor_to_decoder=True)
            env = SimpleNamespace(goal_dim=FUTURE*GOAL_FRAME_DIM+k, hist_dim=930, nu=2)
            actor = nets.actor_init(jax.random.PRNGKey(3), env, hidden=(8,), latent=3, anchor_to_decoder=k)
            checkpoint = dict(actor=actor, actor_ema=actor)
            self.assertEqual(position_feedback_enabled(cfg), enabled)
            validate_feedback(checkpoint, cfg)
            validate_feedback(checkpoint, dict(cfg, decoder_position_feedback=enabled))
            with self.assertRaises(ValueError):
                validate_feedback(checkpoint, dict(cfg, decoder_position_feedback=not enabled))
            with self.assertRaises(ValueError):
                validate_feedback(checkpoint, dict(cfg, obs_mode="anchor_ori" if enabled else "anchor_pos_ori"))
            for name in ('actor', 'actor_ema'):
                bad = dict(checkpoint, **{name: dict(actor, anchor_dec=jnp.zeros(15-k))})
                with self.assertRaises(ValueError):
                    validate_feedback(bad, cfg)
                first, bias = actor["dec"][0]
                wrong_decoder = [(jnp.zeros((first.shape[0]+3, first.shape[1])), bias), *actor["dec"][1:]]
                bad = dict(checkpoint, **{name: dict(actor, dec=wrong_decoder)})
                with self.assertRaises(ValueError):
                    validate_feedback(bad, cfg)
                bad_actor = dict(actor)
                del bad_actor["anchor_dec"]
                with self.assertRaises(ValueError):
                    validate_feedback(dict(checkpoint, **{name: bad_actor}), cfg)

    def test_unknown_or_inconsistent_observation_modes_are_rejected(self):
        with self.assertRaises(KeyError):
            position_feedback_enabled({})
        for cfg in (dict(obs_mode="unknown"), dict(obs_mode="anchor_ori", decoder_position_feedback=True),
                    dict(obs_mode="anchor_pos_ori", decoder_position_feedback=False),
                    dict(obs_mode="anchor_ori", decoder_position_feedback=0),
                    dict(obs_mode="anchor_pos_ori"), dict(obs_mode="anchor_pos_ori", anchor_to_decoder=False),
                    dict(obs_mode="anchor_ori", anchor_to_decoder=1)):
            with self.assertRaises(ValueError):
                position_feedback_enabled(cfg)
        with self.assertRaises(ValueError):
            validate_feedback(dict(actor={}), dict(obs_mode='anchor_ori'))

    def test_current_ppo_actors_do_not_require_decoder_feedback_metadata(self):
        from nablamimic import ppo
        dimensions = SimpleNamespace(goal_dim=686, hist_dim=930, nu=29)
        history, goal = jnp.zeros(930), jnp.zeros(686)
        for architecture in ('mlp', 'supermimic'):
            args = ppo.build_parser().parse_args(['--out', 'unused', '--arch', architecture, '--hidden', '8'])
            cfg = dict(vars(args), **ppo.FIXED, algo='ppo', devices=1,
                       git_commit='test-source', git_dirty=False, gpu='CPU fixture')
            if architecture == 'mlp':
                actor = dict(net=nets.mlp_init(jax.random.PRNGKey(11), [1616, 8, 29]),
                             obs_norm=(jnp.zeros(1616), jnp.ones(1616)))
            else:
                actor = nets.actor_init(jax.random.PRNGKey(12), dimensions, hidden=(8,), latent=32)
            # Native PPO evaluation payload: actor/critic/cnorm alongside the
            # producer's optimization and observation-normalizer records.
            checkpoint = dict(actor=actor, critic=[], cnorm=(jnp.zeros(825), jnp.ones(825)),
                params={}, opt_state=(), onorm=(), cnorm_full=(), bin_q=jnp.ones((2, 1)),
                iter=0, wall_s=0., env_steps=0)
            self.assertNotIn('actor_latent', cfg)
            self.assertNotIn('anchor_to_decoder', cfg)
            self.assertFalse(position_feedback_enabled(cfg))
            validate_feedback(checkpoint, cfg)
            self.assertEqual(nets.actor_apply(actor, history, goal).shape, (29,))

    def test_current_default_and_shac_non_anchor_actors_remain_valid(self):
        from pathlib import Path
        from nablamimic import train
        from nablamimic.params import read_params
        configs = Path(__file__).resolve().parents[1] / 'configs'
        choices = [[], read_params(configs / 'shac.args')]
        dimensions = SimpleNamespace(goal_dim=686, hist_dim=930, nu=29)
        for values in choices:
            args = train.build_parser().parse_args([*values, '--out', 'unused'])
            cfg = dict(vars(args), **train.FIXED, devices=1, processes=1,
                       git_commit='test-source', git_dirty=False, gpu='CPU fixture')
            actor = nets.actor_init(jax.random.PRNGKey(13), dimensions, hidden=(8,), latent=32)
            if args.actor_obs_norm:
                actor['obs_norm'] = (jnp.zeros(1616), jnp.ones(1616))
            checkpoint = dict(actor=actor, critic=[], cnorm=(jnp.zeros(825), jnp.ones(825)),
                st_a=(), st_c=(), bin_q=jnp.ones((2, 1)), iter=0, wall_s=0., env_steps=0)
            self.assertFalse(cfg['anchor_to_decoder'])
            self.assertNotIn('anchor_dec', actor)
            validate_feedback(checkpoint, cfg)
            self.assertEqual(nets.actor_apply(actor, jnp.zeros(930), jnp.zeros(686)).shape, (29,))

    def test_position_uses_measured_torso_rotation(self):
        env, state = self.fixture(True, noise=False)
        state.dx.xquat = jnp.array([[2**-.5, 0., 0., 2**-.5]])
        offset = FUTURE * GOAL_FRAME_DIM
        np.testing.assert_allclose(env.goal(state)[offset:offset+3], [3., -2., 0.], atol=1e-6)

    def test_existing_positional_env_config_argument_is_preserved(self):
        cfg = EnvConfig(7)
        self.assertEqual(cfg.solver_iterations, 7)
        self.assertFalse(cfg.decoder_position_feedback)

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
