"""Focused CPU checks for shared-state perturbations and measured support."""
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

import recovery
from nablamimic import nets
from nablamimic.env import EnvConfig, TrackingEnv, Xi, FUTURE, GOAL_FRAME_DIM, HIST, PROPRIO_DIM


def fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
      <body pos="0 0 .2"><joint name="hinge"/><geom type="sphere" size=".05" mass=".1"/></body>
      </body></worldbody><actuator><position joint="hinge"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = object.__new__(TrackingEnv)
    env.mx = mjx.put_model(model)
    env.d0 = mjx.put_data(model, data)
    env.model_for = lambda xi: env.mx
    env.cfg = EnvConfig(position_feedback="encoder", dr=False, obs_noise=False, reset_noise=False,
                        residual_action=True)
    env.anchor_k, env.torso_id = 0, 1
    env.ref_body_pos = jnp.tile(jnp.array([[[1., 2., 1.]]]), (301, 1, 1))
    env.ref_body_quat = jnp.tile(jnp.array([[[1., 0., 0., 0.]]]), (301, 1, 1))
    env.ref_last = jnp.full(301, 300)
    env.goal_feat = jnp.arange(301 * GOAL_FRAME_DIM, dtype=jnp.float32).reshape(301, GOAL_FRAME_DIM) / 1000
    env.ref_qpos = jnp.tile(env.d0.qpos[None], (301, 1))
    env.action_scale, env.ctrl_lo, env.ctrl_hi = jnp.ones(1) * .2, -jnp.ones(1), jnp.ones(1)
    compact = dict(qpos=env.d0.qpos[None], qvel=jnp.ones((1, model.nv)) * .1,
                   qacc_warmstart=jnp.ones((1, model.nv)) * .25, frame=jnp.array([50]), ep=jnp.array([50]),
                   hist=jnp.arange(HIST * PROPRIO_DIM, dtype=jnp.float32).reshape(1, HIST, PROPRIO_DIM) / 1000,
                   prev_a=jnp.array([[.2]]), xi=Xi(jnp.ones(1), jnp.zeros((1, 1)), jnp.zeros((1, 3))),
                   key=jax.random.split(jax.random.PRNGKey(7), 1))
    return env, compact


class RecoveryTest(unittest.TestCase):
    def test_shared_state_and_translation_recompute_geometry(self):
        env, compact = fixture()
        reference_before = np.asarray(env.ref_body_pos).copy()
        baseline = jax.vmap(env.restore)(recovery.translated_compact(compact, 0.0))
        shifted = jax.vmap(env.restore)(recovery.translated_compact(compact, .30))
        recovery.check_branch_inputs(env, baseline, shifted, .30)
        with self.assertRaises(AssertionError):
            recovery.check_branch_inputs(env, baseline, shifted._replace(frame=shifted.frame + 1), .30)
        with self.assertRaises(AssertionError):
            recovery.check_branch_inputs(env, baseline, shifted._replace(key=shifted.key + 1), .30)
        for name in ("qvel", "qacc_warmstart"):
            np.testing.assert_array_equal(getattr(baseline.dx, name), getattr(shifted.dx, name))
        for name in ("hist", "prev_a", "frame", "ep", "key"):
            np.testing.assert_array_equal(getattr(baseline, name), getattr(shifted, name))
        np.testing.assert_array_equal(baseline.dx.xquat, shifted.dx.xquat)
        # World body 0 stays fixed. Both movable LINK origins and COMs translate together.
        for name in ("xpos", "subtree_com"):
            difference = np.asarray(getattr(shifted.dx, name) - getattr(baseline.dx, name))
            np.testing.assert_allclose(difference[:, 1:, 0], .30, atol=1e-6)
            np.testing.assert_allclose(difference[:, 1:, 1:], 0., atol=1e-6)
        np.testing.assert_array_equal(env.ref_body_pos, reference_before)
        for route, k in (("encoder", 6), ("decoder", 9)):
            actor = nets.actor_init(jax.random.PRNGKey(4),
                                   SimpleNamespace(goal_dim=FUTURE * GOAL_FRAME_DIM + 9,
                                                        hist_dim=HIST * PROPRIO_DIM, nu=1),
                                   hidden=(8,), latent=3, anchor_to_decoder=k)
            before = recovery.policy_probe(actor, env, baseline)
            after = recovery.policy_probe(actor, env, shifted)
            np.testing.assert_array_equal(before['history'], after['history'])
            np.testing.assert_array_equal(before['goal'][:, :-9], after['goal'][:, :-9])
            np.testing.assert_array_equal(before['goal'][:, -6:], after['goal'][:, -6:])
            np.testing.assert_allclose(after['goal'][:, -9:-6] - before['goal'][:, -9:-6],
                                       [[-.30, 0., 0.]], atol=1e-6)
            if route == 'decoder':
                np.testing.assert_array_equal(before['latent'], after['latent'])
            corrected = after['goal'].at[:, -9:-6].set(before['goal'][:, -9:-6])
            np.testing.assert_array_equal(nets.actor_apply(actor, before['history'], corrected), before['action'])

    def test_short_motion_full_horizon_and_failure_accounting(self):
        self.assertEqual(recovery.planned_controls(50, 285), 235)
        short = recovery.summarize([True] * 235, [False] * 235, 235, 235)
        self.assertTrue(short['reference_completed'])
        self.assertFalse(short['full_five_second_metrics_available'])
        self.assertEqual(short['end'], 'reference_completed')
        final_failure = recovery.summarize([True] * 235, [False] * 234 + [True], 235, 235)
        self.assertTrue(final_failure['reference_end_reached'])
        self.assertFalse(final_failure['reference_completed'])
        self.assertEqual(final_failure['end'], 'native_termination')
        full = recovery.summarize([True] * 250, [False] * 250, 250, 900)
        self.assertTrue(full['full_five_second_metrics_available'])
        failed = recovery.summarize([True] * 4, [False] * 3 + [True], 250, 900)
        self.assertEqual(failed['executed_controls'], 4)
        self.assertTrue(failed['native_termination'])
        self.assertFalse(failed['full_five_second_metrics_available'])
        partial = recovery.summarize([True, False], [False, False], 250, 900)
        self.assertEqual(partial['valid_controls'], 1)
        self.assertEqual(partial['end'], 'invalid_physics')
        self.assertEqual(recovery.summarize([], [], 235, 235)['end'], 'partial')

    def test_transition_records_queried_latent_and_executed_action(self):
        env, compact = fixture()
        state = jax.vmap(env.restore)(compact)
        actor = nets.actor_init(jax.random.PRNGKey(2),
                               SimpleNamespace(goal_dim=FUTURE * GOAL_FRAME_DIM + 9,
                                                    hist_dim=HIST * PROPRIO_DIM, nu=1),
                               hidden=(8,), latent=3, anchor_to_decoder=6)
        expected = recovery.policy_probe(actor, env, state)
        functions = dict(step=lambda state, action: state._replace(frame=state.frame + 1, prev_a=action),
                         reward=lambda state, action, previous, prev: (jnp.zeros(1), {'tracking': jnp.ones(1)}),
                         terminated=lambda state: jnp.zeros(1, bool))
        record = dict(terms=('tracking',), record=lambda env, state, active, terminated, prev, terms:
                      dict(frame=state.frame, finite=jnp.ones(1, bool), terminated=terminated,
                           action=state.prev_a))
        with patch.object(recovery.sim, 'vmapped', return_value=functions):
            next_state, trace = recovery.make_transition(env, record)(actor, state, {})
        np.testing.assert_array_equal(trace['latent'], expected['latent'])
        np.testing.assert_array_equal(trace['policy_action'], expected['action'])
        np.testing.assert_array_equal(trace['action'], next_state.prev_a)
        np.testing.assert_array_equal(next_state.frame, state.frame + 1)


    def test_paired_metric_support_retains_invalid_and_short_conditions(self):
        def row(frame, valid, error):
            metrics = np.full((1, 23), error)
            return dict(frame=np.array([frame]), valid=np.array([valid]), metrics=metrics)
        sham = [row(51 + i, True, 1.) for i in range(25)]
        perturbed = [row(51 + i, i != 24, 1.5) for i in range(25)]
        report = recovery.paired_diagnostic(sham, perturbed)
        self.assertEqual(report['shared_valid_prefix_controls'], 24)
        self.assertEqual(report['perturbed_executed_controls'], 25)
        self.assertFalse(report['full_five_seconds_available'])
        body = report['excess_errors']['world_body_mpjpe_m']
        self.assertAlmostEqual(body['shared_prefix_mean'], .5)
        self.assertAlmostEqual(body['shared_prefix_integral'], .24)
        self.assertIsNone(body['at_seconds']['0.5'])
        self.assertIsNone(recovery.paired_diagnostic([], sham)['excess_errors']['torso_xy_m']['shared_prefix_mean'])
        wrong = [row(52, True, 1.5)]
        with self.assertRaisesRegex(ValueError, 'reference frames'):
            recovery.paired_diagnostic(sham, wrong)

    def test_capacity_failure_flushes_returned_physical_sample(self):
        trace = dict(frame=jnp.array([51]), finite=jnp.ones(1, bool), terminated=jnp.zeros(1, bool))
        audit = SimpleNamespace(reset=lambda: None)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.npz'
            with patch.object(recovery, 'capacity_record', side_effect=RuntimeError('counter unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'counter unavailable'):
                    recovery.advance({}, jnp.array(0), 3, lambda actor, state: (state + 1, trace), audit, path)
            with np.load(path) as data:
                np.testing.assert_array_equal(data['frame'], [51])
                np.testing.assert_array_equal(data['capacity_observed'], [False])
                np.testing.assert_array_equal(data['valid'], [False])
            self.assertTrue(path.with_suffix('.failure.json').exists())


if __name__ == '__main__':
    unittest.main()
