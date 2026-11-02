"""Exercise the real differentiable rollout on tiny algebraic state transitions.

No robot, data library or physics engine is instantiated. Only the environment and
network interfaces are replaced; make_rollout, select_reset and TD(lambda) remain real.
Run with JAX_PLATFORMS=cpu in the pinned donor dependency environment.
"""

from contextlib import ExitStack
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic import nets, sim, train


class ScalarState(NamedTuple):
    frame: jax.Array
    ep: jax.Array
    prev_a: jax.Array
    x: jax.Array


def state_at(x, frame=0):
    return ScalarState(jnp.array([frame]), jnp.array([0]), jnp.zeros((1, 1)), jnp.array([x]))


def rollout_case(*, masked=False, end_kind="term", horizon=4, critic_gain=0.0,
                 reset_reward_scale=1.0):
    def step(state, action):
        return ScalarState(state.frame + 1, state.ep + 1, action, state.x + action[:, 0])

    def reward(state, action, previous_action, *, prev):
        scale = jnp.where(state.frame >= 100, reset_reward_scale, 1.0)
        value = scale * state.x
        return value, {"tracking": value}

    functions = {
        "policy_obs": lambda state, keys: (state.x[:, None], jnp.zeros((1, 1))),
        "critic_obs": lambda state: state.x[:, None],
        "step": step,
        "reward": reward,
        "terminated": lambda state: (state.ep >= 2) & (end_kind == "term"),
        "clip_ended": lambda state: (state.ep >= 2) & (end_kind == "clip"),
    }
    rollout = train.make_rollout(object(), horizon, gamma=0.5,
                                 max_ep=2 if end_kind == "timeout" else 100,
                                 remat=False, mask_after_done=masked, record_pre=True)
    actor = {"bias": jnp.array(1.0), "gain": jnp.array(0.25)}
    initial, reset = state_at(0.0), state_at(10.0, frame=100)

    def objective(delta):
        return rollout(actor, jnp.array(critic_gain), None, initial, reset,
                       jax.random.PRNGKey(0), None, delta=delta)

    perturbations = jnp.zeros((horizon, 1, 1))
    with ExitStack() as replacements:
        for module, name, value in (
            (train, "vmapped", lambda env, tables: functions),
            (train, "actor_apply", lambda actor, hist, goal: actor["bias"] + actor["gain"] * hist),
            (train, "critic_apply", lambda gain, obs: gain * obs),
            (train, "normalize", lambda norm, obs: obs),
            (sim, "STATE_SKIP", None),
        ):
            replacements.enter_context(patch.object(module, name, value))
        (loss, (_, records)), gradient = jax.value_and_grad(objective, has_aux=True)(perturbations)
    return -float(loss), -np.asarray(gradient[:, 0, 0]), records


class ResetWindowTest(unittest.TestCase):
    def test_reset_restarts_discount_and_keeps_remaining_window_when_unmasked(self):
        for masked, expected_return, expected_gradient in (
            (True, 2.125, [1.625, 0.5, 0.0, 0.0]),
            (False, 24.5625, [1.625, 0.5, 1.625, 0.5]),
        ):
            with self.subTest(masked=masked):
                value, gradient, records = rollout_case(masked=masked)
                self.assertAlmostEqual(value, expected_return)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6)
                np.testing.assert_array_equal(records["done"][:, 0], [False, True, False, True])
                # The critic sees the reset episode even with actor masking.
                np.testing.assert_allclose(records["cobs_pre"][:, 0, 0], [0.0, 1.0, 10.0, 13.5])

    def test_rewards_after_reset_do_not_change_old_episode_action_gradients(self):
        _, baseline, _ = rollout_case(reset_reward_scale=1.0)
        _, changed, _ = rollout_case(reset_reward_scale=7.0)
        np.testing.assert_array_equal(changed[:2], baseline[:2])
        np.testing.assert_allclose(changed[2:], 7.0 * baseline[2:], rtol=1e-6)

    def test_fall_has_no_bootstrap_and_truncations_use_pre_reset_next_state(self):
        for end_kind, expected_return, expected_gradient, second_target in (
            ("term", 29.125, [1.625, 0.5, 2.0], 2.25),
            ("timeout", 30.25, [2.25, 1.0, 2.0], 4.5),
            ("clip", 30.25, [2.25, 1.0, 2.0], 4.5),
        ):
            with self.subTest(end_kind=end_kind):
                value, gradient, records = rollout_case(end_kind=end_kind, horizon=3, critic_gain=2.0)
                self.assertAlmostEqual(value, expected_return)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6)
                # V(s2)=4.5 comes before reset; V(reset)=20 must not replace it.
                self.assertAlmostEqual(float(records["v2"][1, 0]), 4.5)
                targets = nets.td_lambda(records["r"], records["term"], records["done"],
                                         records["v2"], gamma=0.5, lam=1.0)
                self.assertAlmostEqual(float(targets[1, 0]), second_target)
                self.assertAlmostEqual(float(targets[2, 0]), 27.0)


if __name__ == "__main__":
    unittest.main()
