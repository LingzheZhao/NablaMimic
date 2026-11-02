"""Numerical checks for the paper's outer-update policy averaging rule."""

import pickle
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic.policy_ema import init_policy_ema, update_policy_ema


def actor_at(value):
    return {
        "enc": [(jnp.full((2, 3), value), jnp.full((3,), value + 1.0))],
        "dec": [(jnp.full((3, 1), value + 2.0), jnp.full((1,), value + 3.0))],
        "anchor_dec": jnp.zeros((6,)),
        "obs_norm": (jnp.full((2,), value + 4.0), jnp.full((2,), value + 5.0)),
    }


def assert_tree_equal(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(got, want)


def first_two_updates_use_completed_outer_update_count():
    initial = actor_at(0.0)
    ema, count = init_policy_ema(initial)
    assert int(count) == 0
    assert_tree_equal(ema, initial)
    ema, count = update_policy_ema(ema, actor_at(2.0), count)
    assert int(count) == 1
    np.testing.assert_allclose(ema["enc"][0][0], 1.8, rtol=1e-6)
    ema, count = update_policy_ema(ema, actor_at(4.0), count)
    assert int(count) == 2
    # The second decay is 2/11; the first is 1/10.
    np.testing.assert_allclose(ema["enc"][0][0], 3.6, rtol=1e-6)
    np.testing.assert_allclose(ema["dec"][0][1], 6.6, rtol=1e-6)
    assert_tree_equal(initial, actor_at(0.0))


def decay_caps_at_0999_and_updates_all_network_weights():
    ema, _ = init_policy_ema(actor_at(0.0))
    live = actor_at(10.0)
    averaged, count = jax.jit(update_policy_ema)(ema, live, jnp.int32(10000))
    assert int(count) == 10001
    for component in ("enc", "dec"):
        for old, got in zip(jax.tree_util.tree_leaves(ema[component]),
                            jax.tree_util.tree_leaves(averaged[component])):
            np.testing.assert_allclose(got, old + 0.01, atol=2e-6)


def nontrainable_actor_metadata_uses_current_live_values():
    ema, count = init_policy_ema(actor_at(0.0))
    live = actor_at(10.0)
    # Nonzero values make accidentally averaging this structural marker observable.
    live["anchor_dec"] = jnp.arange(6, dtype=jnp.float32)
    averaged, _ = update_policy_ema(ema, live, count)
    assert_tree_equal(averaged["anchor_dec"], live["anchor_dec"])
    assert_tree_equal(averaged["obs_norm"], live["obs_norm"])
    assert_tree_equal(live, {**actor_at(10.0), "anchor_dec": jnp.arange(6, dtype=jnp.float32)})


def serialized_average_and_count_continue_identically():
    ema, count = init_policy_ema(actor_at(-3.0))
    for value in (1.0, 4.0, -2.0):
        ema, count = update_policy_ema(ema, actor_at(value), count)
    checkpoint = pickle.loads(pickle.dumps({
        "actor": jax.device_get(actor_at(-2.0)),
        "actor_ema": jax.device_get(ema),
        "ema_updates": int(count),
    }))
    restored = jax.tree_util.tree_map(jnp.asarray, checkpoint["actor_ema"])
    expected, expected_count = update_policy_ema(ema, actor_at(7.0), count)
    actual, actual_count = update_policy_ema(restored, actor_at(7.0), checkpoint["ema_updates"])
    assert int(actual_count) == int(expected_count) == 4
    assert_tree_equal(actual, expected)
    assert_tree_equal(checkpoint["actor"], actor_at(-2.0))


class PolicyEMATest(unittest.TestCase):
    def test_first_two_updates_use_completed_outer_update_count(self):
        first_two_updates_use_completed_outer_update_count()

    def test_decay_caps_at_0999_and_updates_all_network_weights(self):
        decay_caps_at_0999_and_updates_all_network_weights()

    def test_nontrainable_actor_metadata_uses_current_live_values(self):
        nontrainable_actor_metadata_uses_current_live_values()

    def test_serialized_average_and_count_continue_identically(self):
        serialized_average_and_count_continue_identically()

    def test_unknown_actor_fields_are_rejected(self):
        for key in ("net", "unrecognized_weight"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                init_policy_ema({**actor_at(0.0), key: jnp.ones((1,))})


if __name__ == "__main__":
    unittest.main()
