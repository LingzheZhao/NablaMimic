"""Fixed reset-start mass and actual selected-state counting contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest
from nablamimic.sampler import fixed_clip_mass, clip_frame_counts


def check_fixed_mass(mass):
    p = jnp.array([.01, .09, 0., .2, .1, .6, 0.], dtype=jnp.float32)
    mask = jnp.array([False, False, False, True, True, True, True])
    q = np.asarray(jax.jit(fixed_clip_mass)(p, mask, mass))
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(), 1, atol=2e-7)
    np.testing.assert_allclose(q[np.asarray(mask)].sum(), mass, atol=2e-7)
    assert q[2] == q[6] == 0
    np.testing.assert_allclose(q[0] / q[1], .01 / .09, rtol=2e-7)
    np.testing.assert_allclose(q[3] / q[5], .2 / .6, rtol=2e-7)


def test_reset_counts_include_only_selected_proposals_and_counts_keep_clock_distinctions():
    owner = jnp.array([0, 0, 1, 1, 1, 2])
    proposals = jnp.array([5, 5, 0, 2])
    selected = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    np.testing.assert_array_equal(clip_frame_counts(proposals, owner, 3, selected), [0, 1, 1])
    incoming_frames = jnp.array([0, 0, 2, 2])
    next_frames = jnp.array([1, 5, 3, 2])
    np.testing.assert_array_equal(clip_frame_counts(incoming_frames, owner, 3), [2, 2, 0])
    np.testing.assert_array_equal(clip_frame_counts(next_frames, owner, 3), [1, 2, 1])


class FixedClipMassTests(unittest.TestCase):
    def test_five_percent(self):
        check_fixed_mass(.05)

    def test_fifteen_percent(self):
        check_fixed_mass(.15)

    def test_selected_counts(self):
        test_reset_counts_include_only_selected_proposals_and_counts_keep_clock_distinctions()
