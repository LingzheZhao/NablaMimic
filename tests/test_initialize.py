import unittest

import jax
import numpy as np

from nablamimic.initialize import initialize_parameters


class InitializeTests(unittest.TestCase):
    def setUp(self):
        self.actor = {'weight': np.zeros((2, 3), np.float32)}
        self.critic = {'weight': np.zeros((3, 1), np.float32)}
        self.norm = (np.zeros(3, np.float32), np.ones(3, np.float32))
        self.checkpoint = dict(actor={'weight': np.full((2, 3), -3, np.float32)},
            actor_ema={'weight': np.full((2, 3), 7, np.float32)}, critic=self.critic, cnorm=self.norm,
            iter=19000, st_a='unused', st_c='unused', st_r='unused', bin_q='unused')

    def test_selects_ema_exactly_and_preserves_critic_normalizer(self):
        values = initialize_parameters(self.checkpoint, self.actor, self.critic, self.norm)
        for selected, expected in zip(values, [self.checkpoint['actor_ema'], self.critic, self.norm]):
            for x, y in zip(jax.tree_util.tree_leaves(selected), jax.tree_util.tree_leaves(expected)):
                np.testing.assert_array_equal(x, y)
        self.assertEqual(self.checkpoint['iter'], 19000)

    def test_shape_dtype_and_nonfinite_rejected(self):
        for bad in [np.zeros((3, 3), np.float32), np.zeros((2, 3), np.float64),
                    np.full((2, 3), np.nan, np.float32)]:
            with self.assertRaises(ValueError):
                initialize_parameters(dict(self.checkpoint, actor_ema={'weight': bad}),
                                      self.actor, self.critic, self.norm)


if __name__ == '__main__':
    unittest.main()
