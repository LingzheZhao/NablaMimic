"""Generated wrench buffers check episode timing without a simulator."""
from types import SimpleNamespace
import unittest

import numpy as np

from nablamimic.source_force_pulse import TorsoWorldForcePulse


class SourceForcePulseTest(unittest.TestCase):
    def test_boundaries_zero_torque_and_impulse(self):
        for force in ((50., 0., 0.), (0., 50., 0.)):
            pulse = TorsoWorldForcePulse(2, force)
            data = SimpleNamespace(xfrc_applied=np.zeros((4, 6)))
            samples = []
            for substep in range(624):
                # The adapter owns clearing, including immediately after the pulse.
                data.xfrc_applied[:] = 0.
                pulse(0, substep, data)
                samples.append(data.xfrc_applied.copy())
            samples = np.stack(samples)
            self.assertFalse(samples[:600].any())
            np.testing.assert_array_equal(samples[600:620, 2, :3], np.tile(force, (20, 1)))
            self.assertFalse(samples[620:].any())
            self.assertFalse(samples[:, :, 3:].any())
            self.assertFalse(samples[:, [0, 1, 3]].any())
            np.testing.assert_allclose(samples[:, 2, :3].sum(0) * .005, np.array(force) * .1)

    def test_worlds_have_same_episode_clock_and_zero_condition(self):
        for world in (0, 1):
            data = SimpleNamespace(xfrc_applied=np.zeros((3, 6)))
            TorsoWorldForcePulse(1, (0., 0., 0.))(world, 600, data)
            self.assertFalse(data.xfrc_applied.any())
            TorsoWorldForcePulse(1, (50., 0., 0.))(world, 600, data)
            self.assertEqual(data.xfrc_applied[1, 0], 50.)

    def test_invalid_parameters(self):
        for kwargs in ({'torso_body_id': 0}, {'start_control': -1},
                       {'duration_controls': 0}, {'start_control': 1.5},
                       {'force_world': (1., 2.)}, {'force_world': (float('nan'), 0., 0.)}):
            parameters = dict(torso_body_id=1, force_world=(50., 0., 0.))
            parameters.update(kwargs)
            with self.assertRaises(ValueError):
                TorsoWorldForcePulse(**parameters)


if __name__ == '__main__':
    unittest.main()
