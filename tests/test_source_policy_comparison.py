"""Generated display support alignment with partial physical trajectories."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from render_source_policy_comparison import comparison_support, depth_for_state, frame_labels
sys.path.pop(0)


def reading(controls):
    return dict(scene_artifact_id='scene', terrain_artifact_id='terrain', reference_artifact_id='reference',
                physics_backend='engine', ray_backend='native', control_hz=50., start_frame=100,
                clip_start_frame=100, start_fraction=0., start_offset=0, requested_controls=10,
                controls=controls, states=controls+1, final_frame=100+controls,
                sensor_noise='on', seed=0, camera_input='measured', reset={}, physics={}, initial_pose_offset={})


class SourcePolicyComparisonTest(unittest.TestCase):
    def test_partial_chain_keeps_terminal_and_remaining_display_support(self):
        rows = [reading(10), reading(4), reading(7)]
        self.assertEqual(comparison_support(rows), 11)
        self.assertEqual([11-row['states'] for row in rows], [0, 6, 3])
        self.assertEqual(rows[1]['final_frame'], 104)

    def test_misaligned_clock_start_and_frame_count_are_rejected(self):
        for changed in (dict(control_hz=25.), dict(start_frame=101), dict(requested_controls=9),
                        dict(states=4), dict(final_frame=999), dict(physics_backend='mujoco')):
            with self.assertRaises(ValueError):
                comparison_support([reading(10), dict(reading(4), **changed)])

    def test_recorded_input_depth_is_never_repeated_at_terminal(self):
        depth = np.arange(4*576).reshape(4, 576)
        np.testing.assert_array_equal(depth_for_state(depth, 3, 4), depth[3].reshape(18, 32))
        self.assertIsNone(depth_for_state(depth, 4, 4))
        self.assertIsNone(depth_for_state(None, 0, 4))

    def test_labels_and_saved_depth_render_without_font_dependencies(self):
        report = dict(controls=4, mean_world_body_error_m=.1, mean_torso_error_m=.2,
                      reference_complete=False, strict_pass=False)
        depth = np.ones((4, 576))
        transform = frame_labels('RENDERFITV2', report, depth)
        pixels = transform(np.zeros((720, 960, 3), dtype=np.uint8), 0)
        self.assertEqual(pixels.shape, (720, 960, 3))
        self.assertEqual(int(pixels[600, 800, 0]), 255)
        terminal = transform(np.zeros_like(pixels), 4)
        self.assertFalse(np.array_equal(pixels, terminal))


if __name__ == '__main__':
    unittest.main()
