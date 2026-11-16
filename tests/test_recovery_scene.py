"""The scene program of the recovery pair (studies/recovery/scene.py) on a small synthetic library; no assets, engine
or GPU.

Library (clips in this order): getup (200 rows: rows 0-149 lying at pelvis height 0.2 m, rows 150-199 upright at
0.78 m), walk (1,300 upright rows at 0.78 m) and sway (100 upright rows at 0.78 m). getup is the hidden clip and getup
and sway are command-excluded, so the start rows are rows 0-149 and the rows of walk (rows 200-1499) with 1,000 upright
rows ahead before its last row are rows 200-498."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from nablamimic.recovery import RecoverySettings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"studies/recovery"))  # the study's programs import declaration.py beside them
SPEC = importlib.util.spec_from_file_location("recovery_scene", ROOT/"studies/recovery/scene.py")
SCENE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENE)

CLIPS = (("getup", 200), ("walk", 1300), ("sway", 100))
# The draws at seed 7, written out: the uniforms and the permutation depend only on the seed and the order of the
# draws, not on the library.
POSITION_0 = [0.06984508037567139, -0.602393627166748, 0.31842339038848877]
ROTATION_0 = [0.8734664916992188, 0.10339879989624023, -0.34344375133514404]
VELOCITY_50 = [0.9219750165939331, 0.7520275115966797, -0.1744670867919922, -0.35515522956848145,
               -0.5345755815505981, 0.8204742670059204]
JOINTS_99_LAST_TWO = [-0.8979747295379639, -0.5212045907974243]
PERMUTATION = [5, 27, 83, 17, 66, 80, 61, 49, 50, 6, 39, 20, 30, 41, 99, 33, 21, 0, 90, 40, 11, 87, 15, 93, 62, 14, 16,
               43, 75, 31, 72, 8, 42, 53, 4, 22, 3, 51, 84, 1, 18, 97, 45, 44, 60, 13, 58, 19, 7, 64, 23, 81, 67, 71,
               26, 70, 12, 9, 59, 69, 78, 86, 10, 98, 54, 88, 77, 52, 48, 68, 95, 63, 89, 92, 25, 29, 32, 73, 91, 46,
               57, 38, 55, 36, 24, 37, 96, 2, 85, 65, 34, 47, 56, 94, 74, 76, 35, 79, 28, 82]


def library():
    qpos = np.zeros((sum(rows for _, rows in CLIPS), 36), np.float32)
    qpos[:, 3] = 1
    qpos[:, 2] = 0.78
    qpos[:150, 2] = 0.2
    starts = np.cumsum([0] + [rows for _, rows in CLIPS[:-1]])
    last = np.concatenate([np.full(rows, start + rows - 1) for (_, rows), start in zip(CLIPS, starts)])
    return SimpleNamespace(qpos=qpos, last=last, starts=starts, names=[name for name, _ in CLIPS])


def settings():
    return RecoverySettings(share=0.5, hidden_clips=("getup",), command_excluded_clips=("getup", "sway"),
                            low_pelvis_height=0.5, low_pelvis_up_z=0.5, upright_run_rows=50, longest_span_rows=375,
                            command_upright_rows=500, reward_scale=2.5, stop_controls=100, blend_controls=75)


def placed(**changes):
    return {**SCENE.read_declaration(ROOT/"studies/recovery/pair.json")["placed_starts"], **changes}


class SceneTest(unittest.TestCase):
    def test_the_files_hold_the_declared_trials_twins_and_every_clip_s_first_row(self):
        files, record = SCENE.scene(library(), settings(), placed())
        self.assertEqual(sorted(files), ["first_rows.json", "placed.json", "twins.json"])
        placed_trials, twins = files["placed.json"]["trials"], files["twins.json"]["trials"]
        self.assertEqual((len(placed_trials), len(twins)), (100, 100))
        self.assertEqual([trial["command_row"] for trial in twins], [trial["command_row"] for trial in placed_trials])
        self.assertTrue(all(set(trial) == {"command_row"} for trial in twins))
        self.assertEqual(files["first_rows.json"]["trials"], [{"command_row": 0}, {"command_row": 200},
                                                              {"command_row": 1500}])
        self.assertEqual(record["first_row_controls"], 1299)
        self.assertEqual([clip["controls"] for clip in record["clips"]], [199, 1299, 99])
        self.assertEqual(record["rows"]["start_rows"], 150)
        self.assertEqual(record["command_rows_with_upright_rows_ahead"], 299)
        for trial in placed_trials:
            self.assertEqual(sorted(trial["reset_uniforms"]), ["joints", "position", "rotation", "velocity"])
            self.assertTrue(all(-1 <= value < 1 for values in trial["reset_uniforms"].values() for value in values))
        json.dumps(files, allow_nan=False)
        json.dumps(record, allow_nan=False)

    def test_body_and_command_rows_are_evenly_spaced_and_paired_by_the_drawn_permutation(self):
        files, record = SCENE.scene(library(), settings(), placed())
        body = [trial["body_row"] for trial in files["placed.json"]["trials"]]
        self.assertEqual(body, np.round(np.linspace(0, 149, 100)).astype(int).tolist())
        self.assertTrue(set(np.diff(body)) <= {1, 2})
        sorted_commands = np.round(np.linspace(200, 498, 100)).astype(int)
        _, permutation = SCENE.draws(7, 100)
        commands = [trial["command_row"] for trial in files["placed.json"]["trials"]]
        self.assertEqual(commands, sorted_commands[permutation.numpy()].tolist())
        self.assertEqual([trial["permutation"] for trial in record["trials"]], permutation.tolist())
        # Every command row has 1,000 upright rows of walk ahead, before walk's last row (1499).
        self.assertTrue(all(200 <= row and row + 1000 < 1499 for row in commands))
        trial = record["trials"][0]
        self.assertEqual((trial["body_clip"], trial["body_row_in_clip"]), ("getup", 0))
        self.assertEqual((trial["command_clip"], trial["command_row_in_clip"]), ("walk", trial["command_row"] - 200))

    def test_the_draws_at_seed_7_are_the_written_ones_and_the_scene_is_determined_by_its_seed(self):
        uniforms, permutation = SCENE.draws(7, 100)
        self.assertEqual(uniforms["position"][0].tolist(), POSITION_0)
        self.assertEqual(uniforms["rotation"][0].tolist(), ROTATION_0)
        self.assertEqual(uniforms["velocity"][50].tolist(), VELOCITY_50)
        self.assertEqual(uniforms["joints"][99, -2:].tolist(), JOINTS_99_LAST_TWO)
        self.assertEqual(permutation.tolist(), PERMUTATION)
        first, second = (SCENE.scene(library(), settings(), placed()) for _ in range(2))
        self.assertEqual(first, second)
        other, _ = SCENE.scene(library(), settings(), placed(uniforms_seed=8))
        self.assertNotEqual(other["placed.json"], first[0]["placed.json"])
        self.assertEqual([trial["body_row"] for trial in other["placed.json"]["trials"]],
                         [trial["body_row"] for trial in first[0]["placed.json"]["trials"]])

    def test_the_yaw_at_placement_turns_the_body_row_by_the_reset_rotation(self):
        qpos = torch.zeros((2, 36))
        qpos[:, 3] = 1
        qpos[1, 3], qpos[1, 6] = np.cos(0.15), np.sin(0.15)  # the command row turned by 0.3 rad of yaw
        yaw = SCENE.placed_yaw(qpos, torch.tensor([0]), torch.tensor([1]), torch.tensor([[0.0, 0.0, 1.0]]))
        self.assertAlmostEqual(float(yaw[0]), 0.2 - 0.3, places=6)

    def test_the_reset_rotation_is_a_rotation_vector_premultiplied_on_a_pitched_body_row(self):
        # The body row pitched by 1 rad; the unit uniforms (0.8, 0, -0.6) give the rotation vector (0.08, 0, -0.12).
        qpos = torch.zeros((2, 36), dtype=torch.float64)
        qpos[:, 3] = 1
        qpos[0, 3], qpos[0, 5] = math.cos(0.5), math.sin(0.5)
        yaw = float(SCENE.placed_yaw(qpos, torch.tensor([0]), torch.tensor([1]),
                                     torch.tensor([[0.8, 0.0, -0.6]], dtype=torch.float64))[0])
        vector = np.array([0.08, 0.0, -0.12])
        angle = np.linalg.norm(vector)
        k = np.array([[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]])/angle
        delta = np.eye(3) + math.sin(angle)*k + (1 - math.cos(angle))*k@k  # Rodrigues' formula
        pitch = np.array([[math.cos(1.0), 0.0, math.sin(1.0)], [0.0, 1.0, 0.0], [-math.sin(1.0), 0.0, math.cos(1.0)]])
        before, after = delta@pitch, pitch@delta
        self.assertAlmostEqual(yaw, math.atan2(before[1, 0], before[0, 0]), places=6)
        self.assertGreater(abs(yaw - math.atan2(after[1, 0], after[0, 0])), 0.1)

    def test_the_recovery_setting_is_read_from_the_recovery_condition_s_arguments(self):
        read = SCENE.recovery_settings(ROOT/"configs/recovery.args")
        self.assertEqual(len(read.hidden_clips), 5)
        self.assertEqual((read.command_upright_rows, read.upright_run_rows, read.longest_span_rows), (500, 50, 375))
        with self.assertRaises(ValueError):
            SCENE.recovery_settings(ROOT/"configs/absolute.args")


if __name__ == "__main__":
    unittest.main()
