"""Videos of rollout records (render_rollouts, ROLLOUTS.md) from a synthetic record and a synthetic processed-clip
cache written by the test; CPU only.

Library (clips in this order): fallAndGetUp2_subject2 (30 rows, 0-29) and walk1_subject1 (40 rows, 30-69); row r holds the root at
(0.01 r, 0, 0.78), the identity orientation and every joint at 0.001 r. Record: 7 controls, 3 trials.
  trial 0: command row 30, never ends;
  trial 1: command row 31, body placed at row 5 of the get-up clip, ended by control 3 with a finite state after it;
  trial 2: command row 60, ended by control 0 with a non-finite state after it (only the placed start is recorded).
The state after control c of trial t has its root at (0.1 c, t, 0.7) and every joint at 0.01 c + 0.1 t; the placed
start has its root at (-0.1, t, 0.7) and every joint at 0.1 t - 0.005. After a trial's end the fields hold NaN, -1 or
false, as the rollout tool writes them. The command row after control c is min(command + c + 1, 69), so a record
with command rows near 69 (CLIP_END_COMMAND) holds commands that reach their clip's last row; a trial whose command
row differs from COMMAND has its body row at its command row.

Where the Menagerie scene is present, the record's start_body_pos_w is computed with the model from start_qpos, and
rendering runs in a subprocess with MUJOCO_GL=egl: it renders when the backend opens and otherwise must give the tool's
refusal naming MUJOCO_GL; a printed line names the path taken."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zlib

import numpy as np

from nablamimic import render_rollouts as render

CONTROLS, TRIALS = 7, 3
CLIPS = (("fallAndGetUp2_subject2", 30), ("walk1_subject1", 40))
GETUP, WALK = (name for name, _ in CLIPS)
COMMAND, BODY = np.array([30, 31, 60], np.int32), np.array([30, 5, 60], np.int32)
ENDS = {1: (3, True), 2: (0, False)}  # trial: (ending control, finite state after it)
CLIP_END_COMMAND = np.array([64, 65, 60], np.int32)  # trial 0 reaches row 69 after control 4, trial 1 after control 3
SCENE_CONTENT = {"trials": [{"command_row": 30}, {"command_row": 31, "body_row": 5}, {"command_row": 60}]}


def reference_qpos(rows):
    qpos = np.zeros((len(rows), 36))
    qpos[:, 0], qpos[:, 2], qpos[:, 3] = 0.01*rows, 0.78, 1.0
    qpos[:, 7:] = 0.001*rows[:, None]
    return qpos


def write_cache(folder):
    """The processed-clip cache of the library (motion.CACHE_KEYS) under ``folder``."""
    from nablamimic import motion
    from nablamimic.env import BM_BODIES
    start = 0
    for name, rows in CLIPS:
        global_rows = np.arange(start, start + rows)
        nb = len(BM_BODIES)
        arrays = dict(qpos=reference_qpos(global_rows), qvel=np.zeros((rows, 35)), body_pos=np.zeros((rows, nb, 3)),
                      body_quat=np.tile([1.0, 0, 0, 0], (rows, nb, 1)), body_lin_vel=np.zeros((rows, nb, 3)),
                      body_ang_vel=np.zeros((rows, nb, 3)))
        assert set(arrays) == set(motion.CACHE_KEYS)
        np.savez(Path(folder)/motion.cache_file_name(name, bodies=BM_BODIES), **arrays)
        start += rows


def library_inputs(cache):
    return {"LAFAN1_G1_DIR": os.environ.get("LAFAN1_G1_DIR"), "MOTION_CACHE_DIR": str(cache),
            "MENAGERIE_DIR": os.environ.get("MENAGERIE_DIR")}


def record_arrays(command=COMMAND, ends=None):
    """The fields of rollouts.npz (ROLLOUTS.md, "Record") for the trials of the module docstring, with other command
    rows or trial ends (ENDS by default) where given."""
    ends = ENDS if ends is None else ends
    c = np.arange(CONTROLS, dtype=np.float32)[:, None]
    t = np.arange(TRIALS, dtype=np.float32)[None]
    qpos = np.zeros((CONTROLS, TRIALS, 36), np.float32)
    qpos[..., 0], qpos[..., 1], qpos[..., 2], qpos[..., 3] = 0.1*c, t, 0.7, 1.0
    qpos[..., 7:] = (0.01*c + 0.1*t)[..., None]
    start = np.zeros((TRIALS, 36), np.float32)
    start[:, 0], start[:, 1], start[:, 2], start[:, 3] = -0.1, np.arange(TRIALS), 0.7, 1.0
    start[:, 7:] = (0.1*np.arange(TRIALS) - 0.005)[:, None]
    frame_after = np.minimum(command[None] + np.arange(1, CONTROLS + 1)[:, None], 69).astype(np.int32)
    clip_last_row = np.full((CONTROLS, TRIALS), 69, np.int32)
    available = np.ones((CONTROLS, TRIALS), bool)
    done = np.zeros((CONTROLS, TRIALS), bool)
    for trial, (end, finite) in ends.items():
        done[end, trial] = True
        if not finite:
            qpos[end, trial] = np.nan
            available[end, trial] = False
        qpos[end + 1:, trial] = np.nan
        available[end + 1:, trial] = False
        frame_after[end + 1:, trial] = -1
        clip_last_row[end + 1:, trial] = -1
    return dict(qpos=qpos, qvel=np.zeros((CONTROLS, TRIALS, 35), np.float32), frame_after=frame_after,
                frame_before=np.concatenate((command[None], frame_after[:-1])), clip_last_row=clip_last_row,
                available=available, done=done, command_row=command,
                body_row=np.where(command == COMMAND, BODY, command).astype(np.int32),
                push_control=np.full(TRIALS, -1, np.int32), push_velocity_xy=np.zeros((TRIALS, 2), np.float32),
                start_qpos=start, start_qvel=np.zeros((TRIALS, 35), np.float32),
                start_body_pos_w=np.zeros((TRIALS, 14, 3), np.float32),
                start_body_quat_w=np.tile(np.float32([1, 0, 0, 0]), (TRIALS, 14, 1)))


def consistent_arrays():
    """record_arrays with start_body_pos_w placed by the model at start_qpos (needs the Menagerie scene)."""
    import mujoco
    from nablamimic.env import BM_BODIES
    model = render.build_model()
    data = mujoco.MjData(model)
    ids = [model.body(name).id for name in BM_BODIES]
    arrays = record_arrays()
    for trial in range(TRIALS):
        data.qpos[:] = arrays["start_qpos"][trial]
        mujoco.mj_kinematics(model, data)
        arrays["start_body_pos_w"][trial] = data.xpos[ids]
    return arrays


def scene_present():
    from nablamimic.motion import SCENE
    return SCENE.exists()


def write_record(folder, cache, arrays=None, **description):
    folder = Path(folder)
    folder.mkdir(parents=True)
    content = dict(controls=CONTROLS, scene_content=SCENE_CONTENT, termination="nonfinite",
                   settings=dict(clips=[name for name, _ in CLIPS], inputs=library_inputs(cache)))
    content.update(description)
    (folder/"description.json").write_text(json.dumps(content))
    with (folder/"rollouts.npz").open("xb") as handle:
        np.savez(handle, **(record_arrays() if arrays is None else arrays))
    return folder


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = self.tmp/"cache"
        self.cache.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp)


class ReplayOfARecord(TempDir):
    def setUp(self):
        super().setUp()
        self.record = render.Record(write_record(self.tmp/"a", self.cache))
        self.arrays = record_arrays()

    def test_each_step_shows_the_state_after_its_control_and_the_command_row_after_it(self):
        steps = render.replay_steps(CONTROLS, 1)
        states = render.panel_states(self.record, 0, steps)
        self.assertEqual(len(states), CONTROLS + 1)
        np.testing.assert_array_equal(states[0]["qpos"], self.arrays["start_qpos"][0])
        self.assertEqual((states[0]["row"], states[0]["control"], states[0]["ended"]), (30, -1, None))
        for step, state in zip(steps[1:], states[1:]):
            control = step - 1
            expected = self.arrays["qpos"][control, 0]
            np.testing.assert_array_equal(state["qpos"][:3], expected[:3])   # root position
            np.testing.assert_array_equal(state["qpos"][3:7], expected[3:7])  # root orientation
            np.testing.assert_array_equal(state["qpos"][7:], expected[7:])   # joints
            self.assertEqual(state["row"], self.arrays["frame_after"][control, 0])
            self.assertEqual((state["control"], state["ended"]), (control, None))

    def test_an_ended_trial_is_held_at_its_last_recorded_state(self):
        states = render.panel_states(self.record, 1, render.replay_steps(CONTROLS, 1))
        for step, state in enumerate(states):
            if step <= 3:  # the start and controls 0 to 2
                self.assertIsNone(state["ended"])
                self.assertEqual(state["shown_control"], step - 1)
            else:  # control 3 ends the trial and its state is recorded
                self.assertEqual((state["ended"], state["shown_control"], state["row"]), (3, 3, 35))
                np.testing.assert_array_equal(state["qpos"], self.arrays["qpos"][3, 1])
        self.assertEqual(render.trial_end(self.record, 1), (3, 3))

    def test_a_trial_ended_by_a_nonfinite_state_at_control_0_shows_its_placed_start(self):
        states = render.panel_states(self.record, 2, render.replay_steps(CONTROLS, 1))
        self.assertIsNone(states[0]["ended"])
        for state in states:
            np.testing.assert_array_equal(state["qpos"], self.arrays["start_qpos"][2])
            self.assertEqual((state["row"], state["shown_control"]), (60, -1))
            self.assertTrue(np.isfinite(state["qpos"]).all())
        self.assertTrue(all(state["ended"] == 0 for state in states[1:]))
        self.assertEqual(render.trial_end(self.record, 2), (0, -1))

    def test_a_nonfinite_state_before_the_end_is_refused(self):
        arrays = record_arrays()
        arrays["qpos"][2, 0, 9] = np.nan
        record = render.Record(write_record(self.tmp/"b", self.cache, arrays))
        with self.assertRaisesRegex(ValueError, "trial 0 holds a non-finite state"):
            render.trial_end(record, 0)

    def test_frame_count_for_a_stride(self):
        for stride, steps in ((1, list(range(8))), (2, [0, 2, 4, 6]), (3, [0, 3, 6]), (7, [0, 7]), (10, [0])):
            self.assertEqual(render.replay_steps(CONTROLS, stride), steps)
            self.assertEqual(len(steps), CONTROLS//stride + 1)
        states = render.panel_states(self.record, 0, render.replay_steps(CONTROLS, 3))
        self.assertEqual([s["control"] for s in states], [-1, 2, 5])
        for stride in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "stride"):
                render.replay_steps(CONTROLS, stride)

    def test_choice_and_order_of_trials(self):
        self.assertEqual(render.select_trials(None, TRIALS), [0, 1, 2])
        self.assertEqual(render.select_trials([2, 0], TRIALS), [2, 0])
        for requested, message in (([3], "not trials"), ([-1], "not trials"), ([1, 1], "repeat"), ([], "no trial")):
            with self.assertRaisesRegex(ValueError, message):
                render.select_trials(requested, TRIALS)

    def test_a_record_that_lacks_a_field_is_refused(self):
        for field in render.RECORD_FIELDS:
            arrays = record_arrays()
            del arrays[field]
            with self.assertRaisesRegex(ValueError, f"lacks the field\\(s\\) {field}"):
                render.Record(write_record(self.tmp/f"no_{field}", self.cache, arrays))
        folder = write_record(self.tmp/"no_scene", self.cache)
        description = json.loads((folder/"description.json").read_text())
        del description["scene_content"], description["settings"]["inputs"]
        (folder/"description.json").write_text(json.dumps(description))
        with self.assertRaisesRegex(ValueError, "lacks scene_content, settings.inputs"):
            render.Record(folder)
        arrays = record_arrays()
        arrays["start_qpos"] = arrays["start_qpos"][:2]
        with self.assertRaisesRegex(ValueError, "start_qpos"):
            render.Record(write_record(self.tmp/"short", self.cache, arrays))
        with self.assertRaisesRegex(ValueError, "holds 7 controls, description.json says 8"):
            render.Record(write_record(self.tmp/"controls", self.cache, controls=8))

    def test_two_records_must_hold_the_same_trials(self):
        render.same_trials(self.record, render.Record(write_record(self.tmp/"same", self.cache)))
        arrays = record_arrays()
        arrays["body_row"] = np.array([30, 6, 60], np.int32)
        cases = dict(body_row=write_record(self.tmp/"body", self.cache, arrays),
                     scene=write_record(self.tmp/"scene", self.cache, scene_content={"trials": []}))
        per_trial = ("command_row", "body_row", "push_control", "push_velocity_xy", "start_qpos", "start_qvel",
                     "start_body_pos_w", "start_body_quat_w")
        arrays = {key: value[:2] if key in per_trial else value[:, :2] for key, value in record_arrays().items()}
        cases["trial count"] = write_record(self.tmp/"two", self.cache, arrays)
        arrays = {key: value[:6] if value.shape[0] == CONTROLS else value for key, value in record_arrays().items()}
        cases["controls"] = write_record(self.tmp/"six", self.cache, arrays, controls=6)
        cases["clip names"] = write_record(self.tmp/"clips", self.cache,
                                           settings=dict(clips=[WALK, GETUP], inputs=library_inputs(self.cache)))
        arrays = record_arrays()
        arrays["start_qpos"][1, 9] += 1e-3
        cases["start_qpos"] = write_record(self.tmp/"start", self.cache, arrays)
        for what, folder in cases.items():
            with self.assertRaisesRegex(ValueError, f"differ in .*{what if what != 'scene' else 'scene content'}"):
                render.same_trials(self.record, render.Record(folder))
        arrays = record_arrays()
        arrays["start_qpos"] += 1e-6  # within the tolerance of one start
        render.same_trials(self.record, render.Record(write_record(self.tmp/"rounded", self.cache, arrays)))


class ModelOfTheRecord(TempDir):
    def test_the_model_places_the_start_bodies_where_the_record_holds_them(self):
        if not scene_present():
            self.skipTest("no Menagerie scene")
        from nablamimic.env import BM_BODIES
        model = render.build_model()
        arrays = consistent_arrays()
        root = BM_BODIES.index("pelvis")
        np.testing.assert_allclose(arrays["start_body_pos_w"][:, root], arrays["start_qpos"][:, :3], atol=1e-6)
        render.check_model(model, render.Record(write_record(self.tmp/"a", self.cache, arrays)), [0, 1, 2])
        moved = {key: value.copy() for key, value in arrays.items()}
        moved["start_body_pos_w"][1, 4, 2] += 0.01
        record = render.Record(write_record(self.tmp/"b", self.cache, moved))
        render.check_model(model, record, [0, 2])  # only the trials rendered are compared
        with self.assertRaisesRegex(ValueError, "up to 0.01 m from the record's start_body_pos_w"):
            render.check_model(model, record, [2, 1])
        bent = {key: value.copy() for key, value in arrays.items()}
        bent["start_qpos"][0, 10] += 0.3  # a joint the record does not place there
        with self.assertRaisesRegex(ValueError, "not the model that wrote the record"):
            render.check_model(model, render.Record(write_record(self.tmp/"c", self.cache, bent)), [0])


class ReferenceRowsFromTheCache(TempDir):
    def setUp(self):
        super().setUp()
        write_cache(self.cache)
        from nablamimic import motion
        self.enterContext(mock.patch.object(motion, "CACHE_DIR", self.cache))
        self.record = render.Record(write_record(self.tmp/"a", self.cache))

    def test_rows_clips_and_reference_poses(self):
        library = render.ReferenceRows([GETUP, WALK])
        self.assertEqual(library.qpos.shape, (70, 36))
        self.assertEqual([library.clip(r) for r in (0, 5, 29, 30, 69)], [GETUP]*3 + [WALK]*2)
        np.testing.assert_allclose(library.qpos[[5, 31, 69]], reference_qpos(np.array([5, 31, 69])))
        library.check(self.record)
        lines = render.caption("A", 1, self.record, library, render.panel_states(self.record, 1, [0, 5])[1], 5, 50.0)
        self.assertEqual(lines, ["A  trial 1", f"command clip {WALK}", f"body placed at {GETUP} row 5",
                                 "control 4  0.10 s", "ended at control 3"])
        start = render.caption("A", 0, self.record, library, render.panel_states(self.record, 0, [0])[0], 0, 50.0)
        self.assertEqual(start, ["A  trial 0", f"command clip {WALK}", "start  0.00 s"])

    def test_a_library_that_differs_from_the_record_is_refused(self):
        arrays = record_arrays()
        arrays["clip_last_row"][0, 0] = 29
        record = render.Record(write_record(self.tmp/"b", self.cache, arrays))
        with self.assertRaisesRegex(ValueError, "clip ends differ"):
            render.ReferenceRows([GETUP, WALK]).check(record)
        with self.assertRaisesRegex(ValueError, "outside the motion library"):
            render.ReferenceRows([GETUP]).check(self.record)
        before = sorted(os.listdir(self.cache))
        with self.assertRaisesRegex(ValueError, "no processed clip"):
            render.ReferenceRows([GETUP, "run1_subject2"])
        self.assertEqual(sorted(os.listdir(self.cache)), before)  # nothing is written to the cache

    def test_library_inputs_must_match_the_record(self):
        with mock.patch.dict(os.environ, {"MOTION_CACHE_DIR": str(self.cache)}):
            render.check_library_inputs(self.record)
        with mock.patch.dict(os.environ, {"MOTION_CACHE_DIR": str(self.tmp)}):
            with self.assertRaisesRegex(ValueError, "MOTION_CACHE_DIR"):
                render.check_library_inputs(self.record)


class VideoLength(TempDir):
    def setUp(self):
        super().setUp()
        self.record = render.Record(write_record(self.tmp/"a", self.cache))
        self.clip_arrays = record_arrays(CLIP_END_COMMAND)
        self.clip_record = render.Record(write_record(self.tmp/"clip", self.cache, self.clip_arrays))

    def test_the_clip_end_is_the_first_control_whose_command_row_is_the_clip_last_row(self):
        rows = self.clip_arrays["frame_after"]
        self.assertEqual(render.clip_end(self.clip_record, 0), 4)
        self.assertEqual(rows[3, 0], 68)
        np.testing.assert_array_equal(rows[4:, 0], 69)  # the command row no longer advances from the clip end on
        self.assertEqual(render.clip_end(self.clip_record, 1), 3)
        self.assertIsNone(render.clip_end(self.clip_record, 2))
        for trial in range(TRIALS):  # commands at rows 31 to 37: the clip end is not in the record
            self.assertIsNone(render.clip_end(self.record, trial))
        # trial 2's fields after its end are -1 in frame_after and clip_last_row alike, which is no clip end
        self.assertTrue((self.record.arrays["frame_after"][1:, 2] == self.record.arrays["clip_last_row"][1:, 2]).all())

    def test_end_points_of_a_trial(self):
        self.assertEqual([render.end_point(self.record, t) for t in range(TRIALS)],
                         [(None, "all controls"), (3, "trial ended"), (0, "trial ended")])
        # trial 0: the clip end at control 4; trial 1: its clip end and its trial end are both control 3
        self.assertEqual([render.end_point(self.clip_record, t) for t in range(TRIALS)],
                         [(4, "clip ended"), (3, "trial ended"), (0, "trial ended")])
        arrays = record_arrays(CLIP_END_COMMAND, ends={0: (5, True)})
        self.assertEqual(render.end_point(render.Record(write_record(self.tmp/"late", self.cache, arrays)), 0),
                         (4, "clip ended"))  # the earlier of the clip end and the trial end
        # trial 1 ends at control 3, before its command would reach row 69 at control 4: no clip end is recorded
        arrays = record_arrays(np.array([30, 64, 60], np.int32))
        record = render.Record(write_record(self.tmp/"early", self.cache, arrays))
        self.assertEqual(arrays["frame_after"][:4, 1].tolist(), [65, 66, 67, 68])
        self.assertEqual((render.clip_end(record, 1), render.end_point(record, 1)), (None, (3, "trial ended")))

    def test_the_stopping_control_and_the_frames(self):
        record, clip = self.record, self.clip_record
        self.assertEqual(render.video_stop([record], 0, 1), (6, "all controls", [(None, "all controls")]))
        self.assertEqual(render.video_stop([record], 1, 1)[:2], (4, "trial ended"))
        self.assertEqual(render.video_stop([record], 2, 1)[:2], (1, "trial ended"))
        self.assertEqual(render.video_stop([clip], 0, 1)[:2], (5, "clip ended"))
        self.assertEqual(render.video_stop([clip], 0, 0)[:2], (4, "clip ended"))
        self.assertEqual(render.video_stop([clip], 0, 100)[:2], (6, "clip ended"))  # never beyond the record
        self.assertEqual(render.video_stop([clip], 0, None)[:2], (6, "all controls"))
        self.assertEqual(render.video_stop([record], 1, None)[:2], (6, "all controls"))
        stop = render.video_stop([record], 1, 1)[0]  # the state after control 4 is step 5
        for stride, steps in ((1, [0, 1, 2, 3, 4, 5]), (2, [0, 2, 4]), (3, [0, 3]), (6, [0])):
            self.assertEqual(render.replay_steps(stop + 1, stride), steps)
        self.assertEqual((render.tail_controls(2.0, 50.0), render.tail_controls(0.1, 50.0)), (100, 5))
        self.assertEqual((render.tail_controls(0.0, 50.0), render.tail_controls(None, 50.0)), (0, None))

    def test_side_by_side_the_video_ends_after_the_later_end_point(self):
        early = render.Record(write_record(self.tmp/"b", self.cache, record_arrays(ends={0: (1, True), 1: (1, True),
                                                                                         2: (0, False)})))
        render.same_trials(self.record, early)
        self.assertEqual(render.video_stop([self.record, early], 1, 1),
                         (4, "trial ended", [(3, "trial ended"), (1, "trial ended")]))
        self.assertEqual(render.video_stop([early, self.record], 1, 1)[:2], (4, "trial ended"))
        # trial 0 of the first record has no end point: every control
        self.assertEqual(render.video_stop([self.record, early], 0, 1),
                         (6, "all controls", [(None, "all controls"), (1, "trial ended")]))
        arrays = record_arrays(CLIP_END_COMMAND, ends={0: (1, True)})
        ended = render.Record(write_record(self.tmp/"c", self.cache, arrays))
        render.same_trials(self.clip_record, ended)
        self.assertEqual(render.video_stop([ended, self.clip_record], 0, 1),
                         (5, "clip ended", [(1, "trial ended"), (4, "clip ended")]))

    def test_tail_and_ghost_options(self):
        parser = render.build_parser()
        base = ["--record", "a", "--out", "b"]
        args = parser.parse_args(base)
        self.assertEqual((args.tail, args.ghost), (2.0, "world"))
        for text, tail in (("all", None), ("-1", None), ("-0.5", None), ("0", 0.0), ("1.5", 1.5)):
            self.assertEqual(parser.parse_args(base + ["--tail", text]).tail, tail)
        self.assertEqual(parser.parse_args(base + ["--ghost", "robot"]).ghost, "robot")
        for wrong in (["--tail", "some"], ["--tail", "nan"], ["--ghost", "elsewhere"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(base + wrong)


class GhostPlacement(TempDir):
    def setUp(self):
        super().setUp()
        write_cache(self.cache)
        from nablamimic import motion
        self.enterContext(mock.patch.object(motion, "CACHE_DIR", self.cache))
        self.library = render.ReferenceRows([GETUP, WALK])
        self.record = render.Record(write_record(self.tmp/"a", self.cache))

    def test_the_ghost_root_at_the_robot_or_at_the_command(self):
        library_before = self.library.qpos.copy()
        states = render.panel_states(self.record, 1, render.replay_steps(CONTROLS, 1))  # held after control 3
        for state in states:
            reference = self.library.qpos[state["row"]]
            world = render.ghost_qpos(reference, state["qpos"], "world")
            robot = render.ghost_qpos(reference, state["qpos"], "robot")
            np.testing.assert_array_equal(world, reference)
            np.testing.assert_array_equal(robot[:2], state["qpos"][:2])
            np.testing.assert_array_equal(robot[2:], reference[2:])  # height, orientation and joints
            self.assertNotEqual(robot[1], world[1])  # trial 1's root is at y = 1, the library's at y = 0
        np.testing.assert_array_equal(self.library.qpos, library_before)  # the library is not written to
        with self.assertRaisesRegex(ValueError, "placement"):
            render.ghost_qpos(reference, state["qpos"], "body")
        self.assertEqual(render.bottom_line(1, "world"), "")
        self.assertEqual(render.bottom_line(2, "world"), "two runs from one start")
        self.assertEqual(render.bottom_line(2, "robot"),
                         "translucent command pose drawn at the robot's position\ntwo runs from one start")

    def test_a_trial_rendered_with_a_stand_in_renderer_and_encoder(self):
        calls, written = [], []

        class Renderer:
            viewport = types.SimpleNamespace(width=8, height=6)

            def render(self, qpos, ghost, top_lines, bottom_line=""):
                calls.append((qpos.copy(), ghost.copy(), bottom_line))
                return np.zeros((6, 8, 3), np.uint8)

        class Writer:
            def __init__(self, ffmpeg, path, width, height, fps):
                self.shape = (height, width, 3)

            def write(self, rgb):
                assert rgb.shape == self.shape
                written.append(rgb)

            def close(self):
                pass

        early = render.Record(write_record(self.tmp/"b", self.cache, record_arrays(ends={1: (1, True), 2: (0, False)})))
        self.enterContext(mock.patch.object(render, "VideoWriter", Writer))
        cases = ((1, [self.record], "robot", 0.02, 1, 6, 4, "trial ended"),
                 (1, [self.record, early], "world", 0.02, 2, 3, 4, "trial ended"),
                 (0, [self.record], "robot", 0.02, 1, 8, 6, "all controls"),
                 (1, [self.record], "world", None, 3, 3, 6, "all controls"))
        for k, (trial, records, ghost, tail_s, stride, frames, stop, reason) in enumerate(cases):
            calls.clear()
            written.clear()
            out = self.tmp/f"out{k}"
            out.mkdir()
            options = dict(stride=stride, ghost=ghost, tail_s=tail_s)
            summary = render.render_trial(records, [self.library]*len(records), ["A", "B"][:len(records)], trial,
                                          Renderer(), "ffmpeg", out, 50.0, 50.0, options)
            self.assertEqual((summary["frames"], len(written), len(calls)), (frames, frames, frames*len(records)))
            self.assertEqual((summary["stop_control"], summary["stop_reason"]), (stop, reason))
            saved = json.loads((out/f"trial_{trial:04d}.json").read_text())
            self.assertEqual((saved["frames"], saved["stop_control"], saved["stop_reason"]), (frames, stop, reason))
            self.assertEqual(saved["options"]["ghost"], ghost)
            self.assertEqual(saved["last_step"], render.replay_steps(stop + 1, stride)[-1])
            width, height, _ = read_png((out/f"trial_{trial:04d}.png").read_bytes())
            self.assertEqual((width, height), (4*4*len(records), 3*(-(-min(frames, 8)//4))))
            for qpos, ghost_pose, bottom in calls:
                if ghost == "robot":
                    np.testing.assert_array_equal(ghost_pose[:2], qpos[:2])
                    self.assertIn("drawn at the robot's position", bottom)
                else:
                    self.assertNotIn("robot's position", bottom)
            if ghost == "world":  # the ghost at the command row's own pose: x = 0.01 row, y = 0
                steps = render.replay_steps(stop + 1, stride)
                rows = [state["row"] for state in render.panel_states(records[-1], trial, steps)]
                np.testing.assert_allclose([c[1][:2] for c in calls[len(records) - 1::len(records)]],
                                           [[0.01*row, 0.0] for row in rows])


def read_png(data):
    """(width, height, pixels [H, W, 3]) of an 8-bit RGB PNG without filters, as png_bytes writes it."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, chunks = 8, {}
    while position < len(data):
        length = int.from_bytes(data[position:position + 4], "big")
        kind = data[position + 4:position + 8]
        chunks[kind] = chunks.get(kind, b"") + data[position + 8:position + 8 + length]
        position += 12 + length
    width, height = int.from_bytes(chunks[b"IHDR"][:4], "big"), int.from_bytes(chunks[b"IHDR"][4:8], "big")
    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(height, 1 + 3*width)
    assert (raw[:, 0] == 0).all()
    return width, height, raw[:, 1:].reshape(height, width, 3)


class Images(unittest.TestCase):
    def test_png_round_trip(self):
        rgb = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
        width, height, pixels = read_png(render.png_bytes(rgb))
        self.assertEqual((width, height), (7, 5))
        np.testing.assert_array_equal(pixels, rgb)

    def test_contact_sheet(self):
        self.assertEqual(render.sheet_indices(20), [0, 3, 5, 8, 11, 14, 16, 19])
        self.assertEqual(render.sheet_indices(3), [0, 1, 2])
        frames = [np.full((6, 10, 3), 10*k, np.uint8) for k in range(8)]
        sheet = render.contact_sheet(frames)
        self.assertEqual(sheet.shape, (6, 20, 3))
        self.assertEqual(sheet[4, 17, 0], 70)  # second row, last column: the eighth frame at half size
        self.assertEqual(render.contact_sheet(frames[:3]).shape, (3, 20, 3))


def ffprobe_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames,codec_name,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


class Encoding(TempDir):
    def test_h264_video_of_the_frames(self):
        try:
            ffmpeg = render.find_encoder()
        except RuntimeError as error:
            print(f"\n[render_rollouts] no H.264 encoder available: {error}")
            self.assertRegex(str(error), "ffmpeg|libx264")
            return
        writer = render.VideoWriter(ffmpeg, self.tmp/"v.mp4", 32, 16, 50.0)
        for k in range(5):
            writer.write(np.full((16, 32, 3), 40*k, np.uint8))
        with self.assertRaisesRegex(ValueError, "shape"):
            writer.write(np.zeros((16, 30, 3), np.uint8))
        writer.close()
        if shutil.which("ffprobe"):
            stream = ffprobe_frames(self.tmp/"v.mp4")
            self.assertEqual((stream["codec_name"], int(stream["nb_read_frames"]), stream["width"]), ("h264", 5, 32))
        print(f"\n[render_rollouts] encoded with {ffmpeg}; ffprobe {'present' if shutil.which('ffprobe') else 'absent'}")


RENDER_CHECK = r"""
import numpy as np
from nablamimic import render_rollouts as render
model = render.build_model()
renderer = render.PanelRenderer(model, 64, 48, 3.5, 135.0, -15.0)
q = np.zeros(36); q[2], q[3] = 0.78, 1.0
far, near = q.copy(), q.copy(); far[0], near[0] = 0.8, 0.1
plain = renderer.render(q, q, [])
text = renderer.render(q, q, ["trial 0"])
ghost = renderer.render(q, far, [])
translucent = renderer.render(q, near, [])
render.GHOST_RGBA = render.GHOST_RGBA[:3] + (1.0,)
opaque = renderer.render(q, near, [])
renderer.close()
changed = lambda a, b: int((a != b).any(-1).sum())
print("changed pixels", plain.shape[:2], changed(text, plain), changed(ghost, plain), changed(translucent, opaque))
"""


def tool(args, **env):
    environment = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    environment.update(env)
    return subprocess.run([sys.executable, "-m", "nablamimic.render_rollouts", *args], env=environment,
                          capture_output=True, text=True, timeout=900)


class CommandLine(TempDir):
    def setUp(self):
        super().setUp()
        write_cache(self.cache)
        self.arrays = consistent_arrays() if scene_present() else record_arrays()
        self.folder = write_record(self.tmp/"a", self.cache, self.arrays)
        self.env = {"MOTION_CACHE_DIR": str(self.cache)}

    def test_an_existing_output_folder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exists"):
            render.main(["--record", str(self.folder), "--out", str(self.tmp)])

    def test_odd_sizes_and_unmatched_labels_are_refused(self):
        out = str(self.tmp/"out")
        with self.assertRaisesRegex(ValueError, "even"):
            render.main(["--record", str(self.folder), "--out", out, "--width", "63"])
        with self.assertRaisesRegex(ValueError, "labels"):
            render.main(["--record", str(self.folder), "--out", out, "--labels", "A", "B"])

    def test_an_unusable_gl_backend_is_refused_naming_its_variable(self):
        out = self.tmp/"out"
        result = tool(["--record", str(self.folder), "--out", str(out)], MUJOCO_GL="no-such-backend", **self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MUJOCO_GL='no-such-backend'", result.stderr)
        self.assertFalse(out.exists())
        if not scene_present():
            return
        result = tool(["--record", str(self.folder), "--out", str(out)], MUJOCO_GL="glfw", **self.env)
        if result.returncode != 0:
            self.assertIn("MUJOCO_GL='glfw'", result.stderr)
            self.assertFalse(out.exists())
        print(f"\n[render_rollouts] MUJOCO_GL=glfw without a display: rc {result.returncode}")

    def test_videos_with_egl_or_its_refusal(self):
        if not scene_present():
            self.skipTest("no Menagerie scene")
        environment = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
        check = subprocess.run([sys.executable, "-c", RENDER_CHECK], env={**environment, "MUJOCO_GL": "egl"},
                               capture_output=True, text=True, timeout=600)
        if check.returncode != 0:
            last = (check.stderr.strip().splitlines() or ["no error text"])[-1]
            print(f"\n[render_rollouts] MUJOCO_GL=egl does not render in this environment: {last}")
            self.assertIn("MUJOCO_GL='egl'", check.stderr)
            out = self.tmp/"out"
            result = tool(["--record", str(self.folder), "--out", str(out)], MUJOCO_GL="egl", **self.env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MUJOCO_GL='egl'", result.stderr)
            self.assertFalse(out.exists())
            return
        text_pixels, ghost_pixels, alpha_pixels = check.stdout.split()[-3:]
        self.assertGreater(int(text_pixels), 0)   # the text line is drawn
        self.assertGreater(int(ghost_pixels), 0)  # the ghost is drawn where the command places it
        self.assertGreater(int(alpha_pixels), 0)  # an overlapping ghost is translucent: it differs from an opaque one
        second = write_record(self.tmp/"b", self.cache, self.arrays)
        out = self.tmp/"out"
        result = tool(["--record", str(self.folder), "--record", str(second), "--labels", "first", "second",
                       "--trials", "2", "1", "--stride", "2", "--width", "64", "--height", "48", "--out", str(out)],
                      MUJOCO_GL="egl", **self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split()[:2], ["trial", "2:"])  # the trials in the order given
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         sorted(f"trial_000{t}.{suffix}" for t in (1, 2) for suffix in ("json", "mp4", "png")))
        summary = json.loads((out/"trial_0001.json").read_text())
        self.assertEqual((summary["frames"], summary["stride"], summary["first_step"], summary["last_step"]),
                         (CONTROLS//2 + 1, 2, 0, 6))
        self.assertEqual((summary["trial"], summary["ended_at_control"], summary["labels"]), (1, [3, 3],
                                                                                            ["first", "second"]))
        self.assertEqual(summary["records"], [str(self.folder.resolve()), str(second.resolve())])
        self.assertEqual(summary["options"]["fps"], 50.0)
        self.assertEqual((summary["stop_control"], summary["stop_reason"], summary["options"]["ghost"],
                          summary["options"]["tail_s"]), (CONTROLS - 1, "trial ended", "world", 2.0))
        width, height, _ = read_png((out/"trial_0001.png").read_bytes())
        self.assertEqual((width, height), (4*64, 24))  # 4 frames of two 64 x 48 panels at half size, one row
        if shutil.which("ffprobe"):
            stream = ffprobe_frames(out/"trial_0001.mp4")
            self.assertEqual((stream["codec_name"], int(stream["nb_read_frames"]), stream["width"], stream["height"]),
                             ("h264", CONTROLS//2 + 1, 128, 48))
        short = self.tmp/"short"
        result = tool(["--record", str(self.folder), "--trials", "1", "--tail", "0.02", "--ghost", "robot", "--width",
                       "64", "--height", "48", "--out", str(short)], MUJOCO_GL="egl", **self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((short/"trial_0001.json").read_text())
        self.assertEqual((summary["frames"], summary["stop_control"], summary["stop_reason"],
                          summary["options"]["ghost"]), (6, 4, "trial ended", "robot"))  # the start, controls 0 to 4
        if shutil.which("ffprobe"):
            self.assertEqual(int(ffprobe_frames(short/"trial_0001.mp4")["nb_read_frames"]), 6)
        print(f"\n[render_rollouts] MUJOCO_GL=egl rendered: {check.stdout.strip()}")


if __name__ == "__main__":
    unittest.main()
