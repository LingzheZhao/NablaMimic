"""The readings of the recovery pair on hand-built rollouts and a small synthetic library; no engine, model or GPU.
The rule tests call studies/recovery/readings_core.py on its arrays; EntryTest and SaveTest call
studies/recovery/readings.py, which builds those arrays from nablamimic.engine_rollouts records.

Library (clips in this order): walk (1,200 upright rows, the pelvis at 0.78 m moving 0.01 m per row along x, every
body upright; with ``low``, a range of rows lowered to a pelvis at 0.3 m) and getup (100 rows). The tracked bodies
sit at fixed offsets from the pelvis, the torso (body 7) among them. A trial's command starts at its command row and
advances one row per control. Its body takes, per control, one of these postures against the command row it reached:
"exact" (the command's bodies), "lying" (pelvis at 0.2 m, pitched 90 degrees), "offset" (the command's bodies with
every body but the pelvis 0.2 m further along x), "done" ("exact" with the trial ended there) or "nan" (a non-finite
state, which ends the trial); or ("height", h) (the command's bodies lowered so that the pelvis is at h),
("tilt", degrees) (the command's bodies pitched by that angle), ("offset", d) (every body but the pelvis d further
along x, so that the body error is 13/14 d), ("turn", angle) (the command's bodies and orientations turned about z
around the pelvis and moved 0.3 m and -0.4 m in the plane) or ("turn_positions", angle) (the same with the positions
alone turned)."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"studies/recovery"))  # the study's programs import the modules beside them


def load(name):
    spec = importlib.util.spec_from_file_location(f"recovery_{name}", ROOT/f"studies/recovery/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load("readings_core")
DECLARATION = json.loads((ROOT/"studies/recovery/pair.json").read_text())
SETTINGS, DT = DECLARATION["fall_recovery"], DECLARATION["control_dt_s"]
# The pelvis and the torso (the anchor) among the 14 tracked bodies of the record, as readings.py passes them.
PELVIS, TORSO = 0, 7
NAMES, STARTS = ["walk", "getup"], [0, 1200]  # the clips of library() and their first rows
OFFSETS = np.stack([0.03*np.arange(14), 0.02*(np.arange(14) % 3), 0.05*(np.arange(14) % 5)], -1)
LYING = np.array([math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0])
CONTROLS = 1000


def library(low=range(0)):
    rows = 1300
    pelvis = np.stack([0.01*np.arange(rows), np.zeros(rows), np.full(rows, 0.78)], -1)
    pelvis[list(low), 2] = 0.3
    body_quat = np.zeros((rows, 14, 4), np.float32)
    body_quat[..., 0] = 1
    starts = np.array([0, 1200])
    last = np.concatenate([np.full(1200, 1199), np.full(100, 1299)])
    return SimpleNamespace(body_pos=(pelvis[:, None] + OFFSETS[None]).astype(np.float32), body_quat=body_quat,
                           starts=starts, last=last, names=["walk", "getup"])


def posture(lib, row, kind):
    position, quaternion = lib.body_pos[row].astype(np.float64), lib.body_quat[row].astype(np.float64)
    if isinstance(kind, tuple):
        name, value = kind
        if name == "height":
            return position - [0.0, 0.0, position[0, 2] - value], quaternion
        if name == "tilt":
            angle = math.radians(value)
            return position, np.repeat([[math.cos(angle/2), 0.0, math.sin(angle/2), 0.0]], 14, 0)
        if name in ("turn", "turn_positions"):
            turned_position, turned_quaternion = turned(lib, row, value, (0.3, -0.4), about=0)
            return turned_position, turned_quaternion if name == "turn" else quaternion
        return position + np.where(np.arange(14)[:, None] > 0, [value, 0.0, 0.0], 0.0), quaternion
    if kind == "lying":
        position = position.copy()
        position[:, 2] = 0.2
        quaternion = np.repeat(LYING[None], 14, 0)
    elif kind == "offset":
        position = position + np.where(np.arange(14)[:, None] > 0, [0.2, 0.0, 0.0], 0.0)
    return position, quaternion


def record(lib, commands, schedules, controls=CONTROLS):
    """A rollout (the arrays of readings_core) of the trials with these command rows, each trial's posture per control
    given by its schedule."""
    trials = len(commands)
    out = dict(body_pos_w=np.full((controls, trials, 14, 3), np.nan, np.float32),
               body_quat_w=np.full((controls, trials, 14, 4), np.nan, np.float32),
               command_row_before=np.full((controls, trials), -1, np.int32),
               command_row_after=np.full((controls, trials), -1, np.int32),
               clip_last_row=np.full((controls, trials), -1, np.int32),
               available=np.zeros((controls, trials), bool), done=np.zeros((controls, trials), bool),
               nonfinite=np.zeros((controls, trials), bool), contacts=np.zeros((controls, trials), np.int32),
               constraint_rows=np.zeros((controls, trials), np.int32),
               contacts_beyond_capacity=np.zeros((controls, trials), bool),
               rows_beyond_capacity=np.zeros((controls, trials), bool))
    for trial, (command, schedule) in enumerate(zip(commands, schedules)):
        for control in range(controls):
            last = int(lib.last[command])
            before, row = min(command + control, last), min(command + control + 1, last)
            out["command_row_before"][control, trial], out["command_row_after"][control, trial] = before, row
            out["clip_last_row"][control, trial] = last
            kind = schedule(control)
            if kind == "nan":
                out["done"][control, trial] = out["nonfinite"][control, trial] = True
                break
            if kind == "done":
                kind, out["done"][control, trial] = "exact", True
            out["body_pos_w"][control, trial], out["body_quat_w"][control, trial] = posture(lib, row, kind)
            out["available"][control, trial] = True
            if out["done"][control, trial]:
                break
    return follow(out, lib)


def follow(out, lib):
    """The command's bodies at the command row after each control (row 0 where the state is not available), as
    readings.py gives them; called again after an edit of the rows or of the available states."""
    index = np.where(out["available"], out["command_row_after"].astype(np.int64), 0)
    out["command_body_pos_w"], out["command_body_quat_w"] = lib.body_pos[index], lib.body_quat[index]
    return out


# The engine_rollouts record field of each array of readings_core.
RECORD_FIELDS = dict(body_pos_w="body_pos_w", body_quat_w="body_quat_w", command_row_before="frame_before",
                     command_row_after="frame_after", clip_last_row="clip_last_row", available="available", done="done",
                     nonfinite="termination_nonfinite", contacts="engine_contacts",
                     constraint_rows="engine_constraint_rows",
                     contacts_beyond_capacity="engine_contact_capacity_exceeded",
                     rows_beyond_capacity="engine_row_capacity_exceeded")


def engine_record(rollout, commands):
    """The rollout as nablamimic.engine_rollouts records it: its fields without the command's bodies, and each trial's
    command and body row."""
    return {**{field: rollout[name] for name, field in RECORD_FIELDS.items()},
            "command_row": np.asarray(commands, np.int32), "body_row": np.asarray(commands, np.int32)}


def phases(*spans):
    """A schedule of postures: ((until_control, kind), ...), the last kind to the end."""
    def schedule(control):
        for until, kind in spans:
            if control < until:
                return kind
        return spans[-1][1]
    return schedule


EXACT = phases((CONTROLS, "exact"))
# The placed starts: trial -> (schedule, expected status). Every trial's twin tracks its command exactly, but trial 4's
# twin, which lies down from control 60.
PLACED = {0: (phases((50, "lying"), (CONTROLS, "exact")), "rejoined"),
          1: (phases((50, "lying"), (CONTROLS, "offset")), "recovered_not_rejoined_at_horizon"),
          2: (phases((CONTROLS, "lying")), "fell_not_recovered_at_horizon"),
          3: (phases((50, "lying"), (200, "offset"), (CONTROLS, "lying")), "recovered_fell_again"),
          4: (phases((50, "lying"), (CONTROLS, "exact")), "recovered_rejoin_unavailable"),
          5: (phases((10, "lying"), (CONTROLS, "nan")), "numerical_failure"),
          6: (EXACT, "placed_start_outside_fall_rule")}
COMMANDS = [10 + 20*trial for trial in PLACED]


def readings(lib, placed, twins, trials):
    out, twin_out = {}, {}
    for trial in trials:
        twin = CORE.trial_columns(twins, trial, 0.0, PELVIS)
        twin_out[trial] = CORE.read_trial(twin, SETTINGS, DT, placed=False)
        out[trial] = CORE.read_trial(CORE.trial_columns(placed, trial, 0.0, PELVIS), SETTINGS, DT, placed=True,
                                     twin=twin)
    return out, twin_out


def records(lib):
    placed = record(lib, COMMANDS, [schedule for schedule, _ in PLACED.values()])
    twins = record(lib, COMMANDS, [phases((60, "exact"), (CONTROLS, "lying")) if trial == 4 else EXACT
                                   for trial in PLACED])
    return placed, twins


def pair_readings(schedules, twin_schedules=None, commands=None, lib=None, placed=True, edit=None,
                  controls=CONTROLS):
    """The readings of trials with these schedules and of their twins (by default tracking exactly); ``edit``
    changes the placed record before it is read."""
    lib = library() if lib is None else lib
    commands = [10 + 20*trial for trial in range(len(schedules))] if commands is None else commands
    out = record(lib, commands, schedules, controls)
    twins = record(lib, commands, twin_schedules or [EXACT]*len(schedules), controls)
    if edit is not None:
        edit(out)
        follow(out, lib)
    readings, twin_readings = {}, {}
    for trial in range(len(schedules)):
        twin = CORE.trial_columns(twins, trial, 0.0, PELVIS)
        twin_readings[trial] = CORE.read_trial(twin, SETTINGS, DT, placed=False)
        readings[trial] = CORE.read_trial(CORE.trial_columns(out, trial, 0.0, PELVIS), SETTINGS, DT,
                                          placed=placed, twin=twin if placed else None)
    return readings, twin_readings


class TrialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = library()
        cls.placed, cls.twins = records(cls.lib)
        cls.readings, cls.twin_readings = readings(cls.lib, cls.placed, cls.twins, PLACED)

    def test_each_hand_built_trial_reads_its_status(self):
        self.assertEqual({trial: reading["status"] for trial, reading in self.readings.items()},
                         {trial: expected for trial, (_, expected) in PLACED.items()})
        self.assertEqual({trial: reading["status"] for trial, reading in self.twin_readings.items()},
                         {**{trial: "no_fall" for trial in PLACED}, 4: "fell_not_recovered_at_horizon"})

    def test_a_trial_that_gets_up_and_rejoins_reads_its_controls(self):
        fall = self.readings[0]["falls"][0]
        self.assertEqual((fall["onset_control"], fall["onset_by_declaration"]), (0, True))
        # A hold of 0.5 s is 26 controls from the first upright control; the rejoin's trailing mean of 1 s (50
        # controls) is first defined 49 controls after the get-up and holds for 2 s (101 controls).
        self.assertEqual((fall["recovered_control"], fall["recovery_confirmed_control"]), (50, 75))
        self.assertAlmostEqual(fall["time_to_recover_s"], 1.0)
        rejoin = fall["rejoin"]
        self.assertEqual((rejoin["rejoined_control"], rejoin["rejoin_confirmed_control"]), (99, 199))
        self.assertAlmostEqual(rejoin["time_to_rejoin_s"], 1.98)
        self.assertEqual(self.readings[0]["support"]["end_reason"], "horizon")

    def test_a_trial_that_gets_up_without_rejoining_and_one_that_falls_again(self):
        self.assertEqual(self.readings[1]["falls"][0]["rejoin"]["status"], "not_rejoined")
        falls = self.readings[3]["falls"]
        self.assertEqual([fall["onset_control"] for fall in falls], [0, 200])
        self.assertEqual((falls[0]["recovered_control"], falls[1]["recovered"]), (50, False))
        self.assertEqual(falls[0]["rejoin"]["status"], "not_rejoined")
        self.assertEqual(self.readings[2]["falls"], [dict(onset_control=0, onset_by_declaration=True,
                                                          recovered=False)])

    def test_a_rejoin_is_unavailable_where_the_twin_falls_first(self):
        reading = self.readings[4]
        self.assertEqual(reading["falls"][0]["rejoin"], dict(status="unavailable",
                                                             reason="twin fell: onset at twin control 60"))
        self.assertEqual(reading["twin"]["usable_end"], 60)
        self.assertEqual(CORE.rejoin_unavailable_cause(reading, self.twin_readings[4]), "twin_fell")

    def test_a_numerical_failure_ends_the_support_before_its_control(self):
        self.assertEqual({key: self.readings[5]["support"][key] for key in ("end", "end_reason")},
                         dict(end=10, end_reason="numerical_failure"))
        self.assertIs(self.readings[6]["placed_start_in_fall_rule"], False)

    def test_the_counts_of_the_placed_starts_and_twins(self):
        # R: placed starts that get up; J: placed starts that rejoin; F: twins with a counted fall.
        counts = CORE.placed_quantities(self.readings, self.twin_readings)
        self.assertEqual({key: counts["R"][key] for key in ("numerator", "denominator")},
                         dict(numerator=4, denominator=7))
        self.assertEqual(counts["R"]["share"], 4/7)
        self.assertEqual((counts["R"]["numerical_failure"], counts["R"]["placed_start_outside_fall_rule"]), ([5], [6]))
        self.assertEqual((counts["J"]["numerator"], counts["J"]["recovered_rejoin_unavailable_by_cause"]),
                         (1, {"twin_fell": [4]}))
        # Rejoined over the recovered trials whose rejoin was readable: trial 4 counts neither way.
        self.assertEqual((counts["J_over_R"]["value"], counts["J_over_R"]["denominator"]), (1/3, 3))
        self.assertEqual(counts["J_over_R"]["over_every_recovered_trial"]["value"], 1/4)
        self.assertEqual((counts["F"]["numerator"], counts["F"]["twins_with_a_fall"],
                          counts["F"]["rejoin_readings_removed_by_a_twin_fall"]), (1, [4], 1))
        self.assertEqual(sum(counts["statuses"].values()), 7)
        self.assertEqual(counts["time_to_rejoin_s"][0][0], 0)
        self.assertAlmostEqual(counts["time_to_rejoin_s"][0][1], 1.98)
        json.dumps(counts, allow_nan=False)

    def test_a_twin_down_from_control_0_counts_no_fall_and_leaves_its_trial_s_rejoin_unavailable(self):
        readings, twins = pair_readings([phases((50, "lying"), (CONTROLS, "exact"))],
                                        twin_schedules=[phases((CONTROLS, "lying"))])
        self.assertEqual((twins[0]["falls"], twins[0]["status"]), ([], "fall_rule_met_not_counted"))
        self.assertEqual(twins[0]["not_counted_as_falls"], [dict(kind="at_origin", control=0, controls=CONTROLS)])
        self.assertEqual(readings[0]["status"], "recovered_rejoin_unavailable")
        self.assertEqual(CORE.rejoin_unavailable_cause(readings[0], twins[0]),
                         "twin_in_the_fall_rule_where_no_onset_counts")
        counts = CORE.placed_quantities(readings, twins)
        self.assertEqual((counts["F"]["numerator"], counts["F"]["twins_in_the_fall_rule_not_counted"]), (0, [0]))
        self.assertEqual(counts["J"]["recovered_rejoin_unavailable_by_cause"],
                         {"twin_in_the_fall_rule_where_no_onset_counts": [0]})
        self.assertIsNone(counts["J_over_R"]["value"])

    def test_a_trial_that_gets_up_and_then_fails_numerically_counts_as_not_got_up(self):
        readings, twins = pair_readings([phases((50, "lying"), (150, "exact"), (CONTROLS, "nan")),
                                         phases((50, "lying"), (CONTROLS, "exact"))])
        self.assertEqual((readings[0]["status"], readings[0]["support"]["end"]), ("numerical_failure", 150))
        self.assertEqual(readings[0]["falls"][0]["recovered_control"], 50)
        counts = CORE.placed_quantities(readings, twins)
        self.assertEqual((counts["R"]["numerator"], counts["R"]["numerical_failure"],
                          counts["R"]["got_up_before_numerical_failure"]), (1, [0], [0]))

    def test_a_rejoin_is_unavailable_where_the_twin_s_rows_differ_or_its_support_ends(self):
        lib = library()
        placed = record(lib, [10], [phases((50, "lying"), (CONTROLS, "exact"))])
        for twin_record, cause in ((record(lib, [11], [EXACT]), "twin_command_row_differs"),
                                   (record(lib, [10], [phases((60, "exact"), (CONTROLS, "done"))]),
                                    "twin_support_ended: environment_done")):
            twin = CORE.trial_columns(twin_record, 0, 0.0, PELVIS)
            reading = CORE.read_trial(CORE.trial_columns(placed, 0, 0.0, PELVIS), SETTINGS, DT, placed=True,
                                      twin=twin)
            self.assertEqual(reading["status"], "recovered_rejoin_unavailable")
            twin_reading = CORE.read_trial(twin, SETTINGS, DT, placed=False)
            self.assertEqual(CORE.rejoin_unavailable_cause(reading, twin_reading), cause)

    def test_a_trial_that_rejoins_and_then_falls_again_reads_rejoined(self):
        readings, _ = pair_readings([phases((50, "lying"), (300, "exact"), (CONTROLS, "lying"))])
        self.assertEqual(readings[0]["status"], "rejoined")
        self.assertEqual([fall["onset_control"] for fall in readings[0]["falls"]], [0, 300])


class ThresholdTest(unittest.TestCase):
    """One trial on each side of each declared value: the get-up's 0.8 of the command's pelvis height (0.624 m under
    the 0.78 m command), its 30 degrees and its 0.5 s hold (26 controls), the rejoin's 0.05 m excess, and the fall
    rule's 0.50 m and 60 degrees."""

    def statuses(self, schedules, placed=True):
        readings, _ = pair_readings(schedules, placed=placed)
        return [readings[trial] for trial in range(len(schedules))]

    def test_the_get_up_and_rejoin_values(self):
        cases = {("height", 0.63): "rejoined", ("height", 0.60): "fell_not_recovered_at_horizon",
                 ("tilt", 25.0): "rejoined", ("tilt", 35.0): "fell_not_recovered_at_horizon",
                 ("offset", 0.04*14/13): "rejoined", ("offset", 0.06*14/13): "recovered_not_rejoined_at_horizon"}
        readings = self.statuses([phases((50, "lying"), (CONTROLS, kind)) for kind in cases])
        self.assertEqual([reading["status"] for reading in readings], list(cases.values()))

    def test_the_rejoin_compares_each_side_in_its_own_pelvis_heading_frame(self):
        # Turned about z around the pelvis with its orientations, the body reads zero error; with its positions alone
        # turned it does not.
        turned_whole, turned_positions = self.statuses([phases((50, "lying"), (CONTROLS, ("turn", 0.7))),
                                                        phases((50, "lying"), (CONTROLS, ("turn_positions", 0.7)))])
        self.assertEqual((turned_whole["status"], turned_whole["falls"][0]["rejoin"]["rejoined_control"]),
                         ("rejoined", 99))
        self.assertEqual(turned_positions["status"], "recovered_not_rejoined_at_horizon")
        columns = CORE.trial_columns(record(library(), [10], [phases((CONTROLS, ("turn", 0.7)))]), 0, 0.0, PELVIS)
        np.testing.assert_allclose(CORE.body_errors(columns), 0.0, atol=1e-5)

    def test_the_get_up_needs_a_hold_of_26_controls(self):
        short, held = self.statuses([phases((50, "lying"), (75, "exact"), (100, "lying"), (CONTROLS, "exact")),
                                     phases((50, "lying"), (76, "exact"), (CONTROLS, "lying"))])
        self.assertEqual(short["falls"][0]["recovered_control"], 100)
        self.assertEqual((held["falls"][0]["recovered_control"], held["falls"][0]["recovery_confirmed_control"]),
                         (50, 75))
        self.assertEqual((held["status"], held["falls"][1]["onset_control"]), ("recovered_fell_again", 76))

    def test_the_fall_rule_values(self):
        cases = {("height", 0.52): [], ("height", 0.48): [10], ("tilt", 55.0): [], ("tilt", 65.0): [10]}
        readings = self.statuses([phases((10, "exact"), (CONTROLS, kind)) for kind in cases], placed=False)
        self.assertEqual([[fall["onset_control"] for fall in reading["falls"]] for reading in readings],
                         list(cases.values()))


def turned(lib, row, angle, shift, lift=0.0, about=TORSO):
    """The command's bodies turned by ``angle`` about z around the body ``about`` (the torso by default), moved by
    ``shift`` in the plane and lifted by ``lift``."""
    position = lib.body_pos[row].astype(np.float64)
    torso = position[about]
    relative = position - torso
    rotated = np.stack([math.cos(angle)*relative[:, 0] - math.sin(angle)*relative[:, 1],
                        math.sin(angle)*relative[:, 0] + math.cos(angle)*relative[:, 1], relative[:, 2]], -1)
    quaternion = np.repeat([[math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)]], 14, 0)
    return rotated + torso + [shift[0], shift[1], lift], quaternion


class SupportTest(unittest.TestCase):
    """Each way a trial's support can end, and the statuses of a trial that ends before the horizon; and a fall rule
    met under a low command, which is no onset."""

    def test_each_end_of_the_support(self):
        def skip_a_row(out):
            out["command_row_after"][300:, 0] += 1

        def unavailable(out):
            out["available"][400, 0] = False

        exact, lying = EXACT, phases((CONTROLS, "lying"))
        cases = (  # schedule, placed start, command row, record edit, controls -> end reason, end, status
            (exact, False, 10, skip_a_row, CONTROLS, "reference_row_discontinuity", 300, "no_fall_ended_early"),
            (phases((30, "exact"), (CONTROLS, "done")), False, 10, None, CONTROLS, "environment_done", 31,
             "no_fall_ended_early"),
            (exact, False, 1150, None, CONTROLS, "reference_end", 49, "no_fall_ended_early"),
            (exact, False, 10, None, 500, "record_end", 500, "no_fall_ended_early"),
            (exact, False, 10, unavailable, CONTROLS, "post_step_state_unavailable", 400, "no_fall_ended_early"),
            (phases((30, "lying"), (CONTROLS, "done")), False, 10, None, CONTROLS, "environment_done", 31,
             "fall_rule_met_not_counted_ended_early"),
            (phases((30, "lying"), (CONTROLS, "done")), True, 10, None, CONTROLS, "environment_done", 31,
             "fell_ended_early"),
            (phases((50, "lying"), (300, "offset"), (CONTROLS, "done")), True, 10, None, CONTROLS, "environment_done",
             301, "recovered_ended_early"),
            (lying, True, 10, skip_a_row, CONTROLS, "reference_row_discontinuity", 300, "fell_ended_early"))
        for schedule, placed, command, edit, controls, reason, end, status in cases:
            readings, _ = pair_readings([schedule], commands=[command], placed=placed, edit=edit, controls=controls)
            reading = readings[0]
            self.assertEqual((reading["support"]["end_reason"], reading["support"]["end"], reading["status"]),
                             (reason, end, status))

    def test_a_fall_rule_met_under_a_low_command_is_no_onset_and_is_listed_once_the_command_rises(self):
        # The command of row 100 passes the low rows 200-299 at controls 99-198.
        lib = library(low=range(200, 300))
        readings, _ = pair_readings([EXACT, phases((150, "exact"), (CONTROLS, "lying"))], commands=[100, 100],
                                    lib=lib, placed=False)
        self.assertEqual((readings[0]["falls"], readings[0]["not_counted_as_falls"], readings[0]["status"]),
                         ([], [], "no_fall"))
        self.assertEqual(readings[1]["falls"], [])
        self.assertEqual(readings[1]["not_counted_as_falls"],
                         [dict(kind="at_command_rise", control=199, controls=CONTROLS - 199)])

    def test_a_rejoin_is_unavailable_where_the_excess_is_undefined_where_a_hold_could_lie(self):
        def holes(out):
            out["body_pos_w"][100::50, 0, 3] = np.nan

        readings, _ = pair_readings([phases((50, "lying"), (CONTROLS, "exact"))], edit=holes)
        self.assertEqual(readings[0]["status"], "recovered_rejoin_unavailable")
        self.assertIn("excess over the twin undefined", readings[0]["falls"][0]["rejoin"]["reason"])


class FirstRowTest(unittest.TestCase):
    def first_rows(self, lib, poses, done_at=None):
        out = record(lib, [0, 1200], [EXACT, EXACT], controls=1199)
        for trial, pose in enumerate(poses):
            for control in range(1199):
                row = out["command_row_after"][control, trial]
                if out["available"][control, trial]:
                    out["body_pos_w"][control, trial], out["body_quat_w"][control, trial] = pose(row)
            if done_at is not None and trial == 1:
                out["done"][done_at, trial] = True
                out["available"][done_at + 1:, trial] = False
        return follow(out, lib)

    def test_a_rigid_planar_shift_and_turn_reads_zero_error_and_a_lift_reads_its_height(self):
        lib = library()
        out = self.first_rows(lib, [lambda row: turned(lib, row, 0.7, (0.3, -0.4)),
                                    lambda row: turned(lib, row, -1.2, (1.0, 2.0), lift=0.03)])
        # C: first-row clips completed; E_m: the aligned body-position error E (m) over the clips.
        reading = CORE.first_row_reading(out, NAMES, STARTS, ["getup"], TORSO)
        errors = [clip["aligned_body_error_m"] for clip in reading["clips"]]
        self.assertAlmostEqual(errors[0], 0.0, delta=1e-5)
        self.assertAlmostEqual(errors[1], 0.03, delta=1e-5)
        self.assertAlmostEqual(reading["E_m"]["value"], 0.015, delta=1e-5)
        self.assertEqual((reading["C"], reading["C_getup"]), (dict(completed=2, of=2), {"getup": True}))
        self.assertEqual([clip["support_controls"] for clip in reading["clips"]], [1199, 99])
        self.assertGreater(reading["global_body_error_m"], 0.3)

    def test_a_clip_ended_by_the_termination_is_read_before_its_end_and_not_completed(self):
        lib = library()
        out = self.first_rows(lib, [lambda row: turned(lib, row, 0.0, (0.0, 0.0)),
                                    lambda row: turned(lib, row, 0.0, (0.0, 0.0), lift=0.01*(row - 1200))],
                              done_at=5)
        reading = CORE.first_row_reading(out, NAMES, STARTS, ["getup"], TORSO)
        getup = reading["clips"][1]
        self.assertEqual((getup["support_controls"], getup["first_termination"], getup["completed"]), (5, 5, False))
        # The lift at control i is 0.01 (i + 1) m: the mean over controls 0-4 is 0.03 m.
        self.assertAlmostEqual(getup["aligned_body_error_m"], 0.03, delta=1e-5)
        self.assertEqual((reading["C"]["completed"], reading["C_getup"]), (1, {"getup": False}))

    def test_a_clip_whose_rows_skip_is_not_completed_and_capacity_is_counted_in_its_support(self):
        lib = library()
        out = record(lib, [0, 1200], [EXACT, EXACT], controls=1199)
        out["command_row_after"][500:, 0] += 1
        out["contacts_beyond_capacity"][[3, 4], 0] = True
        out["rows_beyond_capacity"][[4, 7], 1] = True
        out["rows_beyond_capacity"][150, 1] = True  # beyond the getup clip's 99 controls
        reading = CORE.first_row_reading(follow(out, lib), NAMES, STARTS, ["getup"], TORSO)
        walk, getup = reading["clips"]
        self.assertEqual((walk["rows_advance_by_one"], walk["completed"], walk["support_controls"]),
                         (False, False, 1199))
        self.assertEqual((getup["rows_advance_by_one"], getup["completed"]), (True, True))
        self.assertEqual([clip["capacity_exceeded_controls_in_support"] for clip in reading["clips"]], [2, 2])
        self.assertEqual(reading["C"], dict(completed=1, of=2))

    def test_the_first_row_rollout_must_start_each_clip_at_its_first_row_and_hold_its_rows_as_an_available_prefix(self):
        lib = library()
        for commands in ([0, 1201], [1200, 0]):
            with self.assertRaisesRegex(ValueError, "first row", msg=commands):
                CORE.first_row_reading(record(lib, commands, [EXACT]*2, controls=1199), NAMES, STARTS, [], TORSO)
        with self.assertRaisesRegex(ValueError, "not one per clip"):
            CORE.first_row_reading(record(lib, [0], [EXACT], controls=1199), NAMES, STARTS, [], TORSO)
        with self.assertRaises(ValueError):
            CORE.first_row_reading(record(lib, [0, 1200, 1200], [EXACT]*3, controls=1199), NAMES, STARTS, [], TORSO)
        with self.assertRaisesRegex(ValueError, "walk needs 1199"):
            CORE.first_row_reading(record(lib, [0, 1200], [EXACT, EXACT], controls=1000), NAMES, STARTS, [], TORSO)
        out = record(lib, [0, 1200], [EXACT, EXACT], controls=1199)
        out["available"][10, 1] = False
        with self.assertRaisesRegex(ValueError, "getup"):
            CORE.first_row_reading(follow(out, lib), NAMES, STARTS, [], TORSO)

    def test_the_engine_report_reads_available_controls(self):
        out = record(library(), [10, 30, 50], [EXACT, EXACT, phases((20, "exact"), (CONTROLS, "nan"))])
        out["contacts"][5, 1], out["constraint_rows"][7, 0] = 40, 90
        out["contacts_beyond_capacity"][9, 0] = True
        out["rows_beyond_capacity"][30, 2] = True  # after trial 2's end
        out["contacts"][30, 2] = 500
        self.assertEqual(CORE.engine_report(out), dict(largest_contacts=40, largest_constraint_rows=90,
                                                       trials_with_capacity_exceeded=1))


class CoreTest(unittest.TestCase):
    """The core's save reading, its refusal of other arrays, and its imports."""

    def test_a_save_is_read_from_its_three_rollouts(self):
        lib = library()
        placed, twins = records(lib)
        first_rows = record(lib, [0, 1200], [EXACT, EXACT], controls=1199)
        found = CORE.save_readings(placed, twins, first_rows, SETTINGS, DT, PELVIS, TORSO, NAMES, STARTS, ["getup"])
        quantities = found["quantities"]
        self.assertEqual((quantities["R"]["numerator"], quantities["J"]["numerator"], quantities["F"]["numerator"]),
                         (4, 1, 1))
        self.assertEqual({trial: reading["status"] for trial, reading in found["readings"].items()},
                         {trial: status for trial, (_, status) in PLACED.items()})
        self.assertEqual(found["twin_readings"][4]["status"], "fell_not_recovered_at_horizon")
        self.assertEqual((found["first_rows"]["C"], found["engine"]["placed"]["trials_with_capacity_exceeded"]),
                         (dict(completed=2, of=2), 0))

    def test_rollouts_without_an_array_with_another_shape_or_of_other_trials_are_refused(self):
        lib = library()
        placed, twins = records(lib)
        first_rows = record(lib, [0, 1200], [EXACT, EXACT], controls=1199)
        fewer = record(lib, COMMANDS[:-1], [EXACT]*(len(COMMANDS) - 1))
        for name, changed in (("placed", {k: v for k, v in placed.items() if k != "nonfinite"}),
                              ("twins", {**twins, "done": twins["done"][:, :, None]}),
                              ("first_rows", {**first_rows, "body_quat_w": first_rows["body_quat_w"][..., :3]}),
                              ("placed", {**placed, "command_body_pos_w": placed["command_body_pos_w"][:, :, :13]}),
                              ("twins", fewer)):
            given = {**dict(placed=placed, twins=twins, first_rows=first_rows), name: changed}
            with self.assertRaises(ValueError, msg=name):
                CORE.save_readings(given["placed"], given["twins"], given["first_rows"], SETTINGS, DT, PELVIS, TORSO,
                                   NAMES, STARTS, ["getup"])

    def test_the_core_imports_numpy_and_the_standard_library_alone_and_reads_no_file(self):
        text = (ROOT/"studies/recovery/readings_core.py").read_text()
        tree = ast.parse(text)
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {"math", "numpy"})
        for call in ("open(", "np.load", "read_text", "read_bytes"):
            self.assertNotIn(call, text)


class EntryTest(unittest.TestCase):
    """readings.py's arrays for the core from engine_rollouts records."""

    @classmethod
    def setUpClass(cls):
        cls.entry = load("readings")

    def test_the_core_is_given_the_record_s_fields_and_the_library_s_bodies_at_the_row_after_each_control(self):
        lib = library()
        rollout = record(lib, COMMANDS, [schedule for schedule, _ in PLACED.values()])
        arrays = self.entry.rollout_arrays(engine_record(rollout, COMMANDS), lib)
        self.assertEqual(set(arrays), set(rollout))
        for name, value in rollout.items():
            np.testing.assert_array_equal(arrays[name], value, err_msg=name)

    def test_the_pelvis_and_torso_indices_are_those_of_the_tracked_bodies(self):
        from nablamimic.env import ANCHOR, BM_BODIES
        self.assertEqual((BM_BODIES.index("pelvis"), BM_BODIES.index(ANCHOR)), (PELVIS, TORSO))


class SaveTest(unittest.TestCase):
    """read_save on rollout directories as engine_rollouts writes them."""

    def write(self, directory, lib):
        directory = Path(directory)
        placed, twins = (engine_record(rollout, COMMANDS) for rollout in records(lib))
        first_rows = engine_record(record(lib, [0, 1200], [EXACT, EXACT], controls=1199), [0, 1200])
        scene = {"placed.json": {"trials": [{"command_row": row, "body_row": row} for row in COMMANDS]},
                 "twins.json": {"trials": [{"command_row": row} for row in COMMANDS]},
                 "first_rows.json": {"trials": [{"command_row": 0}, {"command_row": 1200}]}}
        inputs = dict(LAFAN1_G1_DIR="a", MOTION_CACHE_DIR="b", MENAGERIE_DIR="c")
        trials = [dict(body_clip="walk", command_clip="walk", pelvis_yaw_body_minus_command_rad=0.1*trial)
                  for trial in range(len(COMMANDS))]
        (directory/"scene").mkdir()
        for name, content in {**scene, "scene.json": dict(first_row_controls=1199, library_inputs=inputs,
                                                          trials=trials)}.items():
            (directory/"scene"/name).write_text(json.dumps(content))
        rollouts = {}
        for name, data, termination, controls in (("placed", placed, "nonfinite", 1000),
                                                  ("twins", twins, "nonfinite", 1000),
                                                  ("first_rows", first_rows, "recipe", 1199)):
            path = directory/name
            path.mkdir()
            np.savez(path/"rollouts.npz", **data)
            (path/"description.json").write_text(json.dumps(dict(
                run="/runs/control_s0", checkpoint="/runs/control_s0/ckpt/it_003000.pkl", completed_updates=3000,
                actor_key="actor_ema", run_source_revision="abc", source_revision="abc", gpu="card",
                scene_content=scene[f"{name}.json"], termination=termination, controls=controls,
                settings=dict(inputs=inputs), engine_diagnostics={})))
            rollouts[name] = path
        return directory/"scene", rollouts

    def change(self, path, **changes):
        description = json.loads((path/"description.json").read_text())
        (path/"description.json").write_text(json.dumps({**description, **changes}))

    def test_a_save_is_read_from_its_three_rollouts_and_other_records_are_refused(self):
        entry, lib = load("readings"), library()
        with tempfile.TemporaryDirectory() as directory:
            scene, rollouts = self.write(directory, lib)
            result, lines = entry.read_save(DECLARATION, scene, rollouts, lib, PELVIS, TORSO, ["getup"])
            self.assertEqual((result["R"]["numerator"], result["J"]["numerator"], result["F"]["numerator"]),
                             (4, 1, 1))
            self.assertEqual((result["first_rows"]["C"]["completed"], result["save"]["completed_updates"]), (2, 3000))
            self.assertEqual([line["reading"]["status"] for line in lines], [status for _, status in PLACED.values()])
            self.assertEqual([(item["trial"], item["status"], item["rad"])
                              for item in result["pelvis_yaw_at_placement"]],
                             [(trial, status, 0.1*trial) for trial, (_, status) in PLACED.items()])
            json.dumps([result, lines], allow_nan=False)
            for name, changes in (("twins", dict(termination="recipe")), ("first_rows", dict(controls=1000)),
                                  ("placed", dict(actor_key="actor")), ("twins", dict(completed_updates=1500)),
                                  ("placed", dict(settings=dict(inputs={}))),
                                  ("first_rows", dict(scene_content={"trials": []}))):
                original = (rollouts[name]/"description.json").read_text()
                self.change(rollouts[name], **changes)
                with self.assertRaises(ValueError, msg=f"{name} {changes}"):
                    entry.read_save(DECLARATION, scene, rollouts, lib, PELVIS, TORSO, ["getup"])
                (rollouts[name]/"description.json").write_text(original)
            for name, field, trial in (("placed", "command_row", 2), ("placed", "body_row", 3),
                                       ("twins", "command_row", 0)):
                with np.load(rollouts[name]/"rollouts.npz") as stored:
                    data = {key: stored[key] for key in stored.files}
                original = data[field].copy()
                data[field][trial] += 1
                np.savez(rollouts[name]/"rollouts.npz", **data)
                with self.assertRaisesRegex(ValueError, "command and body rows", msg=f"{name} {field}"):
                    entry.read_save(DECLARATION, scene, rollouts, lib, PELVIS, TORSO, ["getup"])
                data[field] = original
                np.savez(rollouts[name]/"rollouts.npz", **data)
            description = json.loads((rollouts["twins"]/"description.json").read_text())
            del description["engine_diagnostics"]
            (rollouts["twins"]/"description.json").write_text(json.dumps(description))
            with self.assertRaises(ValueError):
                entry.read_save(DECLARATION, scene, rollouts, lib, PELVIS, TORSO, ["getup"])
            twins = {"trials": [{"command_row": row + 1} for row in COMMANDS]}
            (scene/"twins.json").write_text(json.dumps(twins))
            with self.assertRaisesRegex(ValueError, "twins' command rows"):
                entry.read_save(DECLARATION, scene, rollouts, lib, PELVIS, TORSO, ["getup"])


if __name__ == "__main__":
    unittest.main()
