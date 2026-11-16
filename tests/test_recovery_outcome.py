"""The outcome and exposure programs of the recovery pair (studies/recovery/outcome.py and exposure.py) on hand-built
readings, run records and logs; no engine or GPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"studies/recovery"))  # the study's programs import declaration.py beside them


def load(name):
    spec = importlib.util.spec_from_file_location(f"recovery_{name}", ROOT/f"studies/recovery/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OUTCOME, EXPOSURE, DECLARATION_PROGRAM = load("outcome"), load("exposure"), load("declaration")
DECLARATION = json.loads((ROOT/"studies/recovery/pair.json").read_text())
HIDDEN = ("fallAndGetUp1_subject1", "fallAndGetUp1_subject5", "fallAndGetUp2_subject2", "fallAndGetUp2_subject3",
          "fallAndGetUp3_subject1")
# Example readings of the eight saves: R (placed starts that get up), J over R (rejoined over the trials that got up
# with a readable rejoin), F (twins with a counted fall), C (first-row clips completed) and E (aligned body-position
# error, m).
SAVES = {("control", 0, 1500): (24, 22/24, 0, 40, 0.0345), ("control", 0, 3000): (35, 34/35, 0, 40, 0.0286),
         ("control", 1, 1500): (26, 1.0, 1, 40, 0.0343), ("control", 1, 3000): (58, 55/58, 0, 40, 0.0285),
         ("recovery", 0, 1500): (99, 1.0, 2, 36, 0.0542), ("recovery", 0, 3000): (99, 1.0, 1, 39, 0.0390),
         ("recovery", 1, 1500): (99, 98/99, 0, 38, 0.0533), ("recovery", 1, 3000): (99, 1.0, 0, 39, 0.0385)}


def values(changes=None, getup=True):
    out = {key: dict(R=r, J_over_R=j, F=f, C=c, E_m=e, C_getup={clip: getup for clip in HIDDEN})
           for key, (r, j, f, c, e) in SAVES.items()}
    for key, change in (changes or {}).items():
        out[key] = {**out[key], **change}
    return out


def at(**saves):
    """Changes {(condition, seed, updates): {name: value}} from keywords such as recovery_0_3000=dict(R=3)."""
    return {(name.split("_")[0], int(name.split("_")[1]), int(name.split("_")[2])): change
            for name, change in saves.items()}


class OutcomeOrderTest(unittest.TestCase):
    def outcome(self, changes=None, getup=True):
        return OUTCOME.outcome(DECLARATION, values(changes, getup))

    def test_the_example_saves_give_tracking_cost_at_exposure(self):
        result = self.outcome()
        self.assertEqual(result["outcome"], "tracking_cost_at_exposure")
        self.assertEqual(result["classes"], {0: "gets_up", 1: "gets_up"})
        self.assertEqual(result["larger_control_R"], 58)
        self.assertEqual((result["margins"]["tracking_E_m_1500"], result["margins"]["tracking_C_3000"]), (0.005, 2))
        self.assertEqual(result["tracking"]["seed 0 against control 3000"]["terms"], dict(C=True, E=False, F=True))
        self.assertTrue(result["tracking"]["seed 1 against control 1500"]["within"])
        self.assertEqual([row["name"] for row in result["rows_in_order"]],
                         [item["name"] for item in DECLARATION["outcomes"]])

    def test_hand_built_readings_reach_each_outcome_in_the_declared_order(self):
        # run_stopped, the first row, is decided by read_pair before any save is read (PairTest).
        cases = {
            "split_seeds": dict(changes=at(recovery_1_3000=dict(R=3))),
            "control_within": dict(changes=at(control_0_3000=dict(R=95), control_1_3000=dict(R=92))),
            "precondition_unmet": dict(changes=at(recovery_0_3000=dict(R=30), recovery_1_3000=dict(R=30)),
                                       getup=False),
            "still_rising": dict(changes=at(recovery_0_1500=dict(R=10), recovery_0_3000=dict(R=30),
                                            recovery_1_1500=dict(R=10), recovery_1_3000=dict(R=30))),
            "no_get_up": dict(changes=at(recovery_0_1500=dict(R=1), recovery_0_3000=dict(R=2),
                                         recovery_1_1500=dict(R=2), recovery_1_3000=dict(R=3))),
            "some_get_up": dict(changes=at(recovery_0_1500=dict(R=28), recovery_0_3000=dict(R=30),
                                           recovery_1_1500=dict(R=2), recovery_1_3000=dict(R=3))),
            "gets_up_without_rejoining": dict(changes=at(recovery_0_3000=dict(J_over_R=0.3))),
            "gets_up_and_tracks": dict(changes=at(recovery_0_3000=dict(E_m=0.030), recovery_1_3000=dict(E_m=0.030))),
            "tracking_cost_at_exposure": dict(),
            "tracking_worse": dict(changes=at(recovery_0_3000=dict(E_m=0.05), recovery_1_3000=dict(E_m=0.05))),
            "no_call": dict(changes=at(recovery_0_3000=dict(E_m=0.030)))}
        self.assertEqual(["run_stopped", *cases], [item["name"] for item in DECLARATION["outcomes"]])
        for name, case in cases.items():
            result = self.outcome(**case)
            self.assertEqual(result["outcome"], name, name)
            self.assertFalse(result["rows_in_order"][0]["holds"])
        self.assertEqual(self.outcome(at(recovery_0_3000=dict(R=3)))["outcome"], "split_seeds")

    def test_an_unavailable_j_over_r_holds_neither_below_nor_at_its_margin(self):
        result = self.outcome(at(recovery_0_3000=dict(J_over_R=None)))
        self.assertEqual((result["outcome"], result["J_over_R_unavailable_at_seeds"]), ("no_call", [0]))

    def test_the_margins_widen_to_the_control_seeds_spread(self):
        bounds = OUTCOME.margins(DECLARATION, values(at(control_1_3000=dict(C=35, E_m=0.0386),
                                                        recovery_1_3000=dict(R=80))))
        self.assertEqual(bounds["tracking_C_3000"], 5)
        self.assertAlmostEqual(bounds["tracking_E_m_3000"], 0.01)
        self.assertEqual(bounds["still_rising"], 19)


class BoundaryTest(unittest.TestCase):
    """Each class, margin and row at its declared value and one step beyond it."""

    def test_the_classes_at_their_bounds(self):
        # near_zero at R <= 5; gets_up at R >= 50 and R >= the larger control R at 3,000 + 20, here 58 + 20 or, with
        # the control R lowered to 20 and 25, 25 + 20 = 45 below 50.
        low_control = at(control_0_3000=dict(R=20), control_1_3000=dict(R=25))
        cases = ((5, None, "near_zero"), (6, None, "some"), (77, None, "some"), (78, None, "gets_up"),
                 (49, low_control, "some"), (50, low_control, "gets_up"))
        for r, changes, expected in cases:
            found = OUTCOME.classes(DECLARATION, values({**(changes or {}), **at(recovery_0_3000=dict(R=r),
                                                                               recovery_1_3000=dict(R=r))}))
            self.assertEqual(found, {0: expected, 1: expected}, (r, changes))

    def test_tracking_within_at_its_margins(self):
        # Against control seed 0 at 3,000 (C 40, E 0.0286 m, F 0) with the margins 2, 0.005 m and 2.
        control = values()["control", 0, 3000]
        for change, within in ((dict(C=38), True), (dict(C=37), False), (dict(E_m=control["E_m"] + 0.005), True),
                               (dict(E_m=control["E_m"] + 0.005 + 1e-9), False), (dict(F=2), True),
                               (dict(F=3), False)):
            found = values(at(recovery_0_3000=dict(dict(C=40, E_m=control["E_m"], F=0), **change)))
            bounds = OUTCOME.margins(DECLARATION, found)
            self.assertEqual(OUTCOME.tracking_within(found, bounds, 0, 3000, 3000)["within"], within, change)

    def test_still_rising_above_its_margin_and_control_within_at_its_margin(self):
        # Both recovery seeds at R 30 at 3,000: the margin is max(|30 - 30|, 5) = 5.
        for rise, expected in ((5, False), (6, True)):
            found = values(at(recovery_0_1500=dict(R=30 - rise), recovery_0_3000=dict(R=30),
                              recovery_1_1500=dict(R=30 - rise), recovery_1_3000=dict(R=30)))
            self.assertEqual(OUTCOME.outcome_rules(DECLARATION, found)["rules"]["still_rising"], expected, rise)
        # The control R at 3,000 within 10 of the recovery R (99) at both seeds.
        for control_r, expected in ((89, True), (88, False)):
            found = values(at(control_0_3000=dict(R=control_r), control_1_3000=dict(R=95)))
            self.assertEqual(OUTCOME.outcome_rules(DECLARATION, found)["rules"]["control_within"], expected, control_r)


RECOVERY_OPTIONS = OUTCOME.declared_recovery_options(DECLARATION)


def engine(condition, **changes):
    """The engine record of a config.json, with the library where the condition's job put it."""
    return {"key": "lib-1", "sources_sha256": "s", "consts_sha256": "c", "maxcon": 96, "path": f"/jobs/{condition}/lib",
            "artifact": f"/jobs/{condition}/artifact", **changes}


def config(condition, run_seed, **changes):
    """A config.json of the trainings' fields that the programs read, with ``changes``."""
    return {"seed": run_seed, "iters": 3000, "ckpt_every": 250, "hours": None, "num_envs": 4096, "horizon": 64,
            "out": f"runs/{condition}_s{run_seed}", "config": DECLARATION["trainings"]["conditions"][condition],
            "gpu": "card", "git_commit": "abc", "declared_environment": "image", "torch_version": "2.9.1",
            "engine": engine(condition), **RECOVERY_OPTIONS[condition], **changes}


def reading(condition, seed, updates, **changes):
    r, j, f, c, e = SAVES[condition, seed, updates]
    out = dict(save=dict(run=str(Path(f"/runs/{condition}_s{seed}").resolve()), completed_updates=updates),
               trials=100, R=dict(numerator=r), J_over_R=dict(value=j), F=dict(numerator=f),
               first_rows=dict(C=dict(completed=c), E_m=dict(value=e), C_getup={clip: True for clip in HIDDEN}))
    for key, value in changes.items():
        out[key] = {**out[key], **value}
    return out


COMPLETED = dict(completed=True, cause=None, completed_updates=3000, detail=None)


class DeclarationTest(unittest.TestCase):
    def test_a_declaration_with_another_key_or_without_one_is_refused(self):
        self.assertEqual(DECLARATION_PROGRAM.read_declaration(ROOT/"studies/recovery/pair.json"), DECLARATION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"pair.json"
            for changed in ({**DECLARATION, "extra": 1}, {k: v for k, v in DECLARATION.items() if k != "quantities"}):
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    DECLARATION_PROGRAM.read_declaration(path)

    def test_the_trainings_given_are_those_of_the_declaration_each_once(self):
        roots = DECLARATION_PROGRAM.training_roots(DECLARATION, ["control:0=a", "recovery:1=b/c"])
        self.assertEqual(roots, {("control", 0): Path("a"), ("recovery", 1): Path("b/c")})
        for items in (["control:2=a"], ["other:0=a"], ["control:0=a", "control:0=b"]):
            with self.assertRaises(ValueError, msg=items):
                DECLARATION_PROGRAM.training_roots(DECLARATION, items)


class PairTest(unittest.TestCase):
    def read(self, readings=None, configs=None, ends=None):
        keys = [(condition, seed) for condition in ("control", "recovery") for seed in (0, 1)]
        readings = readings if readings is not None else [reading(*key, updates) for key in keys
                                                          for updates in (1500, 3000)]
        trainings = {key: Path(f"/runs/{key[0]}_s{key[1]}") for key in keys}
        configs = {key: config(*key) for key in keys} | (configs or {})
        ends = {key: COMPLETED for key in keys} | (ends or {})
        return OUTCOME.read_pair(DECLARATION, readings, trainings, configs, ends, RECOVERY_OPTIONS)

    def test_readable_saves_and_trainings_give_the_outcome(self):
        result = self.read()
        self.assertEqual((result["outcome"]["outcome"], result["unreadable"]), ("tracking_cost_at_exposure", []))
        json.dumps(result, allow_nan=False)

    def test_each_unreadable_rule_gives_no_outcome(self):
        full = [reading(c, s, u) for c in ("control", "recovery") for s in (0, 1) for u in (1500, 3000)]
        cases = dict(
            missing_save=dict(readings=full[1:]),
            no_error=dict(readings=full[:-1] + [reading("recovery", 1, 3000, first_rows=dict(E_m=dict(value=None)))]),
            other_seed=dict(configs={("control", 0): config("control", 0, seed=1)}),
            other_updates=dict(configs={("recovery", 1): config("recovery", 1, iters=2000)}),
            options_of_the_other_condition=dict(configs={("recovery", 0): config("control", 0, out="x")}),
            other_card=dict(configs={("recovery", 1): config("recovery", 1, gpu="another card")}),
            other_revision=dict(configs={("recovery", 1): config("recovery", 1, git_commit="def")}),
            other_engine=dict(configs={("recovery", 1): config("recovery", 1, engine=engine("recovery", key="lib-2"))}),
            other_capacity=dict(configs={("control", 0): config("control", 0, engine=engine("control", maxcon=128))}),
            no_engine_record=dict(configs={("control", 0): config("control", 0, engine=None)}),
            other_end=dict(ends={("control", 1): dict(completed=False, cause="other", completed_updates=40,
                                                      detail="nonfinite_rows")}))
        for name, case in cases.items():
            result = self.read(**case)
            self.assertIsNone(result["outcome"]["outcome"], name)
            self.assertTrue(result["unreadable"], name)

    def test_the_engine_library_s_locations_may_differ_between_the_trainings_of_a_seed_and_its_identity_not(self):
        # config() puts each condition's library under its own job; the identity fields and capacities are equal.
        self.assertNotEqual(config("control", 0)["engine"]["path"], config("recovery", 0)["engine"]["path"])
        result = self.read()
        self.assertEqual((result["outcome"]["outcome"], result["unreadable"]), ("tracking_cost_at_exposure", []))
        for location, other in (("path", "artifact"), ("artifact", "path")):
            # Only this location differs: the other one is the control's.
            changed = engine("recovery", **{other: engine("control")[other]})
            result = self.read(configs={("recovery", 0): config("recovery", 0, engine=changed)})
            self.assertEqual(result["unreadable"], [], location)
        for field in ("key", "sources_sha256", "consts_sha256"):
            changed = engine("recovery", **{field: "other"})
            result = self.read(configs={("recovery", 0): config("recovery", 0, engine=changed)})
            self.assertEqual(result["unreadable"], [f"seed 0: the two config.json files differ in ['engine.{field}']"])

    def test_a_capacity_overflow_gives_run_stopped(self):
        for cause in ("capacity_overflow",):
            result = self.read(readings=[reading("control", 0, 1500)],
                               ends={("recovery", 1): dict(completed=False, cause=cause, completed_updates=1200,
                                                           detail="")})
            self.assertEqual(result["outcome"]["outcome"], "run_stopped")
            self.assertEqual(len(result["stopped"]), 1)

    def test_a_save_of_a_training_not_given_or_read_twice_is_refused(self):
        full = [reading(c, s, u) for c in ("control", "recovery") for s in (0, 1) for u in (1500, 3000)]
        stranger = reading("control", 0, 1500, save=dict(run="/runs/elsewhere"))
        for readings in (full + [stranger], full + [reading("control", 0, 1500)]):
            with self.assertRaises(ValueError):
                self.read(readings=readings)

    def test_the_declared_recovery_options_are_those_of_the_arguments_files(self):
        self.assertTrue(all(value is None for value in RECOVERY_OPTIONS["control"].values()))
        recovery = RECOVERY_OPTIONS["recovery"]
        self.assertEqual((len(recovery["recovery_hidden_clips"]), recovery["recovery_share"],
                          recovery["recovery_rows"]), (5, 0.5, [50, 375, 500]))
        self.assertEqual(sorted(recovery["recovery_hidden_clips"]), sorted(HIDDEN))


def failed(invalid):
    """A FAILED.json as torch_train writes it when an update finds ``invalid`` non-zero engine counters: the error
    names the invalid counters and the update's metrics, whose engine diagnostics hold every counter."""
    counters = ("fwd_unsupported_worlds", "fwd_contact_overflow_worlds", "fwd_row_overflow_worlds", "nonfinite_rows",
                "nonfinite_qpos_rows", "nonfinite_qvel_rows", "nonfinite_ctrl_rows", "nonfinite_cotangent_rows",
                "replay_mismatch_worlds")
    metrics = dict(actor_loss=float("nan"), engine_diagnostics={name: invalid.get(name, 0) for name in counters})
    error = RuntimeError("Invalid training condition: " + json.dumps(dict(invalid=invalid, metrics=metrics)))
    return dict(error=repr(error), completed_updates=812, training_s=1000.0)


class TrainingEndTest(unittest.TestCase):
    def test_how_a_training_ended_is_read_from_its_records(self):
        cases = (({"terminal.json": dict(status="completed", completed_updates=3000), "DONE": "completed\n"},
                  (True, None)),
                 ({"terminal.json": dict(status="completed", completed_updates=2100), "DONE": "completed\n"},
                  (False, "other")),
                 ({"terminal.json": dict(status="interrupted", completed_updates=2100)}, (False, "other")),
                 ({"FAILED.json": failed({"fwd_row_overflow_worlds": 3})}, (False, "capacity_overflow")),
                 # Both kinds of counter: the overflow ends the run as a capacity overflow.
                 ({"FAILED.json": failed({"fwd_contact_overflow_worlds": 1, "nonfinite_rows": 2})},
                  (False, "capacity_overflow")),
                 # The overflow counters are named at zero in the metrics, not among the invalid counters.
                 ({"FAILED.json": failed({"nonfinite_rows": 1})}, (False, "other")),
                 ({"FAILED.json": dict(error=repr(RuntimeError("Matmul precision changed")), completed_updates=0)},
                  (False, "other")),
                 ({}, (False, "other")))
        for files, expected in cases:
            with tempfile.TemporaryDirectory() as directory:
                for name, content in files.items():
                    (Path(directory)/name).write_text(content if isinstance(content, str) else json.dumps(content))
                end = OUTCOME.training_end(directory, 3000)
                self.assertEqual((end["completed"], end["cause"]), expected, files)


def log(recovery, updates=3000):
    """Log rows; a recovery log switches on odd updates, 3 times per row up to 1,500 updates and once after."""
    rows = []
    for update in range(1, updates + 1):
        row = dict(completed_updates=update, actor_loss=0.0)
        if recovery:
            row.update(recovery_tracking_window_rows_total=200_000*update,
                       recovery_switches=(3 if update <= 1500 else 1)*(update % 2),
                       recovery_switch_yaw_mean_rad=1.0 if update <= 1500 else 2.0,
                       recovery_window_row_share_tracking=0.7)
            if not update % 2:
                row["recovery_switch_yaw_mean_rad"] = None
        rows.append(row)
    return rows


class ExposureTest(unittest.TestCase):
    def test_the_tracking_rows_switch_yaw_and_matched_update_are_read_from_the_logs(self):
        logs = {("control", 0): (config("control", 0), log(False)), ("recovery", 0): (config("recovery", 0), log(True))}
        result = EXPOSURE.exposure(DECLARATION, logs)
        self.assertEqual(result["control seed 0"]["tracking_window_rows"],
                         {"1500": 1500*262_144, "3000": 3000*262_144})
        recovery = result["recovery seed 0"]
        self.assertEqual(recovery["tracking_window_rows"], {"1500": 300_000_000, "3000": 600_000_000})
        self.assertEqual(recovery["switch_yaw"]["1500"], dict(switches=2250, mean_rad=1.0))
        # Weighted by the switches: (2,250 x 1.0 + 750 x 2.0) / 3,000, not the mean 1.5 of the rows with switches.
        self.assertEqual(recovery["switch_yaw"]["3000"], dict(switches=3000, mean_rad=1.25))
        # The first control update with at least 600,000,000 rows at 262,144 per update.
        self.assertEqual(recovery["matched_control_update"], {"1500": 1145, "3000": 2289})

    def test_a_missing_log_row_or_a_log_of_the_other_condition_is_refused(self):
        for logs in ({("control", 0): (config("control", 0), log(False, updates=2999))},
                     {("control", 0): (config("control", 0), log(True))},
                     {("recovery", 0): (config("recovery", 0), log(False))}):
            with self.assertRaises(ValueError):
                EXPOSURE.exposure(DECLARATION, logs)


class ProgramTest(unittest.TestCase):
    """outcome.py and exposure.py run on run directories and save readings written into a temporary directory."""

    def test_the_outcome_and_exposure_programs_write_their_readings_and_refuse_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            arguments, saves = ["--pair", str(ROOT/"studies/recovery/pair.json")], []
            for condition in ("control", "recovery"):
                for seed in (0, 1):
                    root = directory/f"runs/{condition}_s{seed}"
                    root.mkdir(parents=True)
                    (root/"config.json").write_text(json.dumps(config(condition, seed)))
                    (root/"terminal.json").write_text(json.dumps(dict(status="completed", completed_updates=3000)))
                    (root/"DONE").write_text("completed\n")
                    (root/"log.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                          for row in log(condition == "recovery")))
                    arguments += ["--training", f"{condition}:{seed}={root}"]
                    for updates in (1500, 3000):
                        save = directory/f"readings/{condition}_s{seed}-{updates}"
                        save.mkdir(parents=True)
                        (save/"save_readings.json").write_text(json.dumps(reading(
                            condition, seed, updates, save=dict(run=str(root.resolve())))))
                        saves += ["--save", str(save)]
            for program, extra, written in ((OUTCOME, saves, "outcome.json"), (EXPOSURE, [], "exposure.json")):
                out = directory/written.split(".")[0]
                with contextlib.redirect_stdout(io.StringIO()):
                    program.main(arguments + extra + ["--out", str(out)])
                result = json.loads((out/written).read_text())
                with self.assertRaises(ValueError):
                    program.main(arguments + extra + ["--out", str(out)])
                if program is OUTCOME:
                    self.assertEqual((result["outcome"]["outcome"], result["unreadable"]),
                                     ("tracking_cost_at_exposure", []))
                else:
                    self.assertEqual(result["recovery seed 1"]["tracking_window_rows"]["3000"], 600_000_000)


if __name__ == "__main__":
    unittest.main()
