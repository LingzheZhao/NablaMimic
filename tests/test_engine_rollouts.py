"""Rollouts of a trained policy (engine_rollouts, RECOVERY.md) on a synthetic library with a stand-in engine forward;
no assets, simulator, GPU or stored outputs.

Library (clips in this order): getup (80 rows: rows 0-29 lying at pelvis height 0.2 m, pitched 90 degrees about y,
rows 30-79 upright at 0.55 m), sway (40 upright rows at 0.66 m), walk (300 upright rows at 0.78 m, moving along x; its
row 10 holds joint 0 at 1.5 rad, outside the soft range of +-1 rad) and tilted (20 rows pitched like the lying rows at
0.78 m). The model has 16 bodies, of which bodies 2 to 15 are tracked. The stand-in forward moves every joint halfway to
its setpoint, turns the root by 0.05 rad of yaw and keeps its position, except that a pelvis below 0.7 m sinks by
0.04 m per control and a state whose pelvis would be below 0.28 m turns non-finite; it reports contacts beyond capacity
for a pelvis below 0.6 m. A body placed at a sway row under a walk command therefore leaves the recipe's 0.25 m height
threshold at control 3 and turns non-finite at control 9; a body placed at a lying row turns non-finite at control 0;
an upright body under a tilted command leaves the recipe's 0.8 gravity-z threshold at control 0 at the same height."""

import importlib.util
import math
from types import SimpleNamespace
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch backend is not installed")

import numpy as np
import torch

from nablamimic import engine_rollouts as rollouts
from nablamimic.recovery import RecoverySettings
from nablamimic.torch_env import TorchTrackingEnv, quat_mul, quat_rotate
from nablamimic.torch_train import ResetSampler

CLIPS = (("getup", 80), ("sway", 40), ("walk", 300), ("tilted", 20))
SWAY, WALK, TILTED = 80, 120, 420  # first rows of sway, walk and tilted
LYING = (math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0)
BODIES = 16
TRACKED = np.arange(2, BODIES)
OFFSETS = torch.zeros((BODIES, 3))
OFFSETS[:, 0] = torch.arange(BODIES)*0.1
TURN = torch.tensor([math.cos(0.025), 0.0, 0.0, math.sin(0.025)])
MASS = 40.0
PER_TRIAL = ("command_row", "body_row", "push_control", "push_velocity_xy", "start_qpos", "start_qvel",
             "start_body_pos_w", "start_body_quat_w")


def kinematics(q, v, com):
    normalized = q[:, 3:7]/torch.linalg.vector_norm(q[:, 3:7], dim=-1, keepdim=True)
    q = torch.cat((q[:, :3], normalized, q[:, 7:]), -1)
    xpos = q[:, None, :3]+quat_rotate(OFFSETS[None], normalized[:, None])
    xquat = normalized[:, None].expand(-1, BODIES, -1)
    cvel = torch.zeros((len(q), BODIES, 6), dtype=q.dtype)
    return q, xpos, xquat, cvel, q[:, None, :3].expand(-1, BODIES, -1)


class SinkingForward:
    """A stand-in for torch_ffi.forward (module docstring); counts calls that received a non-finite state."""

    def __init__(self):
        self.nonfinite_inputs = 0

    def __call__(self, q, v, warm, ctrl, friction, com, *, iterations, substeps, **flags):
        assert flags == dict(tape=False, kinematics=True)
        self.nonfinite_inputs += int(not (torch.isfinite(q).all() and torch.isfinite(v).all()))
        height = torch.where(q[:, 2] < 0.7, q[:, 2]-0.04, q[:, 2])
        turned = quat_mul(TURN.expand(len(q), 4), q[:, 3:7])
        qn = torch.cat((q[:, :2], height[:, None], turned, 0.5*q[:, 7:]+0.5*ctrl), -1)
        broken = height < 0.28
        qn = torch.where(broken[:, None], float("nan"), qn)
        v = torch.where(broken[:, None], float("nan"), v)
        bits = torch.where(height < 0.6, rollouts.OVERFLOW_CONTACTS, 0)
        info = torch.stack((torch.full_like(bits, iterations), torch.full_like(bits, 8), torch.full_like(bits, 40),
                            bits), -1).to(torch.int32)
        return (qn, v, warm, info, *kinematics(qn, v, com))


def reference(**settings):
    qpos = []
    for name, rows in CLIPS:
        q = np.zeros((rows, 36), np.float32)
        q[:, 3] = 1
        if name == "getup":
            q[:, 2] = 0.55
            q[:30, 2] = 0.2
            q[:30, 3:7] = LYING
        elif name == "sway":
            q[:, 2] = 0.66
        elif name == "walk":
            q[:, 2] = 0.78
            q[:, 0] = 0.01*np.arange(rows)
        else:
            q[:, 2] = 0.78
            q[:, 3:7] = LYING
        qpos.append(q)
    qpos = np.concatenate(qpos)
    frames = len(qpos)
    qpos[:, 7:] = 0.05*np.sin(0.1*np.arange(frames)[:, None]+np.arange(29)[None])
    qpos[WALK+10, 7] = 1.5
    quaternion = torch.from_numpy(qpos[:, 3:7])
    positions = (torch.from_numpy(qpos[:, None, :3])
                 + quat_rotate(OFFSETS[TRACKED][None], quaternion[:, None])).numpy()
    starts = np.cumsum([0]+[rows for _, rows in CLIPS[:-1]])
    last = np.concatenate([np.full(rows, start+rows-1) for (_, rows), start in zip(CLIPS, starts)])
    cfg = SimpleNamespace(residual_action=False, action_rate_space="setpoint", solver_iterations=10, term_height=0.25,
                          term_tilt=0.8, **settings)
    return SimpleNamespace(
        m=SimpleNamespace(nq=36, nv=35, nu=29, body_subtreemass=np.array([MASS, 30.0])), cfg=cfg, nu=29,
        q0=np.zeros(29), action_scale=np.full(29, 0.5), ctrl_lo=np.full(29, -2), ctrl_hi=np.full(29, 2),
        soft_lo=np.full(29, -1), soft_hi=np.full(29, 1), body_ids=TRACKED, anchor_k=1, torso_id=TRACKED[1],
        root_com_id=0, foot_pairs=np.array([0, 1]), starts=starts, lengths=np.array([rows for _, rows in CLIPS]),
        names=[name for name, _ in CLIPS], ref_qpos=qpos, ref_qvel=np.zeros((frames, 35), np.float32),
        ref_body_pos=positions, ref_body_quat=np.repeat(qpos[:, None, 3:7], len(TRACKED), 1),
        ref_body_lin=np.zeros((frames, len(TRACKED), 3)), ref_body_ang=np.zeros((frames, len(TRACKED), 3)),
        ref_last=last,
        goal_feat=np.random.default_rng(0).normal(size=(frames, 68)).astype(np.float32))


def environment(step_fn, recovery=None, dr=False, obs_noise=False, reset_noise=True):
    return TorchTrackingEnv.from_reference(reference(dr=dr, obs_noise=obs_noise, reset_noise=reset_noise), device="cpu",
                                           step_fn=step_fn, kinematics_fn=kinematics, recovery=recovery)


def rollout_environment():
    """The environment as engine_rollouts builds it: no recovery setting, no randomisation or observation noise, reset
    noise from the scene, the tape-free step."""
    forward = SinkingForward()
    return environment(rollouts.TapeFreeStep(forward)), forward


def training_environment(recovery):
    """The environment as training builds it: randomisation, observation and reset noise."""
    return environment(SinkingForward(), recovery, dr=True, obs_noise=True)


def recovery_settings():
    return RecoverySettings(share=0.5, hidden_clips=("getup",), command_excluded_clips=("getup", "sway"),
                            low_pelvis_height=0.5, low_pelvis_up_z=0.5, upright_run_rows=10, longest_span_rows=60,
                            command_upright_rows=20, reward_scale=2.5, stop_controls=100, blend_controls=75)


def actor():
    """A one-layer actor whose every output is a sum of exact products (inputs times powers of two), so that its
    values do not depend on how a matrix product of a given batch size orders its sums: the latents read two goal
    entries, and each joint's action reads latent 0 and the joint's position in the newest history frame."""
    encoder, decoder = torch.zeros(680, 2), torch.zeros(2+930+6, 29)
    encoder[3, 0], encoder[100, 1] = 0.5, 0.25
    decoder[0] = 0.5
    decoder[2+9*93+torch.arange(29), torch.arange(29)] = -0.5
    return dict(enc=[(encoder, torch.zeros(2))], dec=[(decoder, torch.zeros(29))], anchor_dec=torch.zeros(6))


def run(trials, controls, termination="recipe"):
    env, forward = rollout_environment()
    records = rollouts.rollout(env, actor(), rollouts.trial_inputs({"trials": trials}, env, controls), termination)
    return records, forward


def same(a, b):
    return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def walk(row, **trial):
    return {"command_row": WALK+row, **trial}


def assert_start(test, records, state):
    """The records' placed state is ``state``."""
    tracked = torch.as_tensor(TRACKED)
    for key, value in (("start_qpos", state["qpos"]), ("start_qvel", state["qvel"]),
                       ("start_body_pos_w", state["xpos"][:, tracked]),
                       ("start_body_quat_w", state["xquat"][:, tracked]), ("frame_before", state["frame"])):
        recorded = records[key][0] if key == "frame_before" else records[key]
        test.assertTrue(np.array_equal(recorded, value.numpy()), key)


class PlacementTest(unittest.TestCase):
    def test_a_placed_start_is_the_body_of_a_recovery_training_start_at_the_same_rows_and_noise(self):
        settings = recovery_settings()
        training = training_environment(settings)
        sampler = ResetSampler(training, 500, .99)
        torch.manual_seed(3)
        started = sampler.sample(64)
        # The sampler's draws in their order (RECOVERY.md, "Starts").
        torch.manual_seed(3)
        frames = torch.multinomial(sampler.probs, 64, replacement=True).to(torch.int32)
        noise = training.sample_reset_noise(64)
        chosen = torch.rand(64) < settings.share
        start = training.start_frames[torch.randint(training.start_frames.shape[0], (64,))]
        command = training.command_frames[torch.randint(training.command_frames.shape[0], (64,))]
        self.assertTrue(bool(chosen.any() and (~chosen).any()))
        command_rows, body_rows = torch.where(chosen, command, frames), torch.where(chosen, start, frames)
        trials = [{"command_row": int(command_rows[i]), "body_row": int(body_rows[i]),
                   "reset_uniforms": {name: noise[name][i].tolist() for name in rollouts.UNIFORMS}} for i in range(64)]
        records, _ = run(trials, 1)
        assert_start(self, records, started)

    def test_a_twin_is_an_ordinary_reset_at_the_command_row_without_noise(self):
        training = training_environment(None)
        noise = training.sample_reset_noise(2)
        for name in rollouts.UNIFORMS:
            noise[name] = torch.zeros_like(noise[name])
        rows = torch.tensor([WALK+10, WALK+40], dtype=torch.int32)
        ordinary = training.reset(rows, noise)
        records, _ = run([walk(10), walk(40)], 1)
        assert_start(self, records, ordinary)
        # The root is the command row's; the joints are limited to the soft range as at every reset.
        twin = records["start_qpos"]
        reference_qpos = training.ref_qpos[rows.long()].numpy()
        self.assertTrue(np.array_equal(twin[:, :7], reference_qpos[:, :7]))
        self.assertEqual(float(twin[0, 7]), 1.0)
        self.assertTrue(np.array_equal(twin[0, 8:], reference_qpos[0, 8:]))
        self.assertTrue(np.array_equal(twin[1, 7:], reference_qpos[1, 7:]))

    def test_the_reset_uniforms_map_onto_the_reset_noise_by_value(self):
        uniforms = dict(position=[1.0, -1.0, 0.5], rotation=[0.0, 0.0, 1.0],
                        velocity=[1.0, 0.0, -1.0, 0.5, 0.0, 1.0], joints=[1.0, -0.5]+[0.0]*27)
        records, _ = run([walk(40, body_row=SWAY+5, reset_uniforms=uniforms)], 1)
        body = torch.from_numpy(reference().ref_qpos[SWAY+5])
        qpos, qvel = torch.from_numpy(records["start_qpos"][0]), torch.from_numpy(records["start_qvel"][0])
        torch.testing.assert_close(qpos[:3], body[:3]+torch.tensor([0.05, -0.05, 0.005]))
        torch.testing.assert_close(qpos[3:7], quat_mul(torch.tensor([math.cos(0.1), 0, 0, math.sin(0.1)]), body[3:7]))
        torch.testing.assert_close(qvel[:6], torch.tensor([0.5, 0.0, -0.2, 0.26, 0.0, 0.78]))
        torch.testing.assert_close(qpos[7:9], body[7:9]+torch.tensor([0.1, -0.05]))
        self.assertTrue(torch.equal(qpos[9:], body[9:]))
        self.assertEqual(int(records["command_row"][0]), WALK+40)
        self.assertEqual(int(records["body_row"][0]), SWAY+5)


class PushTest(unittest.TestCase):
    def test_a_push_changes_only_its_trial_s_planar_root_velocity_and_only_from_its_control(self):
        turned = dict(position=[0.0]*3, rotation=[0.0, 0.0, 1.0], velocity=[0.0]*6, joints=[0.0]*29)
        trials = [walk(0), walk(30, reset_uniforms=turned), walk(60, reset_uniforms=turned)]
        plain, _ = run(trials, 8)
        trials[1]["push"] = {"control": 3, "velocity_xy": [0.4, -0.2]}
        trials[2]["push"] = {"control": 5, "force": 100.0, "duration": 0.2, "direction": "left"}
        pushed, _ = run(trials, 8)
        # A world change is not turned by the pelvis yaw. 100 N for 0.2 s on 40 kg is 0.5 m/s to the left of the
        # pelvis, whose yaw at control 5 is the start's 0.2 rad and five turns of 0.05 rad.
        left = torch.tensor([-math.sin(0.45), math.cos(0.45)])*0.5
        torch.testing.assert_close(torch.from_numpy(pushed["push_velocity_xy"][2]), left)
        self.assertTrue(np.array_equal(pushed["push_velocity_xy"][:2], np.float32([[0.0, 0.0], [0.4, -0.2]])))
        self.assertEqual(pushed["push_control"].tolist(), [-1, 3, 5])
        applied = np.zeros((8, 3), bool)
        applied[3, 1] = applied[5, 2] = True
        self.assertTrue(np.array_equal(pushed["push_applied"], applied))
        change = np.zeros_like(plain["qvel"])
        change[3:, 1, :2] = [0.4, -0.2]
        change[5:, 2, :2] = left.numpy()
        np.testing.assert_allclose(pushed["qvel"]-plain["qvel"], change, atol=1e-6, rtol=0)
        self.assertTrue(np.array_equal(pushed["qvel"][3:, 1, :2]-plain["qvel"][3:, 1, :2],
                                       np.broadcast_to(np.float32([0.4, -0.2]), (5, 2))))
        self.assertTrue(np.array_equal(pushed["qvel"][:3], plain["qvel"][:3]))
        self.assertTrue(np.array_equal(pushed["qvel"][:, 0], plain["qvel"][:, 0]))
        self.assertTrue(np.array_equal(pushed["qvel"][:, :, 2:], plain["qvel"][:, :, 2:]))
        for key in plain:
            if key not in ("qvel", "push_applied", "push_control", "push_velocity_xy"):
                self.assertTrue(same(pushed[key], plain[key]), key)


class RecordTest(unittest.TestCase):
    def test_the_record_holds_each_field_in_its_shape_and_type(self):
        records, _ = run([walk(0), walk(40, body_row=SWAY+5, push={"control": 1, "velocity_xy": [0.5, 0.0]})], 4)
        float32, int32, bool_ = np.dtype(np.float32), np.dtype(np.int32), np.dtype(bool)
        expected = dict(
            body_pos_w=((4, 2, 14, 3), float32), body_quat_w=((4, 2, 14, 4), float32), qpos=((4, 2, 36), float32),
            qvel=((4, 2, 35), float32), frame_before=((4, 2), int32), frame_after=((4, 2), int32),
            clip_last_row=((4, 2), int32), available=((4, 2), bool_), done=((4, 2), bool_),
            termination_height=((4, 2), bool_), termination_tilt=((4, 2), bool_),
            termination_nonfinite=((4, 2), bool_), push_applied=((4, 2), bool_),
            engine_newton_iterations=((4, 2), int32), engine_contacts=((4, 2), int32),
            engine_constraint_rows=((4, 2), int32), engine_info_bits=((4, 2), int32),
            engine_contact_capacity_exceeded=((4, 2), bool_), engine_row_capacity_exceeded=((4, 2), bool_),
            command_row=((2,), int32), body_row=((2,), int32), push_control=((2,), int32),
            push_velocity_xy=((2, 2), float32), start_qpos=((2, 36), float32), start_qvel=((2, 35), float32),
            start_body_pos_w=((2, 14, 3), float32), start_body_quat_w=((2, 14, 4), float32))
        self.assertEqual({key: (value.shape, value.dtype) for key, value in records.items()}, expected)
        self.assertEqual(records["frame_before"][:, 0].tolist(), list(range(WALK, WALK+4)))
        self.assertEqual(records["frame_after"][:, 0].tolist(), list(range(WALK+1, WALK+5)))
        self.assertTrue((records["clip_last_row"] == WALK+299).all())
        # The tracked bodies (2 to 15) of the placed state: the walk row's root at the origin, unturned.
        offsets = 0.1*TRACKED.astype(np.float32)
        np.testing.assert_allclose(records["start_body_pos_w"][0, :, 0], offsets, atol=1e-6, rtol=0)
        np.testing.assert_allclose(records["start_body_quat_w"][0], np.tile([1.0, 0.0, 0.0, 0.0], (14, 1)), atol=1e-6)
        # After the control the state is the stepped one: the sway body sank from 0.66 to 0.62 m, and the walk body
        # turned by 0.05 rad.
        np.testing.assert_allclose(records["start_qpos"][1, 2], 0.66, atol=1e-6)
        np.testing.assert_allclose(records["qpos"][0, 1, 2], 0.62, atol=1e-6)
        np.testing.assert_allclose(records["body_pos_w"][0, 1, :, 2], np.full(14, 0.62), atol=1e-6)
        np.testing.assert_allclose(records["body_pos_w"][0, 0, :, 0], offsets*math.cos(0.05), atol=1e-6)
        np.testing.assert_allclose(records["body_quat_w"][0, 0], np.tile(TURN.numpy(), (14, 1)), atol=1e-6)

    def test_the_clip_end_holds_the_command_at_its_last_row_and_ends_no_trial(self):
        records, _ = run([walk(295)], 10)
        self.assertEqual(records["frame_after"][:, 0].tolist(), [WALK+296, WALK+297, WALK+298]+[WALK+299]*7)
        self.assertEqual(records["frame_before"][4:, 0].tolist(), [WALK+299]*6)
        self.assertTrue((records["clip_last_row"] == WALK+299).all())
        self.assertFalse(records["done"].any())
        self.assertTrue(records["available"].all())


class TerminationTest(unittest.TestCase):
    def test_the_two_terminations_give_identical_records_up_to_the_first_recipe_termination(self):
        trials = [walk(0), walk(40, body_row=SWAY+5)]
        recipe, _ = run(trials, 12, "recipe")
        nonfinite, _ = run(trials, 12, "nonfinite")
        first = 3
        self.assertEqual(np.flatnonzero(recipe["done"][:, 1]).tolist(), [first])
        self.assertTrue(bool(recipe["termination_height"][first, 1]))
        for key in recipe:
            if key in PER_TRIAL:
                self.assertTrue(same(recipe[key], nonfinite[key]), key)
                continue
            self.assertTrue(same(recipe[key][:first], nonfinite[key][:first]), key)
            self.assertTrue(same(recipe[key][:, 0], nonfinite[key][:, 0]), key)
            if key != "done":
                self.assertTrue(same(recipe[key][first], nonfinite[key][first]), key)
        # After the recipe's termination the trial is unavailable; without it the trial runs to its non-finite state.
        self.assertFalse(recipe["available"][first+1:, 1].any())
        self.assertTrue(np.isnan(recipe["qpos"][first+1:, 1]).all())
        self.assertTrue((recipe["frame_after"][first+1:, 1] == -1).all())
        self.assertEqual(np.flatnonzero(nonfinite["done"][:, 1]).tolist(), [9])
        # The ending control records the stepped state, non-finite here, not the finite state it started from.
        for key in ("qpos", "qvel", "body_pos_w", "body_quat_w"):
            self.assertTrue(np.isnan(nonfinite[key][9, 1]).all(), key)
            self.assertTrue(np.isfinite(nonfinite[key][8, 1]).all(), key)
        self.assertEqual(nonfinite["available"][:, 1].tolist(), [True]*9+[False]*3)
        self.assertTrue(bool(nonfinite["termination_nonfinite"][9, 1]))
        self.assertEqual(nonfinite["frame_after"][:9, 1].tolist(), list(range(WALK+41, WALK+50)))
        # The engine's report: contacts beyond capacity while the pelvis is below 0.6 m (from control 1).
        self.assertEqual(nonfinite["engine_contact_capacity_exceeded"][:9, 1].tolist(), [False]+[True]*8)
        self.assertFalse(nonfinite["engine_row_capacity_exceeded"].any())
        self.assertTrue((nonfinite["engine_contacts"][:, 0] == 8).all())
        self.assertTrue((nonfinite["engine_newton_iterations"][:, 0] == 10).all())

    def test_a_tilt_beyond_the_threshold_ends_a_trial_under_the_recipe_only(self):
        tilted = [{"command_row": TILTED, "body_row": WALK}]
        recipe, _ = run(tilted, 3, "recipe")
        self.assertEqual(recipe["done"][:, 0].tolist(), [True, False, False])
        self.assertTrue(bool(recipe["termination_tilt"][0, 0]))
        self.assertFalse(bool(recipe["termination_height"][0, 0]))
        self.assertEqual(recipe["available"][:, 0].tolist(), [True, False, False])
        nonfinite, _ = run(tilted, 3, "nonfinite")
        self.assertFalse(nonfinite["done"].any())
        self.assertTrue(nonfinite["available"].all() and nonfinite["termination_tilt"].all())
        self.assertFalse(nonfinite["termination_height"].any())

    def test_an_ended_trial_leaves_the_other_trials_records_bit_equal_to_a_batch_without_it(self):
        others = [walk(0), walk(70, reset_uniforms=dict(position=[0.2, -0.3, 0.1], rotation=[0.5, 0.0, -1.0],
                                                         velocity=[0.1]*6, joints=[0.3]*29))]
        for termination, ending in (("recipe", walk(40, body_row=SWAY+5)), ("nonfinite", walk(40, body_row=5))):
            with self.subTest(termination=termination):
                ending = {**ending, "push": {"control": 6, "velocity_xy": [1.0, 0.0]}}
                alone, _ = run(others, 10, termination)
                together, forward = run([others[0], ending, others[1]], 10, termination)
                for key in alone:
                    kept = together[key][[0, 2]] if key in PER_TRIAL else together[key][:, [0, 2]]
                    self.assertTrue(same(kept, alone[key]), key)
                end = 3 if termination == "recipe" else 0
                self.assertEqual(np.flatnonzero(together["done"][:, 1]).tolist(), [end])
                self.assertFalse(together["available"][end+1:, 1].any())
                # A push after the end is not applied, and no non-finite state reaches the engine again.
                self.assertFalse(together["push_applied"][:, 1].any())
                self.assertTrue(np.array_equal(together["push_velocity_xy"][1], [0.0, 0.0]))
                self.assertEqual(forward.nonfinite_inputs, 0)


class SceneTest(unittest.TestCase):
    def test_unknown_keys_and_invalid_values_in_a_scene_are_refused(self):
        env, _ = rollout_environment()
        uniforms = {name: [0.0]*size for name, size in rollouts.UNIFORMS.items()}
        cases = dict(
            scene_key={"trials": [walk(0)], "seed": 7},
            no_trials={"trials": []},
            trial_key={"trials": [walk(0, body=3)]},
            no_command={"trials": [{"body_row": 3}]},
            row_beyond={"trials": [{"command_row": 440}]},
            negative_row={"trials": [walk(0, body_row=-1)]},
            boolean_row={"trials": [{"command_row": True}]},
            uniform_key={"trials": [walk(0, reset_uniforms={**uniforms, "friction": [0.0]})]},
            uniform_missing={"trials": [walk(0, reset_uniforms={"position": [0.0]*3})]},
            uniform_beyond_one={"trials": [walk(0, reset_uniforms={**uniforms, "position": [1.5, 0.0, 0.0]})]},
            uniform_length={"trials": [walk(0, reset_uniforms={**uniforms, "joints": [0.0]*28})]},
            push_key={"trials": [walk(0, push={"control": 1, "velocity_xy": [1.0, 0.0], "frame": "pelvis"})]},
            push_both_forms={"trials": [walk(0, push={"control": 1, "velocity_xy": [1.0, 0.0], "force": 1.0,
                                                      "duration": 0.2, "direction": "left"})]},
            push_beyond_the_rollout={"trials": [walk(0, push={"control": 5, "velocity_xy": [1.0, 0.0]})]},
            push_direction={"trials": [walk(0, push={"control": 1, "force": 500.0, "duration": 0.2,
                                                     "direction": "up"})]},
            push_force={"trials": [walk(0, push={"control": 1, "force": -5.0, "duration": 0.2, "direction": "left"})]},
            push_velocity={"trials": [walk(0, push={"control": 1, "velocity_xy": [float("inf"), 0.0]})]})
        for name, scene in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                rollouts.trial_inputs(scene, env, 5)

    def test_rollouts_refuse_unknown_terminations_and_environments_other_than_the_rollout_environment(self):
        env, _ = rollout_environment()
        inputs = rollouts.trial_inputs({"trials": [walk(0)]}, env, 2)
        with self.assertRaisesRegex(ValueError, "termination 'fall'"):
            rollouts.rollout(env, actor(), inputs, "fall")
        # Each environment differs from the rollout environment in one property.
        settings = "reset noise on and domain randomisation and observation noise off"
        cases = dict(
            recovery=(environment(rollouts.TapeFreeStep(SinkingForward()), recovery_settings()), "recovery setting"),
            step_without_report=(environment(SinkingForward()), "tape-free engine step"),
            randomisation=(environment(rollouts.TapeFreeStep(SinkingForward()), dr=True), settings),
            observation_noise=(environment(rollouts.TapeFreeStep(SinkingForward()), obs_noise=True), settings),
            no_reset_noise=(environment(rollouts.TapeFreeStep(SinkingForward()), reset_noise=False), settings))
        for name, (wrong, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                rollouts.rollout(wrong, actor(), inputs, "recipe")

    def test_the_command_line_takes_the_run_checkpoint_weights_scene_termination_controls_and_output(self):
        args = rollouts.build_parser().parse_args(["--run", "RUN", "--iter", "3000", "--scene", "scene.json",
                                                   "--termination", "nonfinite", "--controls", "1000", "--out", "OUT"])
        self.assertEqual((str(args.run), args.iter, args.actor_key, str(args.scene), args.termination, args.controls,
                          str(args.out)), ("RUN", 3000, "actor_ema", "scene.json", "nonfinite", 1000, "OUT"))
        with self.assertRaises(SystemExit):
            rollouts.build_parser().parse_args(["--run", "RUN", "--iter", "1", "--scene", "s", "--termination",
                                                "fall", "--controls", "1", "--out", "OUT"])


if __name__ == "__main__":
    unittest.main()
