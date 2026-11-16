"""Recovery training from placed lying starts (RECOVERY.md) on a synthetic get-up library with a stand-in step that
holds or scripts the root; no assets, simulator, GPU or stored outputs.

Library (clips in this order): getup_a (80 rows, the hidden clip: rows 0-29 lying at pelvis height 0.2 m, pitched 90
degrees about y, rows 30-79 upright at 0.55 m), getup_b (40 rows, lying for 15 rows, then upright; neither hidden nor
a command clip) and walk (700 upright rows at 0.78 m, moving along x, the command clip). With 10 upright rows ending a
get-up, the get-up of getup_a ends at its row 39 and its start rows are rows 0-29. A body at the get-up's last pose is
within 0.25 m of the command's pelvis height; 0.05 m lower it is not."""

import importlib.util
import json
import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch backend is not installed")

import numpy as np
import torch

from nablamimic import torch_nets as nets
from nablamimic.params import read_params
from nablamimic.recovery import (BLEND, RECOVERY, TRACKING, RecoverySettings, get_up_last_rows, regime,
                                 upright_ahead, upright_run_lengths)
from nablamimic.torch_env import DT, TorchTrackingEnv, quat_rotate, select_reset
from nablamimic.torch_train import (ResetSampler, build_parser, make_control_body, make_window, recovery_metrics,
                                    validate_args)

CLIPS = (("getup_a", 80, 30), ("getup_b", 40, 15), ("walk", 700, 0))
WALK = 120  # first row of walk
GET_UP_LAST = 39
LYING_HEIGHT, RISEN_HEIGHT, WALK_HEIGHT = 0.2, 0.55, 0.78
LYING = (math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0)
BODIES = 14
OFFSETS = torch.zeros((BODIES, 3))
OFFSETS[:, 0] = torch.arange(BODIES)*0.1
# The keys of the environment without the recovery setting, written out here.
BASE_COMPACT_KEYS = ("qpos", "qvel", "warm", "frame", "ep", "hist", "prev_a", "friction", "enc_bias", "com")
BASE_STATE_KEYS = BASE_COMPACT_KEYS + ("xpos", "xquat", "cvel", "subtree_com")
BASE_OUTPUT_KEYS = {"cobs", "cobs_pre", "r", "term", "done", "v2", "frame", "terms", "hist", "goal", "a_pi"}
RECOVERY_KEYS = ("hidden_frame", "recovering", "error_controls", "blend_left", "command_offset")
CRITIC_ROOT_ERROR, CRITIC_ANCHOR_POSITION = slice(779, 783), slice(783, 786)  # after proprioception 93 and goal 686
CONFIGS = Path(__file__).resolve().parents[1]/"configs"


def settings(**changes):
    values = dict(share=0.5, hidden_clips=("getup_a",), command_excluded_clips=("getup_a", "getup_b"),
                  low_pelvis_height=0.5, low_pelvis_up_z=0.5, upright_run_rows=10, longest_span_rows=60,
                  command_upright_rows=20, reward_scale=2.5, stop_controls=100, blend_controls=75)
    values.update(changes)
    return RecoverySettings(**values)


def kinematics(q, v, com):
    normalized = q[:, 3:7]/torch.linalg.vector_norm(q[:, 3:7], dim=-1, keepdim=True)
    q = torch.cat((q[:, :3], normalized, q[:, 7:]), -1)
    xpos = q[:, None, :3]+quat_rotate(OFFSETS[None], normalized[:, None])
    xquat = normalized[:, None].expand(-1, BODIES, -1)
    cvel = torch.zeros((len(q), BODIES, 6), dtype=q.dtype)
    return q, xpos, xquat, cvel, q[:, None, :3].expand(-1, BODIES, -1)


class HeldBody:
    """A stand-in step: every joint moves halfway to its setpoint; the root pose is kept unless ``script(call)``
    returns root poses [B, 7] (NaN where a world keeps its own); ``nonfinite`` maps a call to the worlds whose state
    turns NaN there."""

    def __init__(self, script=None, nonfinite=None):
        self.script, self.nonfinite, self.calls = script, dict(nonfinite or {}), 0

    def __call__(self, q, v, warm, ctrl, friction, com, **kwargs):
        root = q[:, :7]
        pose = None if self.script is None else self.script(self.calls)
        if pose is not None:
            root = torch.where(torch.isnan(pose), root, pose)
        qn = torch.cat((root, 0.5*q[:, 7:]+0.5*ctrl), -1)
        if self.calls in self.nonfinite:
            broken = torch.zeros(len(q), dtype=torch.bool)
            broken[self.nonfinite[self.calls]] = True
            qn = torch.where(broken[:, None], float("nan"), qn)
            v = torch.where(broken[:, None], float("nan"), v)
        self.calls += 1
        return (qn, v, warm, *kinematics(qn, v, com))


def reference(*, residual=False, noise=False, names=True):
    qpos = []
    for name, rows, lying in CLIPS:
        q = np.zeros((rows, 36), np.float32)
        q[:, 3] = 1
        if name == "walk":
            q[:, 2] = WALK_HEIGHT
            q[:, 0] = 0.01*np.arange(rows)
        else:
            q[:, 2] = RISEN_HEIGHT
            q[:lying, 2] = LYING_HEIGHT
            q[:lying, 3:7] = LYING
        qpos.append(q)
    qpos = np.concatenate(qpos)
    frames = len(qpos)
    # Joint references that differ between rows, so that the residual action base tells rows apart.
    qpos[:, 7:] = 0.05*np.sin(0.1*np.arange(frames)[:, None]+np.arange(29)[None])
    quaternion = torch.from_numpy(qpos[:, 3:7])
    positions = (torch.from_numpy(qpos[:, None, :3])+quat_rotate(OFFSETS[None], quaternion[:, None])).numpy()
    starts = np.cumsum([0]+[rows for _, rows, _ in CLIPS[:-1]])
    last = np.concatenate([np.full(rows, start+rows-1) for (_, rows, _), start in zip(CLIPS, starts)])
    cfg = SimpleNamespace(residual_action=residual, action_rate_space="setpoint", solver_iterations=10, dr=noise,
                          obs_noise=noise, reset_noise=noise, term_height=0.25, term_tilt=0.8)
    prepared = SimpleNamespace(
        m=SimpleNamespace(nq=36, nv=35, nu=29), cfg=cfg, nu=29, q0=np.zeros(29), action_scale=np.full(29, 0.5),
        ctrl_lo=np.full(29, -2), ctrl_hi=np.full(29, 2), soft_lo=np.full(29, -1), soft_hi=np.full(29, 1),
        body_ids=np.arange(BODIES), anchor_k=1, torso_id=1, root_com_id=0, foot_pairs=np.array([0, 1]),
        starts=starts, lengths=np.array([rows for _, rows, _ in CLIPS]), ref_qpos=qpos,
        ref_qvel=np.zeros((frames, 35), np.float32), ref_body_pos=positions,
        ref_body_quat=np.repeat(qpos[:, None, 3:7], BODIES, 1), ref_body_lin=np.zeros((frames, BODIES, 3)),
        ref_body_ang=np.zeros((frames, BODIES, 3)), ref_last=last,
        goal_feat=np.random.default_rng(0).normal(size=(frames, 68)).astype(np.float32))
    if names:
        prepared.names = [name for name, _, _ in CLIPS]
    return prepared


def environment(recovery, *, residual=False, noise=False, script=None, nonfinite=None):
    step = HeldBody(script, nonfinite)
    return TorchTrackingEnv.from_reference(reference(residual=residual, noise=noise), device="cpu", step_fn=step,
                                           kinematics_fn=kinematics, recovery=recovery)


def recovery_start(env, hidden_rows, command_rows):
    """Recovery starts at rows of getup_a with commands at rows of walk (no reset noise)."""
    return env.reset(torch.tensor([WALK+row for row in command_rows]), body_frame=torch.tensor(hidden_rows))


def pose(env, rows, height=None, shift=None):
    """Root poses [B, 7] of library rows, with the pelvis height or a horizontal shift replaced."""
    poses = env.ref_qpos[torch.as_tensor(rows), :7].clone()
    if height is not None:
        poses[:, 2] = height
    if shift is not None:
        poses[:, :2] += torch.tensor(shift)
    return poses


def step(env, state, action=None):
    return env.step(state, torch.zeros((state["frame"].shape[0], env.nu)) if action is None else action)


def following(start_row, last_height=None, **changes):
    """A recovery start whose body follows the hidden rows exactly (at the get-up's last row at ``last_height``)."""
    holder = {}

    def script(call):
        row = min(start_row+call+1, GET_UP_LAST)
        return pose(holder["env"], [row], height=last_height if row == GET_UP_LAST else None)
    env = holder["env"] = environment(settings(**changes), script=script)
    return env, recovery_start(env, [start_row], [40])


def window_runner(env, horizon, critic_bias=0.0, term_penalty=None):
    """make_window on small random networks; the critic returns ``critic_bias`` everywhere."""
    generator = torch.Generator().manual_seed(0)
    actor = dict(enc=[(0.01*torch.randn(env.goal_dim-6, 2, generator=generator), torch.zeros(2))],
                 dec=[(0.01*torch.randn(2+env.hist_dim+6, env.nu, generator=generator), torch.zeros(env.nu))],
                 anchor_dec=torch.zeros(6))
    critic = [(torch.zeros(env.critic_dim, 1), torch.full((1,), critic_bias))]
    normalizer = (torch.zeros(env.critic_dim), torch.ones(env.critic_dim))

    def encode(parameters, frame, reset_frame):
        def sequence(f):
            frames = torch.minimum(f[None]+torch.arange(horizon)[:, None], env.ref_last[f][None])
            return nets.actor_encode(parameters, env.goal_ref(frames.flatten())).reshape(horizon, f.shape[0], -1)
        return sequence(frame), sequence(reset_frame)
    body = make_control_body(env, .99, 500, 3, term_penalty)
    window = make_window(env, actor, critic, normalizer, horizon, .99, body, encode)

    def run(start, reset):
        n = start["frame"].shape[0]
        action_noise = torch.zeros((horizon, n, env.nu))
        return window(start, reset, action_noise, torch.zeros((2, horizon, n, 93)), torch.zeros((horizon, n, 6)),
                      torch.zeros_like(action_noise, requires_grad=True), torch.zeros(2, requires_grad=True))
    return run, lambda frame, reset_frame: encode(actor, frame, reset_frame)


class RecoveryRowsTest(unittest.TestCase):
    def test_get_up_end_is_the_last_row_of_the_first_upright_run_inside_the_clip(self):
        upright = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
        # Two clips: rows 0-8 and rows 9-11.
        first = torch.tensor([0]*9+[9]*3)
        last = torch.tensor([8]*9+[11]*3)
        self.assertEqual(upright_run_lengths(upright, first).tolist(), [0, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(get_up_last_rows(upright, first, last, 3).tolist(), [7]*8+[-1, 11, 11, 11])
        # Rows 5 and 6 are upright with the next row; row 7's next row is low.
        self.assertEqual(upright_ahead(upright, first, last, 1).tolist(),
                         [False, False, True, False, False, True, True, False, False, True, False, False])

    def test_start_rows_come_from_the_hidden_clip_and_command_rows_have_upright_rows_ahead(self):
        env = environment(settings())
        self.assertEqual(env.start_frames.tolist(), list(range(30)))
        self.assertEqual(set(env.get_up_last[env.start_frames.long()].tolist()), {GET_UP_LAST})
        self.assertEqual(env.command_frames.tolist(), list(range(WALK, WALK+700-21)))
        self.assertTrue(bool((env.command_frames+20 < env.ref_last[env.command_frames.long()]).all()))
        self.assertEqual(env.recovery_counts, dict(low_rows_of_hidden_clips=30, without_a_get_up_end_in_the_clip=0,
                                                   beyond_the_longest_span=0, start_rows=30, get_ups=1,
                                                   command_rows=679))
        for table in (env.get_up_last, env.start_frames, env.command_frames):
            self.assertEqual(table.dtype, torch.int32)
        narrow = environment(settings(longest_span_rows=20))
        self.assertEqual(narrow.start_frames.tolist(), list(range(19, 30)))
        self.assertEqual(narrow.recovery_counts["beyond_the_longest_span"], 19)

    def test_unknown_clips_empty_tables_and_invalid_settings_are_refused(self):
        with self.assertRaisesRegex(ValueError, "getup_c"):
            environment(settings(hidden_clips=("getup_c",), command_excluded_clips=("getup_a", "getup_c")))
        with self.assertRaisesRegex(ValueError, "start rows and command rows"):
            environment(settings(command_excluded_clips=("getup_a", "getup_b", "walk")))
        with self.assertRaisesRegex(ValueError, "start rows and command rows"):
            environment(settings(hidden_clips=("getup_b",), longest_span_rows=1))
        for changes, message in ((dict(share=0.0), "share"), (dict(share=1.5), "share"),
                                 (dict(share=float("nan")), "share"),
                                 (dict(command_excluded_clips=("getup_b",)), "command-excluded"),
                                 (dict(upright_run_rows=0), "upright_run_rows"),
                                 (dict(blend_controls=0), "blend_controls"),
                                 (dict(reward_scale=0.0), "reward scale"),
                                 (dict(low_pelvis_height=float("inf")), "low-pelvis")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                settings(**changes)

    def test_the_regime_of_a_state_follows_its_flags(self):
        recovering = torch.tensor([True, False, False, True])
        left = torch.tensor([0, 5, 0, 3], dtype=torch.int32)
        self.assertEqual(regime(recovering, left).tolist(), [RECOVERY, BLEND, TRACKING, RECOVERY])


class RecoveryStartTest(unittest.TestCase):
    def test_a_share_of_the_resets_are_recovery_starts_placed_as_an_ordinary_reset(self):
        env = environment(settings(), noise=True)
        plain = environment(None, noise=True)
        sampler = ResetSampler(env, 500, .99)
        count = 4000
        torch.manual_seed(2)
        state = sampler.sample(count)
        # The draws in their order: rows, reset noise, share uniforms, start rows, command rows.
        torch.manual_seed(2)
        frames = torch.multinomial(sampler.probs, count, replacement=True).to(torch.int32)
        noise = env.sample_reset_noise(count)
        chosen = torch.rand(count) < 0.5
        start = env.start_frames[torch.randint(env.start_frames.shape[0], (count,))]
        command = env.command_frames[torch.randint(env.command_frames.shape[0], (count,))]
        self.assertTrue(torch.equal(state["recovering"], chosen))
        self.assertLess(abs(chosen.float().mean().item()-0.5), 0.03)
        self.assertTrue(torch.equal(state["hidden_frame"], torch.where(chosen, start, frames)))
        self.assertTrue(torch.equal(state["frame"], torch.where(chosen, command, frames)))
        self.assertTrue(set(state["hidden_frame"][chosen].tolist()) <= set(env.start_frames.tolist()))
        # The body of every start, recovery or not, is the ordinary reset at its body row with the same noise.
        placed = plain.reset(torch.where(chosen, start, frames), noise)
        for key in BASE_STATE_KEYS:
            if key != "frame":
                self.assertTrue(torch.equal(state[key], placed[key]), key)
        self.assertTrue(bool((state["ep"] == 0).all() and (state["error_controls"] == 0).all()))
        self.assertTrue(bool((state["blend_left"] == 0).all() and (state["command_offset"] == 0).all()))

    def test_without_the_options_nothing_more_is_drawn_and_the_keys_are_the_base_keys(self):
        env = environment(None, noise=True)
        sampler = ResetSampler(env, 500, .99)
        torch.manual_seed(5)
        state = sampler.sample(64)
        after = torch.get_rng_state()
        torch.manual_seed(5)
        frames = torch.multinomial(sampler.probs, 64, replacement=True).to(torch.int32)
        expected = env.reset(frames, env.sample_reset_noise(64))
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        self.assertEqual(tuple(state), BASE_STATE_KEYS)
        for key in BASE_STATE_KEYS:
            self.assertTrue(torch.equal(state[key], expected[key]), key)
        self.assertEqual(tuple(env.compact(state)), BASE_COMPACT_KEYS)
        self.assertEqual(tuple(env.restore(env.compact(state))), BASE_STATE_KEYS)
        self.assertEqual(tuple(step(env, state)), BASE_STATE_KEYS)
        run, _ = window_runner(env, 3, term_penalty=(2., .08, .10))
        records = run(state, state)[3]
        self.assertEqual(set(records), BASE_OUTPUT_KEYS)
        self.assertEqual(set(records["terms"]), {"root_pos", "root_ori", "body_pos", "body_ori", "body_lin_vel",
                                                 "body_ang_vel", "action_rate", "joint_limit", "term_pen"})


class CommandReadersTest(unittest.TestCase):
    """Only the reward, the termination penalty, the recovery stop and the critic read the hidden row."""

    def test_changing_only_the_hidden_fields_leaves_the_actor_inputs_and_setpoints_equal(self):
        for residual in (False, True):
            with self.subTest(residual_action=residual):
                env = environment(settings(), residual=residual)
                state = recovery_start(env, [3, 10], [40, 200])
                changed = {**state, "hidden_frame": state["hidden_frame"]+27,
                           "error_controls": state["error_controls"]+50,
                           "command_offset": state["command_offset"]+torch.tensor([1.5, -0.5])}
                blending = {**state, "recovering": torch.zeros(2, dtype=torch.bool),
                            "blend_left": torch.tensor([30, 75], dtype=torch.int32)}
                action = 0.1*torch.randn((2, env.nu), generator=torch.Generator().manual_seed(1))
                for other in (changed, blending):
                    for one, two in zip(env.policy_obs(state), env.policy_obs(other)):
                        self.assertTrue(torch.equal(one, two))
                    self.assertTrue(torch.equal(env.ctrl_of(action, state["frame"]),
                                                env.ctrl_of(action, other["frame"])))
                    self.assertFalse(torch.equal(env.critic_obs(state), env.critic_obs(other)))
                reward = env.reward(step(env, state, action), action, state["prev_a"], prev=state)[1]
                reward_changed = env.reward(step(env, changed, action), action, changed["prev_a"], prev=changed)[1]
                self.assertFalse(torch.equal(reward["body_pos"], reward_changed["body_pos"]))

    def test_changing_only_the_hidden_fields_leaves_a_window_s_regression_inputs_and_latents_equal(self):
        for residual in (False, True):
            with self.subTest(residual_action=residual):
                results = []
                for change in (False, True):
                    env = environment(settings(), residual=residual)
                    start = recovery_start(env, [3, 10], [40, 200])
                    if change:
                        start = {**start, "hidden_frame": start["hidden_frame"]+27,
                                 "error_controls": start["error_controls"]+50}
                    reset = recovery_start(env, [1, 2], [60, 70])
                    run, encode = window_runner(env, 5)
                    results.append((run(start, reset)[3], encode(start["frame"], reset["frame"])))
                (same, same_latents), (changed, changed_latents) = results
                self.assertFalse(bool(same["done"].any() or changed["done"].any()))
                for key in ("hist", "goal", "a_pi"):
                    self.assertTrue(torch.equal(same[key], changed[key]), key)
                for one, two in zip(same_latents, changed_latents):
                    self.assertTrue(torch.equal(one, two))
                self.assertFalse(torch.equal(same["r"], changed["r"]))


class ClockAndStopTest(unittest.TestCase):
    def test_the_command_clock_runs_on_and_the_hidden_clock_holds_at_the_get_up_end(self):
        env = environment(settings())
        state = recovery_start(env, [20], [40])
        for control in range(1, 31):
            state = step(env, state)
            self.assertEqual(state["frame"].item(), WALK+40+control)
            self.assertEqual(state["hidden_frame"].item(), min(20+control, GET_UP_LAST))
            self.assertTrue(state["recovering"].item())
        for key, dtype in zip(RECOVERY_KEYS, (torch.int32, torch.bool, torch.int32, torch.int32, torch.float32)):
            self.assertEqual(state[key].dtype, dtype, key)

    def test_one_hundred_consecutive_controls_of_target_error_stop_the_episode(self):
        env = environment(settings())
        state = recovery_start(env, [29], [40])
        for control in range(1, 101):
            stepped = step(env, state)
            self.assertEqual(stepped["error_controls"].item(), control)
            # The command's height term would end the episode at once; it is masked while recovering.
            self.assertGreater(env.termination_differences(stepped["qpos"], stepped["frame"])[0].item(), 0.25)
            self.assertEqual(bool(env.terminated(stepped).item()), control == 100)
            self.assertEqual(bool(env.recovery_outputs(state, stepped)["stop"].item()), control == 100)
            state = stepped

    def test_a_control_within_the_thresholds_resets_the_count_and_a_non_finite_state_ends_at_once(self):
        poses = {}
        env = environment(settings(), script=lambda call: poses.get(call), nonfinite={9: [1]})
        poses[7] = pose(env, [33, 33])
        poses[8] = pose(env, [0, 0])
        state = recovery_start(env, [25, 25], [40, 40])
        counts, ended = [], None
        for control in range(1, 11):
            stepped = step(env, state)
            counts.append(stepped["error_controls"][0].item())
            if env.terminated(stepped)[1]:
                self.assertFalse(env.recovery_outputs(state, stepped)["stop"][1].item())
                ended = control
            state = stepped
        # Targets: rows 26-29 lie (no error), rows 30-32 stand (errors 1-3), the body stands at row 33 (reset), then
        # lies at row 34 (error 1); world 1 turns non-finite at control 10.
        self.assertEqual(counts[:9], [0, 0, 0, 0, 1, 2, 3, 0, 1])
        self.assertEqual(ended, 10)

    def test_a_body_held_at_the_last_hidden_pose_outside_the_command_thresholds_is_stopped(self):
        holder = {}
        env = environment(settings(), script=lambda call: holder["pose"])
        holder["pose"] = pose(env, [GET_UP_LAST], height=RISEN_HEIGHT-0.05)
        state = recovery_start(env, [29], [40])
        stopped = None
        for control in range(1, 120):
            stepped = step(env, state)
            outputs = env.recovery_outputs(state, stepped)
            self.assertFalse(outputs["switched"].any())
            if outputs["stop"].item():
                stopped = control
                break
            state = stepped
        # The hidden row reaches its last row at control 10; errors count from there.
        self.assertEqual(stopped, 10+99)


class SwitchAndBlendTest(unittest.TestCase):
    def test_the_switch_comes_at_the_get_up_s_last_row_then_the_blend_and_the_offset(self):
        env, state = following(30)
        regimes, terms, switched_at = [], [], None
        action = torch.zeros((1, env.nu))
        for control in range(1, 100):
            stepped = step(env, state)
            outputs = env.recovery_outputs(state, stepped)
            regimes.append(outputs["regime"].item())
            terms.append(env.reward(stepped, action, state["prev_a"], prev=state)[1])
            self.assertFalse(env.terminated(stepped).any())
            if outputs["switched"].item():
                switched_at = control
                body = stepped["xpos"][0, env.torso_id, :2]
                command = env.ref_body_pos[stepped["frame"][0], env.anchor_k, :2]
                torch.testing.assert_close(stepped["command_offset"][0], body-command)
                self.assertAlmostEqual(outputs["switch_yaw"].item(), 0.0, places=6)
                # The switching control's reward reads the get-up's last row with the height-only anchor term at the
                # recovery scale; the critic of the state it reaches reads the command row with the offset.
                self.assertEqual(stepped["hidden_frame"].item(), GET_UP_LAST)
                self.assertAlmostEqual(terms[-1]["root_pos"].item(), 2.5*DT*0.5, places=7)
                critic = env.critic_obs(stepped)
                self.assertAlmostEqual(critic[0, CRITIC_ROOT_ERROR][0].item(), RISEN_HEIGHT-WALK_HEIGHT, places=6)
                torch.testing.assert_close(critic[0, CRITIC_ANCHOR_POSITION],
                                           torch.tensor([0., 0., WALK_HEIGHT-RISEN_HEIGHT]), atol=1e-6, rtol=0)
            state = stepped
        # Rows 31-39 are reached at controls 1-9; the switch is the last row's control.
        self.assertEqual(switched_at, 9)
        self.assertEqual(regimes, [RECOVERY]*9+[BLEND]*75+[TRACKING]*15)
        self.assertEqual(state["blend_left"].item(), 0)
        full = DT*1.0
        self.assertAlmostEqual(terms[9]["body_ori"].item()/(full/75), 1.0, places=5)
        self.assertAlmostEqual(terms[83]["body_ori"].item()/full, 1.0, places=5)

    def test_a_window_records_the_switch_control_as_recovery_and_the_next_as_blend(self):
        env, state = following(29)
        run, _ = window_runner(env, 12)
        records = run(state, env.reset(torch.tensor([WALK+9])))[3]
        self.assertFalse(records["done"].any())
        self.assertEqual(records["switched"][:, 0].tolist(), [False]*9+[True, False, False])
        self.assertEqual(records["regime"][:, 0].tolist(), [RECOVERY]*10+[BLEND]*2)

    def test_the_command_offset_carries_no_gradient_back_to_the_body_at_the_switch(self):
        env = environment(settings())
        state = recovery_start(env, [GET_UP_LAST-1], [40])
        shift = torch.tensor([[0.3, -0.1]], requires_grad=True)
        reached = env.ref_qpos[[GET_UP_LAST]].clone()
        qpos = torch.cat((reached[:, :2]+shift, reached[:, 2:]), -1)
        qvel = torch.zeros_like(state["qvel"])
        moved = kinematics(qpos, qvel, state["com"])
        result = env.finish_physics_kin(state, torch.zeros((1, env.nu)), (moved[0], qvel, qvel, *moved))
        self.assertFalse(result["recovering"].item())
        self.assertTrue(result["xpos"].requires_grad)
        self.assertFalse(result["command_offset"].requires_grad)
        torch.testing.assert_close(result["command_offset"], (result["xpos"][:, env.torso_id, :2]
                                   - env.ref_body_pos[result["frame"], env.anchor_k, :2]).detach())

    def test_a_switch_is_refused_while_the_command_differences_exceed_the_thresholds(self):
        env, state = following(30, last_height=RISEN_HEIGHT-0.05)
        for _ in range(20):
            stepped = step(env, state)
            self.assertFalse(env.recovery_outputs(state, stepped)["switched"].any())
            self.assertTrue(stepped["recovering"].all())
            state = stepped
        self.assertEqual(state["error_controls"].item(), 20-8)

    def test_the_offset_reaches_only_the_anchor_term_and_the_critic_s_anchor_position(self):
        env = environment(settings())
        state = env.reset(torch.tensor([WALK+40]))
        moved = {**state, "command_offset": torch.tensor([[0.3, -0.2]])}
        for one, two in zip(env.policy_obs(state), env.policy_obs(moved)):
            self.assertTrue(torch.equal(one, two))
        differing = set(torch.nonzero(env.critic_obs(state)[0] != env.critic_obs(moved)[0]).flatten().tolist())
        self.assertTrue(differing and differing <= set(range(783, 786)))
        action = torch.zeros((1, env.nu))
        terms = env.reward(step(env, state), action, state["prev_a"], prev=state)[1]
        terms_moved = env.reward(step(env, moved), action, moved["prev_a"], prev=moved)[1]
        self.assertEqual({name for name in terms if not torch.equal(terms[name], terms_moved[name])}, {"root_pos"})


class RecoveryRewardTest(unittest.TestCase):
    def test_the_recovery_reward_scales_the_tracking_terms_with_a_height_only_anchor_term(self):
        holder = {}
        env = environment(settings(), script=lambda call: holder["poses"])
        holder["poses"] = torch.cat((pose(env, [13]), pose(env, [13], shift=[1.0, -2.0]), pose(env, [13]),
                                     pose(env, [13])))
        # Worlds 0-2 recover from row 12 (world 1 shifted, world 2 with another command); world 3 tracks row 12.
        state = env.reset(torch.tensor([WALK+40, WALK+40, WALK+90, 12]), body_frame=torch.tensor([12]*4))
        self.assertEqual(state["recovering"].tolist(), [True, True, True, False])
        action = torch.zeros((4, env.nu))
        stepped = step(env, state)
        terms = env.reward(stepped, action, state["prev_a"], prev=state)[1]
        for name, value in terms.items():
            torch.testing.assert_close(value[1], value[0], rtol=1e-6, atol=1e-7)
            self.assertTrue(torch.equal(value[2], value[0]), name)
        for name in ("root_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel"):
            torch.testing.assert_close(terms[name][0], 2.5*terms[name][3], rtol=1e-6, atol=0)
        for name in ("action_rate", "joint_limit"):
            self.assertTrue(torch.equal(terms[name][0], terms[name][3]), name)
        hazard = env.hazard_features(stepped)
        self.assertTrue(torch.equal(hazard[0], hazard[3]))
        self.assertAlmostEqual(terms["root_pos"][0].item(), 2.5*DT*0.5, places=7)

    def test_a_recovery_control_s_penalty_and_action_change_are_the_recipe_s_at_their_rows(self):
        for residual in (False, True):
            with self.subTest(residual_action=residual):
                holder = {}
                env = environment(settings(), residual=residual, script=lambda call: holder["pose"])
                plain = environment(None, residual=residual)
                holder["pose"] = pose(env, [13], height=LYING_HEIGHT+0.15)
                generator = torch.Generator().manual_seed(3)
                state = recovery_start(env, [12], [40])
                state = {**state, "ep": torch.tensor([5], dtype=torch.int32),
                         "prev_a": 0.3*torch.randn((1, env.nu), generator=generator)}
                action = 0.3*torch.randn((1, env.nu), generator=generator)
                stepped = step(env, state, action)
                self.assertTrue(stepped["recovering"].all())
                at_hidden_row = {key: stepped[key] for key in BASE_STATE_KEYS}
                at_hidden_row["frame"] = torch.tensor([13], dtype=torch.int32)
                hazard = env.hazard_features(stepped)
                self.assertLess(hazard[0, 0].item(), 0.25-0.08)  # past the penalty's height start
                self.assertTrue(torch.equal(hazard, plain.hazard_features(at_hidden_row)))
                terms = env.reward(stepped, action, state["prev_a"], prev=state)[1]
                base_prev = {key: state[key] for key in BASE_STATE_KEYS}
                recipe = plain.reward(at_hidden_row, action, state["prev_a"], prev=base_prev)[1]
                self.assertNotEqual(recipe["action_rate"].item(), 0.0)
                for name in ("action_rate", "joint_limit"):
                    self.assertTrue(torch.equal(terms[name], recipe[name]), name)
                if residual:
                    hidden_prev = {**base_prev, "frame": torch.tensor([12], dtype=torch.int32)}
                    hidden_base = plain.reward(at_hidden_row, action, state["prev_a"], prev=hidden_prev)[1]
                    self.assertFalse(torch.equal(hidden_base["action_rate"], recipe["action_rate"]))


class ReplayAndWindowTest(unittest.TestCase):
    def test_a_recovery_state_keeps_its_fields_through_compact_restore_and_selection(self):
        env = environment(settings())
        state = recovery_start(env, [5, 9], [40, 41])
        state = {**state, "error_controls": torch.tensor([3, 0], dtype=torch.int32),
                 "command_offset": torch.tensor([[0.0, 0.0], [0.1, 0.2]])}
        compact = env.compact(state)
        self.assertEqual(tuple(compact), BASE_COMPACT_KEYS+RECOVERY_KEYS)
        restored = env.restore({key: value.clone() for key, value in compact.items()})
        self.assertEqual(tuple(restored), BASE_STATE_KEYS+RECOVERY_KEYS)
        for key in RECOVERY_KEYS:
            self.assertTrue(torch.equal(restored[key], state[key]), key)
            self.assertEqual(restored[key].dtype, state[key].dtype, key)
        other = env.reset(torch.tensor([WALK+7]*2))
        chosen = select_reset(torch.tensor([True, False]), restored, other)
        self.assertEqual(chosen["recovering"].tolist(), [True, False])
        self.assertEqual(chosen["hidden_frame"].tolist(), [5, WALK+7])
        for key in RECOVERY_KEYS:
            self.assertEqual(chosen[key].dtype, state[key].dtype, key)
        run, _ = window_runner(env, 3)
        records = run(restored, other)[3]
        self.assertTrue(bool((records["regime"] == RECOVERY).all()))
        continued = restored
        for _ in range(3):
            continued = step(env, continued)
        self.assertEqual(continued["hidden_frame"].tolist(), [5+3, 9+3])

    def test_a_recovery_stop_in_a_window_settles_without_a_bootstrap(self):
        env = environment(settings(stop_controls=3))
        run, _ = window_runner(env, 4, critic_bias=1.0)
        objective, _, _, records = run(recovery_start(env, [29], [40]), env.reset(torch.tensor([WALK+9])))
        self.assertEqual(records["stop"][:, 0].tolist(), [False, False, True, False])
        self.assertTrue(bool(records["term"][2, 0] and records["done"][2, 0] and not records["clip_end"][2, 0]))
        reward = records["r"][:, 0].double()
        # The critic returns 1 everywhere: no value after the stop, one after the last control.
        expected = reward[0]+.99*reward[1]+.99**2*reward[2]+reward[3]+.99*1.0
        self.assertAlmostEqual(-objective.item(), expected.item(), places=5)


class ResetBinsAndLogTest(unittest.TestCase):
    def test_recovery_and_blend_transitions_leave_the_reset_bins_as_they_are(self):
        env = environment(settings())
        masked, alone, untouched = (ResetSampler(env, 500, .99) for _ in range(3))
        cobs = torch.zeros((1, 2, env.critic_dim))
        rows = torch.tensor([[WALK+100, WALK+300]], dtype=torch.int32)
        targets = torch.tensor([[5.0, 7.0]])

        def normalizer():
            return torch.zeros(env.critic_dim), torch.ones(env.critic_dim)
        masked.update(dict(cobs=cobs, frame=rows, regime=torch.tensor([[TRACKING, RECOVERY]], dtype=torch.int8)),
                      targets, normalizer())
        alone.update(dict(cobs=cobs[:, :1], frame=rows[:, :1], regime=torch.tensor([[TRACKING]], dtype=torch.int8)),
                     targets[:, :1], normalizer())
        self.assertTrue(torch.equal(masked.bin_q, alone.bin_q) and torch.equal(masked.probs, alone.probs))
        statistic = untouched.bin_q[0].clone()
        untouched.update(dict(cobs=cobs, frame=rows, regime=torch.tensor([[BLEND, RECOVERY]], dtype=torch.int8)),
                         targets, normalizer())
        self.assertTrue(torch.equal(untouched.bin_q[0], statistic))

    def test_the_log_counts_clip_ends_and_episode_limits_of_recovery_controls_only(self):
        no = torch.zeros((2, 3), dtype=torch.bool)
        outs = dict(regime=torch.tensor([[TRACKING, BLEND, RECOVERY]]*2, dtype=torch.int8),
                    r=torch.tensor([[0.06, 0.1, 0.2], [0.03, 0.1, 0.2]]), switched=no, stop=no,
                    switch_yaw=torch.zeros((2, 3)), clip_end=torch.tensor([[True, False, False], [False]*3]),
                    limit=torch.tensor([[False]*3, [True]*3]), term=torch.tensor([[False, True, False], [False]*3]))
        metrics = recovery_metrics(outs, 0.12, 10)
        self.assertEqual((metrics["recovery_clip_ends"], metrics["recovery_episode_limits"],
                          metrics["recovery_terminations_in_blend"]), (0, 1, 1))
        self.assertAlmostEqual(sum(metrics[f"recovery_window_row_share_{name}"]
                                   for name in ("tracking", "recovery", "blend")), 1.0)
        self.assertEqual(metrics["recovery_tracking_window_rows_total"], 12)
        self.assertAlmostEqual(metrics["recovery_tracking_reward_mean"], 0.375)
        self.assertEqual((metrics["recovery_switches"], metrics["recovery_switch_yaw_mean_rad"],
                          metrics["recovery_switch_yaw_p90_rad"]), (0, None, None))
        json.dumps(metrics, allow_nan=False)
        switched = torch.tensor([[False, False, True], [False, False, True]])
        metrics = recovery_metrics({**outs, "regime": torch.full((2, 3), RECOVERY, dtype=torch.int8),
                                    "switched": switched, "switch_yaw": torch.tensor([[0., 0., .1], [0., 0., .3]]),
                                    "stop": torch.tensor([[True, False, False], [False]*3])}, 0.12, 0)
        self.assertEqual((metrics["recovery_switches"], metrics["recovery_stops"]), (2, 1))
        self.assertAlmostEqual(metrics["recovery_switch_yaw_mean_rad"], 0.2)
        self.assertAlmostEqual(metrics["recovery_switch_yaw_p90_rad"], 0.28)
        self.assertEqual((metrics["recovery_tracking_reward_mean"], metrics["recovery_tracking_window_rows_total"],
                          metrics["recovery_clip_ends"]), (None, 0, 1))
        json.dumps(metrics, allow_nan=False)


class OptionTest(unittest.TestCase):
    def parsed(self, **changes):
        args = build_parser().parse_args(read_params(CONFIGS/"recovery.args")+["--out", "unused"])
        return Namespace(**{**vars(args), **changes})

    def test_the_recovery_options_are_validated_together(self):
        validate_args(self.parsed())
        validate_args(self.parsed(recovery_share=1.0))
        validate_args(self.parsed(**{name: None for name in vars(self.parsed()) if name.startswith("recovery_")}))
        for changes, message in ((dict(recovery_controls=None), "go together"),
                                 (dict(recovery_share=0.0), "share"), (dict(recovery_share=1.5), "share"),
                                 (dict(residual_action=True), "absolute action base"),
                                 (dict(recovery_rows=[0, 375, 500]), "upright_run_rows"),
                                 (dict(recovery_controls=[100, 0]), "blend_controls"),
                                 (dict(recovery_reward_scale=0.0), "reward scale"),
                                 (dict(recovery_command_excluded_clips=["fallAndGetUp1_subject4"]),
                                  "command-excluded")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                validate_args(self.parsed(**changes))


if __name__ == "__main__":
    unittest.main()
