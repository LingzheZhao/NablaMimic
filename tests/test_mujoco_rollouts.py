"""Rollouts on MuJoCo C (mujoco_rollouts, ROLLOUTS.md "Rollouts on MuJoCo C") with the G1 model the trainer builds
(skipped without the Menagerie checkout) and a synthetic motion library; no NablaEngine, GPU or stored outputs.

Library (clips in this order): stand (60 rows, the knees-bent pose with small joint offsets, the lowest collision
geom 2 mm into the floor) and high (40 rows, the same poses with the pelvis at 2.5 m, clear of the floor)."""

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch backend is not installed")

import mujoco
import numpy as np
import torch

from nablamimic import engine_rollouts, mujoco_rollouts as mr
from nablamimic.env import ANCHOR, BM_BODIES, Q0, EnvConfig, build_model
from nablamimic.motion import SCENE
from nablamimic.torch_env import SUBSTEPS, TorchTrackingEnv, quat_mul

if not SCENE.exists():
    raise unittest.SkipTest(f"no Menagerie checkout at {SCENE} (scripts/setup_menagerie.sh)")

CLIPS = (("stand", 60), ("high", 40))
HIGH = 60  # first row of high
STEP = dict(iterations=10, substeps=SUBSTEPS, tape=False, kinematics=True)


def g1():
    """The model as TorchTrackingEnv holds it: env.build_model with sensors disabled."""
    m, kp = build_model(EnvConfig())
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    return m, kp


MODEL, KP = g1()
FOOT_PAIRS = np.array([i for i in range(MODEL.npair) if "_foot" in MODEL.pair(i).name
                       and MODEL.pair(i).name.endswith("_floor")])
Q_KNEES_BENT = np.array([next((value for key, value in Q0.items() if key in MODEL.joint(MODEL.actuator_trnid[i, 0]).name),
                           0.0) for i in range(MODEL.nu)])


def lowest(qpos):
    """The lowest point of the model's collision capsules and spheres for the positions ``qpos``."""
    data = mujoco.MjData(MODEL)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(MODEL, data)
    low = math.inf
    for geom in range(MODEL.ngeom):
        if not MODEL.geom(geom).name.endswith("_collision"):
            continue
        radius, half = MODEL.geom_size[geom, :2]
        if MODEL.geom_type[geom] == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            low = min(low, data.geom_xpos[geom, 2] - abs(data.geom_xmat[geom, 8])*half - radius)
        elif MODEL.geom_type[geom] == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            low = min(low, data.geom_xpos[geom, 2] - radius)
    return low


def standing(qpos):
    """``qpos`` with the pelvis height that puts the lowest collision geom 2 mm into the floor."""
    qpos = np.array(qpos, np.float64)
    qpos[2] = 0.0
    qpos[2] = -lowest(qpos) - 0.002
    return qpos


def physics(threads=1):
    return mr.MujocoPhysics(threads).bind(MODEL, FOOT_PAIRS)


def reference():
    rows = sum(count for _, count in CLIPS)
    qpos = np.zeros((rows, 36), np.float32)
    qpos[:, 3] = 1
    qpos[:, 7:] = Q_KNEES_BENT + 0.05*np.sin(0.1*np.arange(rows)[:, None] + np.arange(29)[None])
    qpos[:, 2] = [standing(row)[2] for row in qpos]
    qpos[HIGH:, 2] = 2.5
    ids = np.array([MODEL.body(name).id for name in BM_BODIES])
    data = mujoco.MjData(MODEL)
    positions, quaternions = np.empty((rows, 14, 3), np.float32), np.empty((rows, 14, 4), np.float32)
    for row in range(rows):
        data.qpos[:] = qpos[row]
        mujoco.mj_kinematics(MODEL, data)
        positions[row], quaternions[row] = data.xpos[ids], data.xquat[ids]
    starts = np.array([0, HIGH])
    last = np.concatenate([np.full(count, start + count - 1) for (_, count), start in zip(CLIPS, starts)])
    effort = np.array([MODEL.jnt_actfrcrange[MODEL.actuator_trnid[i, 0], 1] for i in range(MODEL.nu)])
    ranges = np.array([MODEL.jnt_range[MODEL.actuator_trnid[i, 0]] for i in range(MODEL.nu)])
    mid, half = ranges.mean(1), 0.5*(ranges[:, 1] - ranges[:, 0])*0.9
    cfg = SimpleNamespace(residual_action=False, action_rate_space="setpoint", solver_iterations=10, term_height=0.25,
                          term_tilt=0.8, dr=False, obs_noise=False, reset_noise=True)
    return SimpleNamespace(
        m=MODEL, cfg=cfg, nu=29, q0=Q_KNEES_BENT, action_scale=0.25*effort/KP, ctrl_lo=MODEL.actuator_ctrlrange[:, 0],
        ctrl_hi=MODEL.actuator_ctrlrange[:, 1], soft_lo=mid - half, soft_hi=mid + half, body_ids=ids,
        anchor_k=BM_BODIES.index(ANCHOR), torso_id=MODEL.body(ANCHOR).id,
        root_com_id=int(MODEL.body_rootid[MODEL.body("pelvis").id]), foot_pairs=FOOT_PAIRS, starts=starts,
        lengths=np.array([count for _, count in CLIPS]), names=[name for name, _ in CLIPS], ref_qpos=qpos,
        ref_qvel=np.zeros((rows, 35), np.float32), ref_body_pos=positions, ref_body_quat=quaternions,
        ref_body_lin=np.zeros((rows, 14, 3), np.float32), ref_body_ang=np.zeros((rows, 14, 3), np.float32),
        ref_last=last, goal_feat=np.random.default_rng(0).normal(size=(rows, 68)).astype(np.float32))


def environment(threads=1):
    """The rollout environment as build_mujoco_environment makes it, on the synthetic library."""
    mujoco_physics = mr.MujocoPhysics(threads)
    env = TorchTrackingEnv.from_reference(reference(), device="cpu",
                                          step_fn=engine_rollouts.TapeFreeStep(mujoco_physics.forward),
                                          kinematics_fn=mujoco_physics.kinematics)
    mujoco_physics.bind(env.m, env.foot_pairs)
    return env, mujoco_physics


def actor():
    """A one-layer actor: each joint's action reads latent 0 (two goal entries) and the joint's position in the
    newest history frame."""
    encoder, decoder = torch.zeros(680, 2), torch.zeros(2 + 930 + 6, 29)
    encoder[3, 0], encoder[100, 1] = 0.5, 0.25
    decoder[0] = 0.05
    decoder[2 + 9*93 + torch.arange(29), torch.arange(29)] = -0.5
    return dict(enc=[(encoder, torch.zeros(2))], dec=[(decoder, torch.zeros(29))], anchor_dec=torch.zeros(6))


def start(count=3):
    """Forward inputs of ``count`` trials: standing, standing turned and moving, and high in the air."""
    qpos = np.zeros((count, 36), np.float32)
    qpos[:, 3], qpos[:, 7:] = 1, Q_KNEES_BENT
    qpos[:, 2] = standing(qpos[0])[2]
    qpos[1, 3:7] = [math.cos(0.3), 0, 0, math.sin(0.3)]
    qpos[2, 2] = 2.5
    qvel = np.zeros((count, 35), np.float32)
    qvel[1, 0], qvel[2, 3:6] = 0.3, [0.5, -0.2, 0.1]
    ctrl = (Q_KNEES_BENT[None] + 0.1*np.arange(count)[:, None]).astype(np.float32)
    return [torch.from_numpy(x) for x in (qpos, qvel, np.zeros((count, 35), np.float32), ctrl,
                                          np.ones(count, np.float32), np.zeros((count, 3), np.float32))]


def seeded(q, v, w):
    data = mujoco.MjData(MODEL)
    mujoco.mj_resetData(MODEL, data)
    data.qpos[:], data.qvel[:], data.qacc_warmstart[:] = q, v, w
    return data


def run_controls(data, ctrl, controls):
    data.ctrl[:] = ctrl
    for _ in range(SUBSTEPS*controls):
        mujoco.mj_step(MODEL, data)
    return data


def next_inputs(outputs, inputs):
    """The inputs of the next control as TorchTrackingEnv passes them: qpos_n, qvel and the warm start."""
    return [outputs[4], outputs[1], outputs[2], *inputs[3:]]


def chain_state(data):
    return np.concatenate((data.qpos, data.qvel, data.qacc_warmstart))


class ModelTest(unittest.TestCase):
    def test_the_trainer_s_model_passes_and_collides_only_through_its_explicit_pairs(self):
        mr.check_model(MODEL, FOOT_PAIRS)
        self.assertEqual(mr.dynamic_collisions(MODEL), [])
        robot = MODEL.geom_bodyid != 0
        self.assertFalse(MODEL.geom_contype[robot].any() or MODEL.geom_conaffinity[robot].any())
        floor = MODEL.geom("floor").id
        self.assertTrue(MODEL.geom_contype[floor] and MODEL.geom_conaffinity[floor])
        record = mr.model_record(MODEL)
        json.dumps(record, allow_nan=False)
        self.assertEqual(len(record["geoms"]), MODEL.ngeom)
        self.assertEqual(len(record["pairs"]), MODEL.npair)
        self.assertIn("mjDSBL_SENSOR", record["disabled"])
        self.assertEqual(record["opt"]["iterations"], 10)
        self.assertEqual(MODEL.opt.integrator, int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST))
        self.assertEqual(MODEL.opt.cone, int(mujoco.mjtCone.mjCONE_PYRAMIDAL))
        self.assertEqual(MODEL.ntree, 1)
        # The island setting the description records is the model's flag.
        self.assertEqual(mr.constraint_islands(MODEL), "mjDSBL_ISLAND" not in record["disabled"])
        without = copy.deepcopy(MODEL)
        without.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
        self.assertFalse(mr.constraint_islands(without))
        self.assertEqual(record["opt"]["timestep"], 0.005)
        self.assertEqual([actuator["gain"] for actuator in record["actuators"]], KP.tolist())
        self.assertEqual([actuator["ctrllimited"] for actuator in record["actuators"]],
                         [bool(flag) for flag in MODEL.actuator_ctrllimited])
        self.assertEqual([actuator["actfrclimited"] for actuator in record["actuators"]],
                         [bool(MODEL.jnt_actfrclimited[MODEL.actuator_trnid[i, 0]]) for i in range(MODEL.nu)])
        self.assertEqual(record["collisions_outside_pairs"], [])

    def test_a_joint_order_other_solver_activation_or_extra_colliding_geom_is_refused(self):
        swapped = copy.deepcopy(MODEL)
        swapped.actuator_trnid[[0, 1]] = swapped.actuator_trnid[[1, 0]]
        with self.assertRaisesRegex(ValueError, "joint order"):
            mr.check_model(swapped, FOOT_PAIRS)
        solver = copy.deepcopy(MODEL)
        solver.opt.solver = int(mujoco.mjtSolver.mjSOL_CG)
        with self.assertRaisesRegex(ValueError, "not Newton"):
            mr.check_model(solver, FOOT_PAIRS)
        integrator = copy.deepcopy(MODEL)
        integrator.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
        with self.assertRaisesRegex(ValueError, "not implicitfast"):
            mr.check_model(integrator, FOOT_PAIRS)
        cone = copy.deepcopy(MODEL)
        cone.opt.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
        with self.assertRaisesRegex(ValueError, "not pyramidal"):
            mr.check_model(cone, FOOT_PAIRS)
        colliding = copy.deepcopy(MODEL)
        geom = MODEL.geom("left_foot1_collision").id
        colliding.geom_contype[geom] = colliding.geom_conaffinity[geom] = 1
        self.assertIn(("floor", "left_foot1_collision"), mr.dynamic_collisions(colliding))
        with self.assertRaisesRegex(ValueError, "outside the explicit pairs"):
            mr.check_model(colliding, FOOT_PAIRS)
        with self.assertRaisesRegex(ValueError, "foot pairs"):
            mr.check_model(MODEL, np.array([MODEL.npair]))


class ChainTest(unittest.TestCase):
    def test_controls_continue_one_float64_chain_per_trial(self):
        mujoco_physics, inputs = physics(), start()
        controls = 5
        for _ in range(controls):
            outputs = mujoco_physics.forward(*inputs, **STEP)
            # A kinematics call between controls (as a reset or restore makes) leaves the chains alone.
            mujoco_physics.kinematics(inputs[0], inputs[1], inputs[5])
            inputs = next_inputs(outputs, inputs)
        q, v, w, u = (x.numpy() for x in start()[:4])
        for trial in range(3):
            uninterrupted = run_controls(seeded(q[trial], v[trial], w[trial]), u[trial], controls)
            self.assertTrue(np.array_equal(chain_state(mujoco_physics.data[trial]), chain_state(uninterrupted)))
            self.assertTrue(np.array_equal(outputs[1][trial].numpy(), uninterrupted.qvel.astype(np.float32)))
        counters = mujoco_physics.read_diagnostics()
        self.assertEqual((counters["seeded_trial_controls"], counters["continued_trial_controls"],
                          counters["trial_controls"], counters["substeps"]), (3, 12, 15, 60))
        # Seeding every control from the float32 copies gives another chain.
        rounded = physics()
        inputs = start()
        for _ in range(controls):
            rounded.clear()
            inputs = next_inputs(rounded.forward(*inputs, **STEP), inputs)
        self.assertFalse(np.array_equal(inputs[1].numpy(), outputs[1].numpy()))

    def test_a_push_adds_its_change_in_float64_and_leaves_the_other_trials_alone(self):
        plain, pushed = physics(), physics()
        inputs_plain = inputs_pushed = start()
        handed_velocity = None
        for control in range(4):
            if control == 2:
                handed_velocity = inputs_pushed[1].numpy().copy()
                qvel = inputs_pushed[1].clone()
                qvel[1, :2] += torch.tensor([0.4, -0.2])
                inputs_pushed = [inputs_pushed[0], qvel, *inputs_pushed[2:]]
                expected = copy.copy(pushed.data[1])
                expected.qvel[:2] += qvel[1, :2].numpy().astype(np.float64) - handed_velocity[1, :2].astype(np.float64)
                run_controls(expected, inputs_pushed[3][1].numpy(), 2)
            out_plain, out_pushed = plain.forward(*inputs_plain, **STEP), pushed.forward(*inputs_pushed, **STEP)
            inputs_plain, inputs_pushed = next_inputs(out_plain, inputs_plain), next_inputs(out_pushed, inputs_pushed)
        self.assertTrue(np.array_equal(chain_state(pushed.data[1]), chain_state(expected)))
        for trial in (0, 2):
            self.assertTrue(np.array_equal(chain_state(pushed.data[trial]), chain_state(plain.data[trial])))
        self.assertFalse(np.array_equal(chain_state(pushed.data[1]), chain_state(plain.data[1])))
        self.assertEqual(pushed.read_diagnostics()["pushed_trial_controls"], 1)

    def test_a_state_other_than_the_handed_out_one_seeds_its_trial(self):
        mujoco_physics, inputs = physics(), start()
        inputs = next_inputs(mujoco_physics.forward(*inputs, **STEP), inputs)
        qpos = inputs[0].clone()
        qpos[2, 2] = 2.0
        inputs = [qpos, *inputs[1:]]
        mujoco_physics.forward(*inputs, **STEP)
        q, v, w, u = (x[2].numpy() for x in inputs[:4])
        self.assertTrue(np.array_equal(chain_state(mujoco_physics.data[2]), chain_state(run_controls(seeded(q, v, w),
                                                                                                     u, 1))))
        counters = mujoco_physics.read_diagnostics(reset=True)
        self.assertEqual((counters["seeded_trial_controls"], counters["continued_trial_controls"]), (4, 2))
        self.assertEqual(mujoco_physics.read_diagnostics()["calls"], 0)

    def test_the_outputs_are_the_bodies_of_the_positions_after_the_step_and_the_step_s_counts(self):
        mujoco_physics, inputs = physics(), start()
        outputs = [x.numpy() for x in mujoco_physics.forward(*inputs, **STEP)]
        for trial in range(3):
            data = mujoco_physics.data[trial]
            after = seeded(data.qpos, data.qvel, data.qacc_warmstart)
            mujoco.mj_kinematics(MODEL, after)
            mujoco.mj_comPos(MODEL, after)
            mujoco.mj_comVel(MODEL, after)
            for key, index in (("xpos", 5), ("xquat", 6), ("cvel", 7), ("subtree_com", 8)):
                self.assertTrue(np.array_equal(outputs[index][trial], getattr(after, key).astype(np.float32)), key)
            self.assertTrue(np.array_equal(outputs[0][trial], data.qpos.astype(np.float32)))
            quaternion = data.qpos[3:7]/np.linalg.norm(data.qpos[3:7])
            self.assertTrue(np.array_equal(outputs[4][trial, 3:7], quaternion.astype(np.float32)))
            self.assertTrue(np.array_equal(outputs[2][trial], data.qacc_warmstart.astype(np.float32)))
            self.assertEqual(outputs[3][trial].tolist(), [int(data.solver_niter[0]), data.ncon, data.nefc, 0])
            self.assertLessEqual(int(outputs[3][trial, 0]), 10)
        self.assertEqual(outputs[3].dtype, np.int32)
        # The trial in the air has no contacts; the standing ones touch the floor.
        self.assertEqual(int(outputs[3][2, 1]), 0)
        self.assertTrue((outputs[3][:2, 1] > 0).all())
        self.assertEqual(int(MODEL.body_weldid[0]), 0)

    def test_threads_do_not_change_the_results(self):
        results = []
        for threads in (1, 3):
            mujoco_physics, inputs = physics(threads), start()
            for _ in range(3):
                outputs = mujoco_physics.forward(*inputs, **STEP)
                inputs = next_inputs(outputs, inputs)
            results.append([x.numpy() for x in outputs])
        for first, second in zip(*results):
            self.assertTrue(np.array_equal(first, second))

    def test_a_diverging_trial_is_handed_out_as_non_finite_and_its_next_state_seeds_it(self):
        mujoco_physics, inputs = physics(), start()
        qvel = inputs[1].clone()
        qvel[1, 0] = 1e11
        outputs = mujoco_physics.forward(inputs[0], qvel, *inputs[2:], **STEP)
        self.assertTrue(all(torch.isnan(outputs[index][1]).all() for index in (0, 1, 2, 4, 5, 6, 7, 8)))
        self.assertTrue(all(torch.isfinite(outputs[index][[0, 2]]).all() for index in (0, 1, 2, 4, 5, 6, 7, 8)))
        # The next control seeds the trial from the state handed in; a second divergence is found again.
        stepped = mujoco_physics.forward(*inputs, **STEP)
        self.assertTrue(torch.isfinite(stepped[1][1]).all())
        again = mujoco_physics.forward(inputs[0], qvel, *inputs[2:], **STEP)
        self.assertTrue(torch.isnan(again[1][1]).all())
        self.assertEqual(mujoco_physics.read_diagnostics()["diverged_trial_controls"], 2)

    def test_mujoco_s_warnings_are_collected_by_text_from_every_thread(self):
        inputs = start()
        qvel = inputs[1].clone()
        qvel[:, 0] = 1e11
        with mr.collected_warnings() as log:
            physics(3).forward(inputs[0], qvel, *inputs[2:], **STEP)
        self.assertEqual(log.counts, {"Nan, Inf or huge value in QVEL at DOF 0. The simulation is unstable.": 3})
        # Concurrent calls lose no count.
        log = mr.WarningLog()
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _: [log(f"warning {i % 3} Time = {i}") for i in range(3000)], range(8)))
        self.assertEqual(log.counts, {f"warning {i}": 8000 for i in range(3)})

    def test_kinematics_normalise_the_root_quaternion_and_compute_the_bodies(self):
        mujoco_physics, inputs = physics(), start()
        qpos = inputs[0].clone()
        qpos[:, 3:7] *= 1.1
        out = [x.numpy() for x in mujoco_physics.kinematics(qpos, inputs[1], inputs[5])]
        np.testing.assert_allclose(np.linalg.norm(out[0][:, 3:7], axis=-1), 1.0, rtol=0, atol=1e-6)
        self.assertTrue(np.array_equal(out[0][:, 7:], qpos[:, 7:].numpy()))
        for trial in range(3):
            data = seeded(qpos[trial].numpy(), inputs[1][trial].numpy(), 0)
            mujoco.mj_kinematics(MODEL, data)
            mujoco.mj_comPos(MODEL, data)
            mujoco.mj_comVel(MODEL, data)
            for key, index in (("xpos", 1), ("xquat", 2), ("cvel", 3), ("subtree_com", 4)):
                self.assertTrue(np.array_equal(out[index][trial], getattr(data, key).astype(np.float32)), key)

    def test_settings_the_c_engine_cannot_step_are_refused(self):
        inputs = start()
        friction, com, doubled = inputs[4].clone(), inputs[5].clone(), [x.double() for x in inputs]
        friction[0], com[1, 0] = 0.8, 0.01
        cases = dict(friction=((*inputs[:4], friction, inputs[5]), STEP, "foot friction"),
                     com=((*inputs[:5], com), STEP, "centre-of-mass"),
                     tape=(inputs, {**STEP, "tape": True}, "forward only"),
                     no_kinematics=(inputs, {**STEP, "kinematics": False}, "forward only"),
                     iterations=(inputs, {**STEP, "iterations": 7}, "solver iterations"),
                     substeps=(inputs, {**STEP, "substeps": 3}, "substeps"),
                     float64=(doubled, STEP, "float32"))
        for name, (arguments, options, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                physics().forward(*arguments, **options)
        mujoco_physics = physics()
        mujoco_physics.forward(*inputs, **STEP)
        with self.assertRaisesRegex(ValueError, "batch of 2"):
            mujoco_physics.forward(*(x[:2] for x in inputs), **STEP)
        with self.assertRaisesRegex(ValueError, "not bound"):
            mr.MujocoPhysics(1).forward(*inputs, **STEP)
        with self.assertRaisesRegex(ValueError, "bound to a model already"):
            mujoco_physics.bind(MODEL, FOOT_PAIRS)
        with self.assertRaisesRegex(ValueError, "centre-of-mass"):
            physics().kinematics(inputs[0], inputs[1], com)
        with self.assertRaisesRegex(ValueError, "threads"):
            mr.MujocoPhysics(0)


class StandInForward:
    """A stand-in for torch_ffi.forward with the layout of its outputs: the state handed in unchanged, a zero info
    [B, 4] (int32) and the bodies of the positions from ``kinematics``."""

    def __init__(self, kinematics):
        self.kinematics = kinematics

    def __call__(self, qpos, qvel, warm, ctrl, friction, com, *, iterations, substeps, tape, kinematics):
        info = torch.zeros((len(qpos), 4), dtype=torch.int32)
        return (qpos, qvel, warm, info, *self.kinematics(qpos, qvel, com))


def layout(records):
    return {key: (value.shape, value.dtype) for key, value in records.items()}


def engine_description_keys():
    """The keys of the description engine_rollouts.main writes before its rollouts, and of its settings, read from the
    dict literal in its source, so that a change there changes what the description test expects."""
    tree = ast.parse(Path(engine_rollouts.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    for node in ast.walk(main):
        if isinstance(node, ast.Assign) and [getattr(target, "id", None) for target in node.targets] == ["description"]:
            keywords = {keyword.arg: keyword.value for keyword in node.value.keywords}
            return set(keywords), {keyword.arg for keyword in keywords["settings"].keywords}
    raise AssertionError("engine_rollouts.main assigns no description dict")


TRIALS = [{"command_row": 0}, {"command_row": 10, "push": {"control": 2, "velocity_xy": [0.3, 0.0]}},
          {"command_row": 5, "body_row": HIGH + 5},
          {"command_row": 20, "reset_uniforms": dict(position=[0.5, -0.5, 0.0], rotation=[0.0, 0.0, 1.0],
                                                     velocity=[0.2]*6, joints=[0.0]*29)}]


class RolloutTest(unittest.TestCase):
    def rollout(self, controls=6, termination="nonfinite"):
        env, mujoco_physics = environment()
        inputs = engine_rollouts.trial_inputs({"trials": TRIALS}, env, controls)
        return env, mujoco_physics, inputs, engine_rollouts.rollout(env, actor(), inputs, termination)

    def test_the_record_holds_the_engine_rollouts_fields_with_the_engine_s_report_from_mujoco(self):
        env, mujoco_physics, inputs, records = self.rollout()
        # The fields, shapes and types of engine_rollouts.rollout with a forward of the engine's output layout.
        kinematics = physics().kinematics
        stand_in_env = TorchTrackingEnv.from_reference(
            reference(), device="cpu", step_fn=engine_rollouts.TapeFreeStep(StandInForward(kinematics)),
            kinematics_fn=kinematics)
        stand_in_inputs = engine_rollouts.trial_inputs({"trials": TRIALS}, stand_in_env, 6)
        stand_in = engine_rollouts.rollout(stand_in_env, actor(), stand_in_inputs, "nonfinite")
        self.assertEqual(layout(records), layout(stand_in))
        self.assertEqual(records["qpos"].shape, (6, 4, 36))
        self.assertTrue(records["available"].all())
        self.assertFalse(records["engine_contact_capacity_exceeded"].any() or records["engine_row_capacity_exceeded"]
                         .any() or records["engine_info_bits"].any())
        self.assertTrue((records["engine_newton_iterations"] <= 10).all())
        # The trial placed in the air has no contacts in its first control; the standing ones have.
        self.assertEqual(int(records["engine_contacts"][0, 2]), 0)
        self.assertTrue((records["engine_contacts"][0, [0, 1, 3]] > 0).all())
        # The record's last state is the chains' state, as handed out.
        for trial in range(4):
            data = mujoco_physics.data[trial]
            self.assertTrue(np.array_equal(records["qvel"][-1, trial], data.qvel.astype(np.float32)))
        counters = mujoco_physics.read_diagnostics()
        self.assertEqual((counters["seeded_trial_controls"], counters["pushed_trial_controls"],
                          counters["continued_trial_controls"], counters["diverged_trial_controls"]), (4, 1, 19, 0))
        self.assertTrue(np.array_equal(records["push_applied"][:, 1], [False, False, True, False, False, False]))
        # The placed state is the MuJoCo C kinematics of the reset.
        placed = mr.start_fields(env, inputs)
        for key in mr.START_FIELDS:
            self.assertTrue(np.array_equal(records[key], placed[key]), key)

    def test_an_ended_trial_is_seeded_at_every_later_control_and_leaves_the_others_alone(self):
        _, unended_physics, _, unended = self.rollout(termination="nonfinite")
        _, mujoco_physics, _, records = self.rollout(termination="recipe")
        # The body placed in the air under a standing command is beyond the height threshold at control 0.
        self.assertEqual(np.flatnonzero(records["done"].any(0)).tolist(), [2])
        self.assertTrue(bool(records["done"][0, 2]))
        counters = mujoco_physics.read_diagnostics()
        self.assertEqual(counters["seeded_trial_controls"], 4 + 5)
        mr.check_chains(records, counters)
        mr.check_chains(unended, unended_physics.read_diagnostics())
        with self.assertRaisesRegex(RuntimeError, "seeded again"):
            mr.check_chains(records, {**counters, "seeded_trial_controls": 10})
        per_trial = {"command_row", "body_row", "push_control", "push_velocity_xy", *mr.START_FIELDS}
        for key in records:
            kept, other = ((records[key][[0, 1, 3]], unended[key][[0, 1, 3]]) if key in per_trial
                           else (records[key][:, [0, 1, 3]], unended[key][:, [0, 1, 3]]))
            self.assertTrue(np.array_equal(kept, other, equal_nan=kept.dtype.kind == "f"), key)

    def test_two_rollouts_of_the_same_trials_repeat(self):
        first, second = self.rollout()[3], self.rollout()[3]
        self.assertEqual(mr.repeat_differences(first, second), dict(identical=True, differing={}))
        changed = {**second, "qpos": second["qpos"] + 1e-3}
        report = mr.repeat_differences(first, changed)
        self.assertEqual(list(report["differing"]), ["qpos"])
        self.assertAlmostEqual(report["differing"]["qpos"]["largest"], 1e-3, places=6)
        # The comparison is of bytes: a sign of zero differs, with a value difference of 0.
        zero, signed = ({key: value.copy() for key, value in first.items()} for _ in range(2))
        zero["qvel"][0, 0, 0], signed["qvel"][0, 0, 0] = 0.0, -0.0
        self.assertEqual(mr.repeat_differences(zero, signed),
                         dict(identical=False, differing=dict(qvel=dict(largest=0.0, trial=0, finite_in_one_only=0))))
        missing = {key: value for key, value in first.items() if key != "done"}
        self.assertEqual(mr.repeat_differences(first, missing)["differing"], {"done": {"only_in": "first"}})

    def test_the_check_mode_runs_with_a_stand_in_for_the_training_engine(self):
        env, mujoco_physics, inputs, records = self.rollout()
        stand_in = mr.MujocoPhysics(1).bind(env.m, env.foot_pairs)
        result = mr.run_checks(env, mujoco_physics, actor(), inputs, "nonfinite", records, stand_in.forward, 2)
        json.dumps(result, allow_nan=False)
        self.assertTrue(all(value["largest"] == 0.0 for value in result["placement"].values()))
        lifted, placed = result["one_control"]["lifted"], result["one_control"]["placed"]
        self.assertEqual(lifted["lift_m"], mr.LIFT_M)
        self.assertEqual(lifted["mujoco_c_minus_engine"]["trials_without_contacts_at_last_substep"], 4)
        self.assertEqual(placed["mujoco_c_minus_engine"]["trials_without_contacts_at_last_substep"], 1)
        for step in (lifted, placed):
            for trial in step["per_trial"]:
                self.assertNotIn("most_contacts_in_a_substep", trial["engine"])
                for name in ("mujoco_c", "mujoco_c_again"):
                    self.assertGreaterEqual(trial[name]["most_contacts_in_a_substep"], trial[name]["contacts"])
        self.assertTrue(all(trial["mujoco_c"]["most_contacts_in_a_substep"] == 0 for trial in lifted["per_trial"]))
        for step in (lifted, placed):
            for label in ("mujoco_c_minus_engine", "engine_minus_engine", "mujoco_c_minus_mujoco_c"):
                self.assertTrue(all(value["largest"] == 0.0 for value in step[label]["all_trials"].values()), label)
        self.assertTrue(all(trial["engine"]["contacts"] == 0 for trial in lifted["per_trial"]))
        self.assertEqual(result["repeat"], dict(controls=6, termination="nonfinite", identical=True, differing={}))
        self.assertEqual(result["model"]["collisions_outside_pairs"], [])

    def test_the_check_mode_refuses_a_mujoco_c_record(self):
        env, mujoco_physics, inputs, records = self.rollout(controls=3)
        scene, library = {"trials": TRIALS}, {"MENAGERIE_DIR": None}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory/"rollouts.npz", **records)
            description = dict(scene_content=scene, termination="nonfinite", settings=dict(inputs=library,
                                                                                          clips=["all"]))
            for engine in (mujoco_physics.describe(), {"name": "MuJoCo"}, None):
                (directory/"description.json").write_text(json.dumps({**description, "engine": engine}))
                with self.subTest(engine=str(engine)[:20]), self.assertRaisesRegex(ValueError, "not a MuJoCo C"):
                    mr.training_engine_record(directory, scene, "nonfinite", library, ["all"])
            (directory/"description.json").write_text(json.dumps({**description, "engine": {"maxcon": 96}}))
            record = mr.training_engine_record(directory, scene, "nonfinite", library, ["all"])
            self.assertTrue(np.array_equal(record["start_qpos"], records["start_qpos"]))
            with self.assertRaisesRegex(ValueError, "termination"):
                mr.training_engine_record(directory, scene, "recipe", library, ["all"])
            with self.assertRaisesRegex(ValueError, "library inputs or clips"):
                mr.training_engine_record(directory, scene, "nonfinite", library, ["stand"])

    def test_the_placement_check_refuses_other_rows_and_another_body_order(self):
        env, _, inputs, records = self.rollout(controls=3)
        placed = mr.start_fields(env, inputs)
        swapped = {**records, "start_body_pos_w": records["start_body_pos_w"][:, [1, 0] + list(range(2, 14))]}
        with self.assertRaisesRegex(ValueError, "body order"):
            mr.placement_differences(swapped, placed, inputs)
        other = {**records, "body_row": records["body_row"] + 1}
        with self.assertRaisesRegex(ValueError, "body_row"):
            mr.placement_differences(other, placed, inputs)
        shifted = {**records, "start_qvel": records["start_qvel"] + np.float32(0.5)}
        self.assertAlmostEqual(mr.placement_differences(shifted, placed, inputs)["start_qvel"]["largest"], 0.5,
                               places=6)


class DescriptionTest(unittest.TestCase):
    def test_the_description_has_the_engine_rollouts_keys_and_names_the_engine(self):
        env, mujoco_physics = environment()
        args = mr.build_parser().parse_args(["--run", "RUN", "--iter", "3000", "--scene", "scene.json",
                                             "--termination", "nonfinite", "--controls", "1000", "--threads", "4",
                                             "--out", "OUT"])
        config = dict(git_commit="abc", engine="engine", gpu="card", clips=["all"], solver_iterations=10,
                      residual_action=False, action_rate_space="setpoint", matmul_precision="nets_tf32")
        description = mr.describe(args, config, Path("RUN/ckpt/it_003000.pkl"), dict(completed_updates=3000),
                                  {"trials": TRIALS}, "def", mujoco_physics, env, {"MENAGERIE_DIR": None},
                                  {"float32_matmul_precision": "highest"}, "card")
        # The keys engine_rollouts writes before the rollouts (engine_diagnostics is added after them).
        keys, settings = engine_description_keys()
        self.assertTrue({"engine", "scene_content", "termination", "controls", "settings"} <= keys)
        self.assertEqual(set(description), keys)
        self.assertEqual(set(description["settings"]), settings)
        engine = description["engine"]
        self.assertEqual((engine["name"], engine["mujoco_version"], engine["threads"]), ("MuJoCo", mujoco.__version__,
                                                                                         1))
        self.assertEqual((engine["ntree"], engine["constraint_islands"]), (1, mr.constraint_islands(MODEL)))
        self.assertEqual(engine["options"]["iterations"], 10)
        self.assertEqual(engine["differences"], list(mr.DIFFERENCES))
        json.dumps(description, allow_nan=False)

    def test_the_command_line_is_engine_rollouts_with_threads_and_checks(self):
        base = ["--run", "RUN", "--iter", "3000", "--scene", "scene.json", "--termination", "recipe", "--controls",
                "10", "--out", "OUT"]
        args = mr.build_parser().parse_args(base + ["--threads", "8", "--checks", "RECORD"])
        self.assertEqual((args.threads, str(args.checks), args.actor_key), (8, "RECORD", "actor_ema"))
        self.assertIsNone(mr.build_parser().parse_args(base + ["--threads", "8"]).checks)
        engine_options = {action.dest for action in engine_rollouts.build_parser()._actions}
        options = {action.dest for action in mr.build_parser()._actions}
        self.assertEqual(options - engine_options, {"threads", "checks"})
        with self.assertRaises(SystemExit):
            mr.build_parser().parse_args(base)


if __name__ == "__main__":
    unittest.main()
