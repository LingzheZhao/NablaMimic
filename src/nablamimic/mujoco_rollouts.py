"""Closed-loop rollouts of a trained policy with the physics stepped by MuJoCo's C engine (the mujoco Python package)
in place of NablaEngine. The trials, the environment, the policy, the terminations and the record are those of
engine_rollouts; only the functions that step the physics and compute the body kinematics differ. ROLLOUTS.md,
"Rollouts on MuJoCo C", gives the command, the set-up differences between the two engines and the check mode.

  python -m nablamimic.mujoco_rollouts --run RUN --iter 3000 --scene scene.json --termination nonfinite \\
      --controls 1000 --threads 16 --out OUT
  python -m nablamimic.mujoco_rollouts --run RUN --iter 3000 --scene scene.json --termination nonfinite \\
      --controls 1000 --threads 16 --checks RECORD --out OUT
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import pickle
import threading

import mujoco
import numpy as np
import torch

from . import torch_nets as nets
from .engine_rollouts import (LIBRARY_INPUTS, OVERFLOW_CONTACTS, OVERFLOW_ROWS, TapeFreeStep, build_parser as
                              engine_parser, rollout, trial_inputs)
from .torch_env import DT, SUBSTEPS, TorchTrackingEnv

# MuJoCo's warnings for a state it resets (mj_checkPos, mj_checkVel, mj_checkAcc) and for a full arena, which drops
# contacts or constraint rows; the latter map onto the engine's capacity bits.
DIVERGENCE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
CAPACITY = ((mujoco.mjtWarning.mjWARN_CONTACTFULL, OVERFLOW_CONTACTS), (mujoco.mjtWarning.mjWARN_CNSTRFULL,
                                                                       OVERFLOW_ROWS))
COUNTERS = ("calls", "trial_controls", "substeps", "seeded_trial_controls", "pushed_trial_controls",
            "continued_trial_controls", "diverged_trial_controls", "contact_capacity_trial_controls",
            "row_capacity_trial_controls")
CHECK_CONTROLS = 50  # controls of the two rollouts that check (d) compares
LIFT_M = 2.0  # height added to the pelvis in check (b), clear of the floor
# The set-up differences between the two engines that are not the engine itself (ROLLOUTS.md, "Rollouts on MuJoCo C").
DIFFERENCES = (
    "state: one float64 MjData per trial for the whole trial; the environment receives float32 copies, as NablaEngine "
    "hands out its float32 state. NablaEngine re-enters its float32 state at every control; the MuJoCo C chain takes "
    "the environment's state only at placement, at a push (the float64 difference of the float32 velocities on "
    "qvel[0:2]) and for ended trials, whose results are not used",
    "model: the MjModel the environment holds and NablaEngine is built from (env.build_model with the run's solver "
    "iterations, sensors disabled), unchanged; contacts from its explicit pairs alone, as no geom pair passes the "
    "contype / conaffinity filter",
    "solver: MuJoCo's Newton solver in float64 with the model's iterations, tolerance, line-search iterations and "
    "line-search tolerance, pyramidal cone and impratio, implicitfast integration; NablaEngine's model classification "
    "(nablaengine.model.fields) gives Newton in float32 with the model's iterations, tolerance (scaled by the mean "
    "inertia, as MuJoCo) and impratio, an exact line search of at most 40 steps that uses neither line-search option, "
    "and implicitfast integration (check (b) measures the step)",
    "constraint islands: MuJoCo C solves each constraint island on its own, with its own tolerance exit, unless "
    "mjDSBL_ISLAND is set (engine.constraint_islands); NablaEngine solves one problem per world; the model is one "
    "kinematic tree (another model is refused), so a substep has at most one island, and engine_newton_iterations is "
    "the largest count over the islands",
    "actuators: both engines apply the model's gain and bias and, where a joint is actfrclimited, its actuator force "
    "limit (jnt_actfrcrange; NablaEngine by its model classification); where an actuator is ctrllimited, MuJoCo C "
    "also clamps ctrl to the float64 ctrlrange, which NablaEngine does not (its documented semantics), after the "
    "environment has clipped ctrl to the float32 copy of that range (check (c) gives the flags; check (b) on lifted "
    "trials measures the actuators)",
    "capacity: NablaEngine holds a fixed number of contacts and constraint rows per world (its build's record, the "
    "engine field of an engine_rollouts description) and drops more (bits 4 and 16); MuJoCo C drops contacts or rows "
    "only when its arena is full (its warnings, mapped onto bits 4 and 16); no other bit is set, as MuJoCo C steps "
    "every feature of its model",
    "divergence: MuJoCo C resets a trial's data when qpos, qvel or qacc holds a value that is non-finite or above "
    "1e10 in magnitude; that control's state is handed out as non-finite (NaN) and counted",
    "randomisation: not modelled; the foot-pair friction must be the model's and the torso centre-of-mass offset zero",
)


class WarningLog:
    """MuJoCo's warning messages by their text before " Time =", with their counts; the worker threads that step the
    trials call it, so a lock guards the counts."""

    def __init__(self):
        self.counts = {}
        self.lock = threading.Lock()

    def __call__(self, message):
        text = message.split(" Time =")[0]
        with self.lock:
            self.counts[text] = self.counts.get(text, 0) + 1


@contextmanager
def collected_warnings():
    """MuJoCo's warnings collected in a WarningLog in place of its default handler, which prints them and appends
    them to MUJOCO_LOG.TXT in the working directory."""
    log = WarningLog()
    mujoco.set_mju_user_warning(log)
    try:
        yield log
    finally:
        mujoco.set_mju_user_warning(None)


def dynamic_collisions(m):
    """The geom pairs (names) that MuJoCo's contype / conaffinity filter lets collide besides the explicit pairs: two
    geoms on different bodies, not both on the world, with (contype1 & conaffinity2) | (contype2 & conaffinity1)
    nonzero. Parent-child and excluded body pairs are not removed, so the list holds every candidate."""
    contype, conaffinity, body = m.geom_contype, m.geom_conaffinity, m.geom_bodyid
    static = m.body_weldid[body] == 0
    found = []
    for first in range(m.ngeom):
        for second in range(first + 1, m.ngeom):
            if body[first] == body[second] or (static[first] and static[second]):
                continue
            if (contype[first] & conaffinity[second]) or (contype[second] & conaffinity[first]):
                found.append((m.geom(first).name, m.geom(second).name))
    return found


def check_model(m, foot_pairs):
    """Raise unless ``m`` is a model this program steps as the environment reads it: no actuator activations (the
    interface carries qpos, qvel and the warm start alone), a free root joint at qpos 0:7, actuator i driving the hinge
    joint at qpos 7 + i and qvel 6 + i (the order of actions, the reference and the record), one kinematic tree (at most
    one constraint island), the Newton solver (the record's engine_newton_iterations), the implicitfast integrator and
    the pyramidal cone (the only ones NablaEngine steps), foot pairs among the pairs, and no geom pair outside the
    explicit pairs."""
    if m.na:
        raise ValueError(f"the model has {m.na} actuator activations; the step's interface carries none")
    free, hinge = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE)
    if m.njnt < 1 or m.jnt_type[0] != free or m.jnt_qposadr[0] != 0 or m.jnt_dofadr[0] != 0:
        raise ValueError("joint 0 of the model is not the free root joint at qpos 0:7")
    for index in range(m.nu):
        joint = int(m.actuator_trnid[index, 0])
        if (m.actuator_trntype[index] != int(mujoco.mjtTrn.mjTRN_JOINT) or m.jnt_type[joint] != hinge
                or m.jnt_qposadr[joint] != 7 + index or m.jnt_dofadr[joint] != 6 + index):
            raise ValueError(f"actuator {index} ({m.actuator(index).name}) does not drive the hinge joint at qpos "
                             f"{7 + index}, qvel {6 + index}: the joint order differs from the actuator order")
    if m.ntree != 1:
        raise ValueError(f"the model has {m.ntree} kinematic trees, not the robot's one")
    if m.opt.solver != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise ValueError(f"the model's solver is {mujoco.mjtSolver(int(m.opt.solver)).name}, not Newton")
    if m.opt.integrator != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError(f"the model's integrator is {mujoco.mjtIntegrator(int(m.opt.integrator)).name}, not "
                         f"implicitfast")
    if m.opt.cone != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError(f"the model's friction cone is {mujoco.mjtCone(int(m.opt.cone)).name}, not pyramidal")
    pairs = np.asarray(foot_pairs)
    if pairs.ndim != 1 or not len(pairs) or pairs.min() < 0 or pairs.max() >= m.npair:
        raise ValueError(f"foot pairs {pairs.tolist()} are not pairs of the model ({m.npair})")
    extra = dynamic_collisions(m)
    if extra:
        raise ValueError(f"{len(extra)} geom pairs collide outside the explicit pairs, e.g. {extra[:5]}")


def constraint_islands(m):
    """Whether MuJoCo C splits the model's constraint problem into islands (mjDSBL_ISLAND not set)."""
    return not int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ISLAND)


def _float32(tensor, what):
    if tensor.dtype != torch.float32:
        raise ValueError(f"{what} must be float32, got {tensor.dtype}")
    return tensor.detach().to("cpu").numpy()


def _normalised(qpos):
    """qpos (float64) with the root quaternion divided by its norm."""
    out = np.array(qpos, dtype=np.float64)
    out[3:7] /= np.linalg.norm(out[3:7])
    return out


class MujocoPhysics:
    """The step and kinematics functions of TorchTrackingEnv (``forward`` wrapped in engine_rollouts.TapeFreeStep, and
    ``kinematics``) on MuJoCo C. Each trial of the batch keeps one MjData (float64) for the whole trial; a call
    continues it when the state handed in is the float32 copy the last call handed out, adds the change of a push
    (only qvel[0:2] differs) in float64, and otherwise seeds it from the state handed in (placement; an ended trial,
    whose state is held). Trials are stepped in ``threads`` threads (mj_step releases the interpreter lock). Bound to
    the environment's model by ``bind`` once the environment exists; ``clear`` drops the trials' data.
    ``most_contacts`` [B] holds, per trial of the last call, the largest contact count over its substeps."""

    def __init__(self, threads):
        if type(threads) is not int or threads < 1:
            raise ValueError(f"threads must be a positive integer, got {threads!r}")
        self.threads = threads
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.m = None
        self.data, self.handed, self.most_contacts = None, None, None
        self.counters = dict.fromkeys(COUNTERS, 0)

    def bind(self, model, foot_pairs):
        if self.m is not None:
            raise ValueError("the physics is bound to a model already")
        check_model(model, foot_pairs)
        self.m = model
        self.foot_pairs = np.asarray(foot_pairs).copy()
        self.foot_friction = model.pair_friction[self.foot_pairs, :2].copy()
        self.scratch = [mujoco.MjData(model) for _ in range(self.threads)]
        return self

    def clear(self):
        self.data, self.handed = None, None

    def read_diagnostics(self, reset=False):
        counters = dict(self.counters)
        if reset:
            self.counters = dict.fromkeys(COUNTERS, 0)
        return counters

    def describe(self):
        options = model_options(self._model())
        return dict(name="MuJoCo", implementation="the C engine through the mujoco Python package",
                    mujoco_version=mujoco.__version__, threads=self.threads, state_precision="float64",
                    handed_out_precision="float32",
                    options={key: options["opt"][key] for key in ("timestep", "integrator", "cone", "solver",
                                                                 "iterations", "tolerance", "ls_iterations",
                                                                 "ls_tolerance", "impratio")},
                    disabled=options["disabled"], enabled=options["enabled"], npair=int(self.m.npair),
                    ntree=int(self.m.ntree), constraint_islands=constraint_islands(self.m),
                    differences=list(DIFFERENCES))

    def _model(self):
        if self.m is None:
            raise ValueError("the physics is not bound to the environment's model (bind)")
        return self.m

    def _map(self, work, count):
        chunks = [chunk for chunk in np.array_split(np.arange(count), self.threads) if len(chunk)]
        if self.pool is None:
            for slot, chunk in enumerate(chunks):
                work(slot, chunk)
        else:
            list(self.pool.map(work, range(len(chunks)), chunks))

    def kinematics(self, qpos, qvel, com):
        """``torch_ffi.kinematics``: the positions with the root quaternion normalised, and the bodies' world positions,
        orientations, com-based velocities and subtree centres of mass, from mj_kinematics, mj_comPos and mj_comVel."""
        m = self._model()
        q, v, c = _float32(qpos, "qpos"), _float32(qvel, "qvel"), _float32(com, "com")
        if np.any(c != 0):
            raise ValueError("a torso centre-of-mass offset is not modelled on MuJoCo C")
        count = len(q)
        out = dict(qpos=np.empty((count, m.nq), np.float32), xpos=np.empty((count, m.nbody, 3), np.float32),
                   xquat=np.empty((count, m.nbody, 4), np.float32), cvel=np.empty((count, m.nbody, 6), np.float32),
                   subtree_com=np.empty((count, m.nbody, 3), np.float32))

        def work(slot, trials):
            data = self.scratch[slot]
            for trial in trials:
                data.qpos[:], data.qvel[:] = q[trial], v[trial]
                self._bodies(data, out, trial)
                out["qpos"][trial] = _normalised(data.qpos)
        self._map(work, count)
        return tuple(torch.from_numpy(out[key]).to(qpos.device) for key in ("qpos", "xpos", "xquat", "cvel",
                                                                             "subtree_com"))

    def _bodies(self, data, out, trial):
        mujoco.mj_kinematics(self.m, data)
        mujoco.mj_comPos(self.m, data)
        mujoco.mj_comVel(self.m, data)
        for key in ("xpos", "xquat", "cvel", "subtree_com"):
            out[key][trial] = getattr(data, key)

    def forward(self, qpos, qvel, warm, ctrl, friction, com, *, iterations, substeps, tape, kinematics):
        """``torch_ffi.forward`` without a tape: (qpos, qvel, warm start, info [B, 4], qpos with the root quaternion
        normalised, xpos, xquat, cvel, subtree_com) after ``substeps`` mj_step calls per trial. info holds the solver
        iterations, contacts (ncon) and constraint rows (nefc) of the last substep and the capacity bits of any
        substep. The body kinematics are those of the positions after the last substep."""
        m = self._model()
        if tape or not kinematics:
            raise ValueError("MuJoCo C steps forward only, with the body kinematics (tape=False, kinematics=True)")
        if iterations != m.opt.iterations:
            raise ValueError(f"solver iterations {iterations} differ from the model's {m.opt.iterations}")
        if type(substeps) is not int or substeps < 1 or not math.isclose(substeps*m.opt.timestep, DT, abs_tol=1e-12):
            raise ValueError(f"{substeps} substeps of {m.opt.timestep} s are not one control of {DT} s")
        q, v, w, u = (_float32(value, name) for value, name in ((qpos, "qpos"), (qvel, "qvel"), (warm, "warm"),
                                                                 (ctrl, "ctrl")))
        f, c = _float32(friction, "friction"), _float32(com, "com")
        if not np.all(f[:, None, None] == self.foot_friction[None]):
            raise ValueError("a foot friction other than the model's foot-pair friction is not modelled on MuJoCo C")
        if np.any(c != 0):
            raise ValueError("a torso centre-of-mass offset is not modelled on MuJoCo C")
        count = len(q)
        if self.data is None:
            self.data = [mujoco.MjData(m) for _ in range(count)]
        elif len(self.data) != count:
            raise ValueError(f"a batch of {count} trials after one of {len(self.data)}; clear() between batches")
        out = dict(raw=np.empty((count, m.nq), np.float32), qpos=np.empty((count, m.nq), np.float32),
                   qvel=np.empty((count, m.nv), np.float32), warm=np.empty((count, m.nv), np.float32),
                   info=np.zeros((count, 4), np.int32), most_contacts=np.zeros(count, np.int32),
                   xpos=np.empty((count, m.nbody, 3), np.float32),
                   xquat=np.empty((count, m.nbody, 4), np.float32), cvel=np.empty((count, m.nbody, 6), np.float32),
                   subtree_com=np.empty((count, m.nbody, 3), np.float32))
        kinds = np.empty(count, dtype=object)
        handed = self.handed

        def work(slot, trials):
            for trial in trials:
                kinds[trial] = self._control(self.data[trial], q[trial], v[trial], w[trial], u[trial], substeps,
                                             None if handed is None else [array[trial] for array in handed], out,
                                             trial)
        self._map(work, count)
        self.handed = (out["qpos"].copy(), out["qvel"].copy(), out["warm"].copy())
        self.most_contacts = out["most_contacts"]
        counted = list(kinds)
        self.counters["calls"] += 1
        self.counters["trial_controls"] += count
        self.counters["substeps"] += count*substeps
        for kind in ("seeded", "pushed", "continued"):
            self.counters[f"{kind}_trial_controls"] += counted.count(kind)
        self.counters["diverged_trial_controls"] += int(np.isnan(out["qpos"][:, 0]).sum())
        bits = out["info"][:, 3]
        self.counters["contact_capacity_trial_controls"] += int(((bits & OVERFLOW_CONTACTS) != 0).sum())
        self.counters["row_capacity_trial_controls"] += int(((bits & OVERFLOW_ROWS) != 0).sum())
        return tuple(torch.from_numpy(out[key]).to(qpos.device)
                     for key in ("raw", "qvel", "warm", "info", "qpos", "xpos", "xquat", "cvel", "subtree_com"))

    def _control(self, data, q, v, w, u, substeps, handed, out, trial):
        m = self.m
        if handed is not None and np.array_equal(q, handed[0]) and np.array_equal(w, handed[2]) \
                and np.array_equal(v[2:], handed[1][2:]):
            if np.array_equal(v, handed[1]):
                kind = "continued"
            else:
                kind = "pushed"
                data.qvel[:2] += v[:2].astype(np.float64) - handed[1][:2].astype(np.float64)
        else:
            kind = "seeded"
            mujoco.mj_resetData(m, data)
            data.qpos[:], data.qvel[:], data.qacc_warmstart[:] = q, v, w
        for warning in range(len(data.warning)):
            data.warning[warning].number = 0
        data.ctrl[:] = u
        diverged = False
        for _ in range(substeps):
            mujoco.mj_step(m, data)
            out["most_contacts"][trial] = max(out["most_contacts"][trial], data.ncon)
            if any(data.warning[int(warning)].number for warning in DIVERGENCE):
                diverged = True
                break
        bits = 0
        for warning, bit in CAPACITY:
            if data.warning[int(warning)].number:
                bits |= bit
        islands = max(int(data.nisland), 1)
        out["info"][trial] = (int(np.max(data.solver_niter[:islands])), data.ncon, data.nefc, bits)
        if diverged:
            for key in ("raw", "qpos", "qvel", "warm", "xpos", "xquat", "cvel", "subtree_com"):
                out[key][trial] = np.nan
            return kind
        # After mj_step the body kinematics are those before the last integration.
        self._bodies(data, out, trial)
        out["raw"][trial], out["qvel"][trial], out["warm"][trial] = data.qpos, data.qvel, data.qacc_warmstart
        out["qpos"][trial] = _normalised(data.qpos)
        return kind


def model_options(m):
    """The model's options (every field of m.opt), its disable and enable flags by name, and per geom its body and
    contact flags, for the record of the check mode."""
    opt = {}
    for name in dir(m.opt):
        if name.startswith("_"):
            continue
        value = getattr(m.opt, name)
        if callable(value):
            continue
        array = np.asarray(value)
        opt[name] = array.tolist() if array.dtype.kind in "biuf" else str(value)
    flags = {}
    for kind, enum, field in (("disabled", mujoco.mjtDisableBit, "disableflags"),
                              ("enabled", mujoco.mjtEnableBit, "enableflags")):
        bits = int(getattr(m.opt, field))
        flags[kind] = sorted(name for name, member in enum.__members__.items()
                             if name.startswith(("mjDSBL_", "mjENBL_")) and bits & int(member))
    return dict(opt=opt, **flags)


def model_record(m):
    """Check (c): the options, the geoms' contact flags, the pairs and the actuators of the model MuJoCo C steps."""
    pairs = [dict(name=m.pair(index).name, condim=int(m.pair_dim[index]), friction=m.pair_friction[index].tolist(),
                  solref=m.pair_solref[index].tolist(), solimp=m.pair_solimp[index].tolist(),
                  margin=float(m.pair_margin[index]), gap=float(m.pair_gap[index])) for index in range(m.npair)]
    geoms = [dict(name=m.geom(index).name, body=m.body(int(m.geom_bodyid[index])).name,
                  type=mujoco.mjtGeom(int(m.geom_type[index])).name, contype=int(m.geom_contype[index]),
                  conaffinity=int(m.geom_conaffinity[index]), condim=int(m.geom_condim[index]))
             for index in range(m.ngeom)]
    actuators = []
    for index in range(m.nu):
        joint = int(m.actuator_trnid[index, 0])
        dof = int(m.jnt_dofadr[joint])
        actuators.append(dict(name=m.actuator(index).name, joint=m.joint(joint).name,
                              gain=float(m.actuator_gainprm[index, 0]), bias=m.actuator_biasprm[index, :3].tolist(),
                              ctrllimited=bool(m.actuator_ctrllimited[index]),
                              forcelimited=bool(m.actuator_forcelimited[index]),
                              actfrclimited=bool(m.jnt_actfrclimited[joint]),
                              ctrlrange=m.actuator_ctrlrange[index].tolist(),
                              forcerange=m.actuator_forcerange[index].tolist(),
                              actfrcrange=m.jnt_actfrcrange[joint].tolist(),
                              armature=float(m.dof_armature[dof]), damping=float(m.dof_damping[dof]),
                              frictionloss=float(m.dof_frictionloss[dof])))
    return dict(**model_options(m), sizes=dict(nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), nbody=int(m.nbody),
                                               ngeom=int(m.ngeom), npair=int(m.npair), nexclude=int(m.nexclude)),
                collisions_outside_pairs=[list(pair) for pair in dynamic_collisions(m)], geoms=geoms, pairs=pairs,
                actuators=actuators)


def build_mujoco_environment(args, physics, device="cuda", **conditions):
    """The environment of ``torch_train.build_environment`` for the run's arguments ``args`` (the motion library of
    ``--clips``, the run's EnvConfig with ``conditions`` changed, no recovery setting), with the step and kinematics
    of ``physics``, which is then bound to the environment's model. build_environment does not take a kinematics
    function, so the same construction is made here."""
    from .env import BM_BODIES, EnvConfig
    from .motion import all_clip_names, load_library
    names = all_clip_names() if args.clips == ["all"] else args.clips
    if args.clips == ["all"] and len(names) != 40:
        raise ValueError("Expected the complete 40-clip library")
    library = load_library(names, bodies=BM_BODIES)
    config = EnvConfig(solver_iterations=args.solver_iterations, residual_action=args.residual_action,
                       action_rate_space=args.action_rate_space, **conditions)
    env = TorchTrackingEnv(library, config, device=device, step_fn=TapeFreeStep(physics.forward),
                           kinematics_fn=physics.kinematics)
    physics.bind(env.m, env.foot_pairs)
    return env


def describe(args, config, checkpoint_path, checkpoint, scene, code, physics, env, library_inputs, precision, gpu):
    """The description of engine_rollouts with the engine field describing MuJoCo C (its version, threads, options and
    the set-up differences); engine_diagnostics is added after the rollouts."""
    return dict(
        run=str(args.run.resolve()), checkpoint=str(checkpoint_path.resolve()),
        completed_updates=checkpoint["completed_updates"], actor_key=args.actor_key,
        run_source_revision=config["git_commit"], run_engine=config.get("engine"), run_gpu=config.get("gpu"),
        source_revision=code, scene=str(args.scene.resolve()), scene_content=scene, termination=args.termination,
        controls=args.controls, engine=physics.describe(), gpu=gpu, torch_version=torch.__version__,
        settings=dict(
            clips=config["clips"], solver_iterations=config["solver_iterations"],
            residual_action=config["residual_action"], action_rate_space=config["action_rate_space"],
            matmul_precision=config["matmul_precision"], observed_precision=precision,
            recovery="not set: the environment is built without the recovery options of the run",
            domain_randomisation="off: foot friction 1, zero joint-encoder bias, zero torso centre-of-mass offset",
            observation_noise="off", actions="deterministic: the actor's output without action noise",
            reset_noise="the scene's unit uniforms (zero where absent); joints limited to the soft range",
            term_height=env.cfg.term_height, term_tilt=env.cfg.term_tilt,
            whole_body_mass_kg=float(env.m.body_subtreemass[0]), inputs=library_inputs))


def check_chains(records, counters):
    """Raise unless every unended trial seeded its MjData once, at placement: the seeded trial controls of a rollout
    are its T trials plus, per ended trial, the controls after the one that ended it (the loop holds an ended trial at
    its earlier state, which seeds it at every later control)."""
    done = np.asarray(records["done"], bool)
    controls, trials = done.shape
    ended = done.any(0)
    expected = trials + int(sum(controls - 1 - int(np.argmax(done[:, trial])) for trial in np.flatnonzero(ended)))
    if counters["seeded_trial_controls"] != expected:
        raise RuntimeError(f"{counters['seeded_trial_controls']} trial controls seeded their MjData where the record "
                           f"gives {expected} ({trials} placements and the controls after each trial's end): a trial "
                           f"was seeded again from the state handed in")


# ------------------------------------------------------------------------------------------------ check mode

def _largest(a, b):
    """The largest absolute difference of two arrays over the entries finite in both (None without any), the trial
    (first axis) where it is, and the count of entries finite in only one."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes {a.shape} and {b.shape} differ")
    both = np.isfinite(a) & np.isfinite(b)
    difference = np.where(both, np.abs(a - b), -1.0)
    largest = float(difference.max(initial=-1.0))
    trial = int(np.unravel_index(int(difference.argmax()), difference.shape)[0]) if largest >= 0 else None
    return dict(largest=largest if largest >= 0 else None, trial=trial,
                finite_in_one_only=int((np.isfinite(a) ^ np.isfinite(b)).sum()))


START_FIELDS = ("start_qpos", "start_qvel", "start_body_pos_w", "start_body_quat_w")


def placement_differences(record, placed, inputs):
    """Check (a): per start field the largest difference between ``placed`` (the start fields of placing the trials
    with MuJoCo C kinematics) and an existing ``record`` of the same trials. Raises when the record's rows are not
    the trials' or when a placed tracked body's nearest recorded body (by position) is another body."""
    for key in ("command_row", "body_row"):
        if not np.array_equal(record[key], inputs[key.split("_")[0]].cpu().numpy()):
            raise ValueError(f"the record's {key} are not the scene's")
    ours, theirs = placed["start_body_pos_w"].astype(np.float64), record["start_body_pos_w"].astype(np.float64)
    distances = np.linalg.norm(ours[:, :, None] - theirs[:, None], axis=-1)
    nearest = distances.argmin(-1)
    order = np.arange(ours.shape[1])
    if not np.all(nearest == order):
        trial, body = (int(x[0]) for x in np.nonzero(nearest != order))
        raise ValueError(f"trial {trial}: tracked body {body} lies nearest recorded body {int(nearest[trial, body])}; "
                         f"the body order differs")
    return {key: _largest(placed[key], record[key]) for key in START_FIELDS}


def start_fields(env, inputs):
    """The record's start fields of placing the trials of ``inputs`` through env.reset (as rollout places them)."""
    with torch.no_grad():
        state = env.reset(inputs["command"], inputs["noise"], body_frame=inputs["body"])
    return dict(start_qpos=state["qpos"].cpu().numpy(), start_qvel=state["qvel"].cpu().numpy(),
                start_body_pos_w=state["xpos"][:, env.body_ids].cpu().numpy(),
                start_body_quat_w=state["xquat"][:, env.body_ids].cpu().numpy())


STEP_FIELDS = dict(qpos=4, qvel=1, warm=2, xpos=5, xquat=6, cvel=7, subtree_com=8)  # forward's outputs read


def control_inputs(env, actor, inputs, lift):
    """The forward's arguments of one control from the trials placed by env.reset, the policy's action at the placed
    state, and the pelvis raised by ``lift`` m."""
    with torch.no_grad():
        state = env.reset(inputs["command"], inputs["noise"], body_frame=inputs["body"])
        action = nets.actor_apply(actor, *env.policy_obs(state))
        qpos = state["qpos"].clone()
        qpos[:, 2] += lift
        return (qpos.contiguous(), state["qvel"].contiguous(), state["warm"].contiguous(),
                env.ctrl_of(action, state["frame"]).contiguous(), state["friction"].contiguous(),
                state["com"].contiguous())


def one_control(arguments, forwards, iterations, substeps):
    """Check (b): {name: outputs} of each forward of ``forwards`` {name: forward} on the same arguments; each forward
    starts from no earlier call."""
    with torch.no_grad():
        return {name: tuple(value.cpu().numpy() for value in forward(*arguments, iterations=iterations,
                                                                      substeps=substeps, tape=False, kinematics=True))
                for name, forward in forwards.items()}


def control_differences(outputs, comparisons, most_contacts):
    """Per comparison (name: (first, second) forward names) the largest differences per field, and per trial the
    solver iterations, contacts, constraint rows and info bits of every forward (those of the last substep) and, for
    the forwards named in ``most_contacts`` {name: [B]}, the largest contact count over the substeps; the same
    differences again over the trials without contacts at the last substep in either forward of the comparison."""
    def counts(name, out, trial):
        found = dict(newton_iterations=int(out[3][trial, 0]), contacts=int(out[3][trial, 1]),
                     constraint_rows=int(out[3][trial, 2]), info_bits=int(out[3][trial, 3]))
        if name in most_contacts:
            found["most_contacts_in_a_substep"] = int(most_contacts[name][trial])
        return found
    trials = len(next(iter(outputs.values()))[0])
    report = dict(per_trial=[{name: counts(name, out, trial) for name, out in outputs.items()}
                             for trial in range(trials)])
    for label, (first, second) in comparisons.items():
        a, b = outputs[first], outputs[second]
        free = (a[3][:, 1] == 0) & (b[3][:, 1] == 0)
        report[label] = dict(
            all_trials={field: _largest(a[index], b[index]) for field, index in STEP_FIELDS.items()},
            trials_without_contacts_at_last_substep=int(free.sum()),
            without_contacts_at_last_substep={field: _largest(a[index][free], b[index][free])
                                              for field, index in STEP_FIELDS.items()} if free.any() else None)
    return report


def _same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == \
        np.ascontiguousarray(b).tobytes()


def repeat_differences(first, second):
    """Check (d): the record fields whose bytes differ between two rollouts of the same trials (so +0.0 and -0.0, or
    two NaN payloads, differ), with the size of the difference: for floating-point fields the largest absolute
    difference over the entries finite in both (0.0 when only signs of zero or NaN payloads differ), otherwise the
    count of unequal entries; a field held by one record only, or with another shape or type, is named so."""
    differing = {}
    for key in sorted(set(first) | set(second)):
        if key not in first or key not in second:
            differing[key] = dict(only_in="first" if key in first else "second")
            continue
        a, b = np.asarray(first[key]), np.asarray(second[key])
        if _same_bits(a, b):
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            differing[key] = dict(shapes=[list(a.shape), list(b.shape)], dtypes=[str(a.dtype), str(b.dtype)])
        elif a.dtype.kind == "f":
            differing[key] = _largest(a, b)
        else:
            differing[key] = dict(entries=int((a != b).sum()))
    return dict(identical=not differing, differing=differing)


def run_checks(env, physics, actor, inputs, termination, record, engine_forward, threads):
    """Checks (a) to (d) (ROLLOUTS.md, "Check mode") with ``engine_forward`` the training engine's forward."""
    m, cfg = env.m, env.cfg
    placement = placement_differences(record, start_fields(env, inputs), inputs)

    stepped = dict(mujoco_c=MujocoPhysics(threads).bind(m, env.foot_pairs),
                   mujoco_c_again=MujocoPhysics(threads).bind(m, env.foot_pairs))
    forwards = dict(engine=engine_forward, engine_again=engine_forward,
                    **{name: physics_c.forward for name, physics_c in stepped.items()})
    comparisons = dict(mujoco_c_minus_engine=("mujoco_c", "engine"), engine_minus_engine=("engine", "engine_again"),
                       mujoco_c_minus_mujoco_c=("mujoco_c", "mujoco_c_again"))
    steps = {}
    for label, lift in (("lifted", LIFT_M), ("placed", 0.0)):
        outputs = one_control(control_inputs(env, actor, inputs, lift), forwards, cfg.solver_iterations, SUBSTEPS)
        most = {name: physics_c.most_contacts for name, physics_c in stepped.items()}
        steps[label] = dict(lift_m=lift, **control_differences(outputs, comparisons, most))
    # A push at a later control does not act in the shorter rollouts.
    short = {**inputs, "controls": min(CHECK_CONTROLS, inputs["controls"])}
    repeats = []
    for _ in range(2):
        physics.clear()
        physics.read_diagnostics(reset=True)
        records = rollout(env, actor, short, termination)
        check_chains(records, physics.read_diagnostics(reset=True))
        repeats.append(records)
    physics.clear()
    return dict(placement=placement, one_control=steps, model=model_record(m),
                repeat=dict(controls=short["controls"], termination=termination, **repeat_differences(*repeats)))


def training_engine_record(directory, scene, termination, library_inputs, clips):
    """The trials' rows and start fields of the rollouts in ``directory`` (description.json and rollouts.npz), which
    check (a) compares with. Raises unless they are a training-engine rollout (engine_rollouts, whose engine field is
    the engine's build record; a MuJoCo C description is refused) of the same scene, termination, library inputs and
    clips."""
    reference = json.loads((directory/"description.json").read_text())
    engine = reference.get("engine")
    if not isinstance(engine, dict) or engine.get("name") == "MuJoCo" or "mujoco_version" in engine:
        raise ValueError(f"{directory}: its engine is not NablaEngine's build record; the check mode compares with a "
                         f"training-engine rollout (engine_rollouts), not a MuJoCo C one")
    for key, value in (("scene_content", scene), ("termination", termination)):
        if reference[key] != value:
            raise ValueError(f"{directory}: its {key} is not this command's")
    if reference["settings"]["inputs"] != library_inputs or reference["settings"]["clips"] != clips:
        raise ValueError(f"{directory}: its library inputs or clips are not this run's")
    with np.load(directory/"rollouts.npz") as stored:
        return {key: stored[key] for key in ("command_row", "body_row") + START_FIELDS}


def build_parser():
    parser = engine_parser()
    parser.prog = "python -m nablamimic.mujoco_rollouts"
    parser.description = "Closed-loop MuJoCo C rollouts of a PyTorch-trainer checkpoint."
    parser.add_argument("--threads", type=int, required=True, help="threads that step the trials")
    parser.add_argument("--checks", type=Path, default=None, metavar="RECORD",
                        help="write checks.json against the engine_rollouts output RECORD of the same scene instead "
                             "of rolling out")
    return parser


def main(argv=None):
    from .params import git_state
    from .torch_train import configure_matmul_precision, write_json
    args = build_parser().parse_args(argv)
    if args.out.exists():
        raise ValueError(f"{args.out} exists")
    config = json.loads((args.run/"config.json").read_text())
    if config.get("training_framework") != "torch":
        raise ValueError(f"{args.run} is not a run of the PyTorch trainer (torch_train)")
    library_inputs = {key: os.environ.get(key) for key in LIBRARY_INPUTS}
    differing = [key for key in LIBRARY_INPUTS if config["inputs"].get(key) != library_inputs[key]]
    if differing:
        raise ValueError("the motion library inputs differ from the run's: " + "; ".join(
            f"{key} {library_inputs[key]!r} here, {config['inputs'].get(key)!r} in the run" for key in differing))
    checkpoint_path = args.run/"ckpt"/f"it_{args.iter:06d}.pkl"
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if args.actor_key not in checkpoint:
        raise ValueError(f"checkpoint it_{args.iter:06d} holds no {args.actor_key}")
    scene = json.loads(args.scene.read_text())
    record = None if args.checks is None else training_engine_record(args.checks, scene, args.termination,
                                                                      library_inputs, config["clips"])
    code = os.environ.get("NABLA_SOURCE_REVISION") or git_state()[0]
    if not code:
        raise ValueError("Declared committed source revision required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one provisioned CUDA device required")
    precision = configure_matmul_precision(config["matmul_precision"])
    physics = MujocoPhysics(args.threads)
    env = build_mujoco_environment(argparse.Namespace(**config), physics, dr=False, obs_noise=False, reset_noise=True)
    inputs = trial_inputs(scene, env, args.controls)
    actor = nets.tree_from_jax(checkpoint[args.actor_key], env.device)
    description = describe(args, config, checkpoint_path, checkpoint, scene, code, physics, env, library_inputs,
                           precision, torch.cuda.get_device_name(0))
    if args.checks is not None:
        from nablaengine import torch_ffi
        if (torch_ffi.OVERFLOW_CONTACTS, torch_ffi.OVERFLOW_ROWS) != (OVERFLOW_CONTACTS, OVERFLOW_ROWS):
            raise ValueError(f"the engine's capacity bits {torch_ffi.OVERFLOW_CONTACTS}, {torch_ffi.OVERFLOW_ROWS} "
                             f"differ from the bits this program reads, {OVERFLOW_CONTACTS}, {OVERFLOW_ROWS}")
        engine = torch_ffi.init(model=env.m, foot_pairs=np.asarray(env.foot_pairs), device=env.device)
        with collected_warnings() as log:
            result = run_checks(env, physics, actor, inputs, args.termination, record, torch_ffi.forward,
                                args.threads)
        result["mujoco_warnings"] = log.counts
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out/"checks.json", dict(description=description, record=str(args.checks.resolve()),
                                               training_engine=engine, **result))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/"description.json", description)
    physics.read_diagnostics(reset=True)
    with collected_warnings() as log:
        records = rollout(env, actor, inputs, args.termination)
    counters = physics.read_diagnostics(reset=True)
    check_chains(records, counters)
    description["engine_diagnostics"] = dict(counters, mujoco_warnings=log.counts)
    write_json(args.out/"description.json", description)
    with (args.out/"rollouts.npz").open("xb") as handle:
        np.savez(handle, **records)


if __name__ == "__main__":
    main()
