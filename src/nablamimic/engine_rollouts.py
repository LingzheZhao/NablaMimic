"""Closed-loop rollouts of a trained policy on NablaEngine, the simulator the trainer uses, for the trials of a scene
file: placed starts, starts at a command row alone, a clip's first row, and pushes. ROLLOUTS.md gives the scene file,
the environment, the terminations and the record.

  python -m nablamimic.engine_rollouts --run RUN --iter 3000 --scene scene.json --termination nonfinite \\
      --controls 1000 --out OUT
"""

import argparse
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np
import torch

from . import recovery
from . import torch_nets as nets
from .torch_env import select_reset

TERMINATIONS = ("recipe", "nonfinite")
UNIFORMS = dict(position=3, rotation=3, velocity=6, joints=29)
DIRECTIONS = dict(forward=(1.0, 0.0), backward=(-1.0, 0.0), left=(0.0, 1.0), right=(0.0, -1.0))
TRIAL_KEYS = {"command_row", "body_row", "reset_uniforms", "push"}
PUSH_KEYS = ({"control", "velocity_xy"}, {"control", "force", "duration", "direction"})
OVERFLOW_CONTACTS, OVERFLOW_ROWS = 4, 16  # torch_ffi's bits of info[:, 3]
LIBRARY_INPUTS = ("LAFAN1_G1_DIR", "MOTION_CACHE_DIR", "MENAGERIE_DIR")


class TapeFreeStep:
    """``torch_ffi.step`` as ``TorchTrackingEnv.step`` calls it, through ``forward`` (``torch_ffi.forward``) without a
    tape, keeping the per-environment ``info`` [B, 4] of the last call (Newton iterations of the last substep, contacts,
    constraint rows, capacity bits). Without autograd ``step`` returns ``forward``'s outputs without the info; its
    backward options do not act on a forward."""

    def __init__(self, forward):
        self.forward, self.info = forward, None

    def __call__(self, qpos, qvel, warm, ctrl, friction, com, *, iterations, substeps, kinematics, **backward):
        outputs = self.forward(qpos, qvel, warm, ctrl, friction, com, iterations=iterations, substeps=substeps,
                               tape=False, kinematics=kinematics)
        self.info = outputs[3]
        return tuple(outputs[:3]) + tuple(outputs[4:])


def _row(value, frames, what):
    if type(value) is not int or not 0 <= value < frames:
        raise ValueError(f"{what} {value!r} is not a row of the motion library (0 to {frames - 1})")
    return value


def _numbers(value, count, what, low=-math.inf, high=math.inf):
    if (not isinstance(value, list) or len(value) != count
            or any(type(x) not in (int, float) or not (math.isfinite(x) and low <= x <= high) for x in value)):
        raise ValueError(f"{what} must be {count} finite numbers in [{low}, {high}], got {value!r}")
    return value


def _positive(value, what):
    if type(value) not in (int, float) or not 0 < value < math.inf:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return value


def trial_inputs(scene, env, controls):
    """The inputs of ``controls`` controls for the trials of a scene document (ROLLOUTS.md, "Scene"): command and
    body rows [T], the reset noise {name: [T, n]}, the push controls [T] (-1 without a push), the world velocity
    changes [T, 2] and the changes in the pelvis yaw frame [T, 2]. Unknown keys, rows outside the library, uniforms
    outside [-1, 1] and push controls outside the rollout are refused."""
    if not isinstance(scene, dict) or set(scene) != {"trials"} or not isinstance(scene["trials"], list) \
            or not scene["trials"]:
        raise ValueError(f"a scene is {{'trials': [trial, ...]}} with at least one trial, got keys "
                         f"{sorted(scene) if isinstance(scene, dict) else scene!r}")
    if not isinstance(controls, int) or controls < 1:
        raise ValueError(f"controls must be a positive integer, got {controls!r}")
    mass = None
    command, body, push_control, world, heading_frame = [], [], [], [], []
    noise = {name: [] for name in UNIFORMS}
    for index, trial in enumerate(scene["trials"]):
        what = f"trial {index}"
        if not isinstance(trial, dict) or "command_row" not in trial or set(trial) - TRIAL_KEYS:
            raise ValueError(f"{what}: keys {sorted(trial) if isinstance(trial, dict) else trial!r}; a trial has "
                             f"command_row and optionally body_row, reset_uniforms and push")
        command.append(_row(trial["command_row"], env.n_frames, f"{what} command_row"))
        body.append(_row(trial.get("body_row", trial["command_row"]), env.n_frames, f"{what} body_row"))
        uniforms = trial.get("reset_uniforms", {name: [0.0]*size for name, size in UNIFORMS.items()})
        if not isinstance(uniforms, dict) or set(uniforms) != set(UNIFORMS):
            raise ValueError(f"{what}: reset_uniforms {uniforms!r}, expected the keys {sorted(UNIFORMS)}")
        for name, size in UNIFORMS.items():
            noise[name].append(_numbers(uniforms[name], size, f"{what} reset_uniforms {name}", -1.0, 1.0))
        push = trial.get("push")
        velocity, local = [0.0, 0.0], [0.0, 0.0]
        if push is None:
            push_control.append(-1)
        else:
            if not isinstance(push, dict) or set(push) not in PUSH_KEYS:
                raise ValueError(f"{what}: push keys {sorted(push) if isinstance(push, dict) else push!r}, expected "
                                 f"{sorted(PUSH_KEYS[0])} or {sorted(PUSH_KEYS[1])}")
            if type(push["control"]) is not int or not 0 <= push["control"] < controls:
                raise ValueError(f"{what}: push control {push['control']!r} is not a control of the rollout "
                                 f"(0 to {controls - 1})")
            push_control.append(push["control"])
            if "velocity_xy" in push:
                velocity = _numbers(push["velocity_xy"], 2, f"{what} push velocity_xy")
            else:
                if push["direction"] not in DIRECTIONS:
                    raise ValueError(f"{what}: push direction {push['direction']!r} is not one of {sorted(DIRECTIONS)}")
                if mass is None:
                    mass = float(env.m.body_subtreemass[0])
                change = _positive(push["force"], f"{what} push force")*_positive(push["duration"],
                                                                                  f"{what} push duration")/mass
                local = [change*value for value in DIRECTIONS[push["direction"]]]
        world.append(velocity)
        heading_frame.append(local)

    def tensor(values, dtype=torch.float32):
        return torch.tensor(values, dtype=dtype, device=env.device)
    return dict(controls=controls, command=tensor(command, torch.int32), body=tensor(body, torch.int32),
                noise={name: tensor(values) for name, values in noise.items()},
                push_control=tensor(push_control, torch.int32), push_world=tensor(world),
                push_heading_frame=tensor(heading_frame))


def after_end(ended, value):
    """``value`` [B, ...] with NaN, -1 or false (by its type) in the trials that ``ended`` before this control."""
    missing = float("nan") if value.is_floating_point() else False if value.dtype == torch.bool else -1
    return torch.where(ended.reshape(ended.shape + (1,)*(value.ndim - 1)), missing, value)


def rollout(env, actor, inputs, termination):
    """The records (ROLLOUTS.md, "Record") of the deterministic controls of ``actor`` for the trial ``inputs`` of
    ``trial_inputs`` on ``env``, whose step is a ``TapeFreeStep``, with the trial end ``termination`` (recipe or
    nonfinite)."""
    if termination not in TERMINATIONS:
        raise ValueError(f"termination {termination!r} is not one of {TERMINATIONS}")
    if env.recovery is not None:
        raise ValueError("rollouts read the environment without the recovery setting")
    if not isinstance(env.engine_step, TapeFreeStep):
        raise ValueError("rollouts need the tape-free engine step that reports per-environment info")
    if not env.cfg.reset_noise or env.cfg.dr or env.cfg.obs_noise:
        raise ValueError(f"rollouts read the environment with reset noise on and domain randomisation and observation "
                         f"noise off, got reset_noise={env.cfg.reset_noise}, dr={env.cfg.dr}, "
                         f"obs_noise={env.cfg.obs_noise}")
    with torch.no_grad():
        # Every trial starts through the reset of recovery training's starts.
        state = env.reset(inputs["command"], inputs["noise"], body_frame=inputs["body"])
        count = state["frame"].shape[0]
        ended = torch.zeros(count, dtype=torch.bool, device=env.device)
        applied_velocity = torch.zeros((count, 2), device=env.device)
        rows = []
        start = dict(start_qpos=state["qpos"], start_qvel=state["qvel"],
                     start_body_pos_w=state["xpos"][:, env.body_ids], start_body_quat_w=state["xquat"][:, env.body_ids])
        for control in range(inputs["controls"]):
            pushed = (inputs["push_control"] == control) & ~ended
            yaw = recovery.heading(state["qpos"][:, 3:7])
            cos, sin = torch.cos(yaw), torch.sin(yaw)
            x, y = inputs["push_heading_frame"].unbind(-1)
            change = inputs["push_world"] + torch.stack((x*cos - y*sin, x*sin + y*cos), -1)
            applied_velocity = torch.where(pushed[:, None], change, applied_velocity)
            qvel = state["qvel"]
            state = {**state, "qvel": torch.cat((torch.where(pushed[:, None], qvel[:, :2] + change, qvel[:, :2]),
                                                 qvel[:, 2:]), -1)}
            history, goal = env.policy_obs(state)
            stepped = env.step(state, nets.actor_apply(actor, history, goal))
            info = env.engine_step.info
            dz, dg = env.termination_differences(stepped["qpos"], stepped["frame"])
            nonfinite = ~torch.isfinite(stepped["qpos"]).all(-1) | ~torch.isfinite(stepped["qvel"]).all(-1)
            height, tilt = dz > env.cfg.term_height, dg > env.cfg.term_tilt
            terminated = env.terminated(stepped)
            if not torch.equal(terminated, height | tilt | nonfinite):
                raise RuntimeError("the recorded termination terms do not make up the recipe's termination")
            ending = (terminated if termination == "recipe" else nonfinite) & ~ended
            bits = info[:, 3]
            row = dict(
                body_pos_w=stepped["xpos"][:, env.body_ids], body_quat_w=stepped["xquat"][:, env.body_ids],
                qpos=stepped["qpos"], qvel=stepped["qvel"], frame_before=state["frame"], frame_after=stepped["frame"],
                clip_last_row=env.ref_last[stepped["frame"]], available=~nonfinite, done=ending,
                termination_height=height, termination_tilt=tilt, termination_nonfinite=nonfinite,
                push_applied=pushed, engine_newton_iterations=info[:, 0], engine_contacts=info[:, 1],
                engine_constraint_rows=info[:, 2], engine_info_bits=bits,
                engine_contact_capacity_exceeded=(bits & OVERFLOW_CONTACTS) != 0,
                engine_row_capacity_exceeded=(bits & OVERFLOW_ROWS) != 0)
            rows.append({key: after_end(ended, value) for key, value in row.items()})
            ended = ended | ending
            # An ended trial keeps the state it started this control from.
            state = select_reset(ended, state, stepped)
    records = {key: torch.stack([row[key] for row in rows]).cpu().numpy() for key in rows[0]}
    records.update({key: value.cpu().numpy() for key, value in start.items()},
                   command_row=inputs["command"].cpu().numpy(), body_row=inputs["body"].cpu().numpy(),
                   push_control=inputs["push_control"].cpu().numpy(), push_velocity_xy=applied_velocity.cpu().numpy())
    return records


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m nablamimic.engine_rollouts",
                                     description="Closed-loop NablaEngine rollouts of a PyTorch-trainer checkpoint.")
    parser.add_argument("--run", type=Path, required=True, help="run directory of torch_train (config.json, ckpt/)")
    parser.add_argument("--iter", type=int, required=True,
                        help="update of the numbered checkpoint ckpt/it_NNNNNN.pkl (six digits)")
    parser.add_argument("--actor-key", choices=("actor_ema", "actor"), default="actor_ema",
                        help="actor_ema: the running average of the weights; actor: the weights as trained")
    parser.add_argument("--scene", type=Path, required=True, help="scene file (JSON) of the trials")
    parser.add_argument("--termination", choices=TERMINATIONS, required=True,
                        help="recipe: the recipe's termination ends a trial; nonfinite: only a non-finite state does")
    parser.add_argument("--controls", type=int, required=True, help="controls per trial")
    parser.add_argument("--out", type=Path, required=True, help="output directory; must not exist")
    return parser


def main(argv=None):
    from .params import git_state
    from .torch_train import build_environment, configure_matmul_precision, write_json
    from nablaengine import torch_ffi
    args = build_parser().parse_args(argv)
    if (torch_ffi.OVERFLOW_CONTACTS, torch_ffi.OVERFLOW_ROWS) != (OVERFLOW_CONTACTS, OVERFLOW_ROWS):
        raise ValueError(f"the engine's capacity bits {torch_ffi.OVERFLOW_CONTACTS}, {torch_ffi.OVERFLOW_ROWS} differ "
                         f"from the bits this program reads, {OVERFLOW_CONTACTS}, {OVERFLOW_ROWS}")
    config = json.loads((args.run/"config.json").read_text())
    if config.get("training_framework") != "torch":
        raise ValueError(f"{args.run} is not a run of the PyTorch trainer (torch_train)")
    # The scene's rows index the motion library, which these variables locate.
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
    code = os.environ.get("NABLA_SOURCE_REVISION") or git_state()[0]
    if not code:
        raise ValueError("Declared committed source revision required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one provisioned CUDA device required")
    precision = configure_matmul_precision(config["matmul_precision"])
    env = build_environment(argparse.Namespace(**config), step_fn=TapeFreeStep(torch_ffi.forward), dr=False,
                            obs_noise=False, reset_noise=True)
    inputs = trial_inputs(scene, env, args.controls)
    engine = torch_ffi.init(model=env.m, foot_pairs=np.asarray(env.foot_pairs), device=env.device)
    actor = nets.tree_from_jax(checkpoint[args.actor_key], env.device)
    args.out.mkdir(parents=True, exist_ok=False)
    description = dict(
        run=str(args.run.resolve()), checkpoint=str(checkpoint_path.resolve()),
        completed_updates=checkpoint["completed_updates"], actor_key=args.actor_key,
        run_source_revision=config["git_commit"], run_engine=config.get("engine"), run_gpu=config.get("gpu"),
        source_revision=code, scene=str(args.scene.resolve()), scene_content=scene, termination=args.termination,
        controls=args.controls, engine=engine, gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
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
    write_json(args.out/"description.json", description)
    torch_ffi.read_diagnostics(reset=True)
    records = rollout(env, actor, inputs, args.termination)
    description["engine_diagnostics"] = torch_ffi.read_diagnostics(reset=True)
    write_json(args.out/"description.json", description)
    with (args.out/"rollouts.npz").open("xb") as handle:
        np.savez(handle, **records)

if __name__ == "__main__":
    main()
