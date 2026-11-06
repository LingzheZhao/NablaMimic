"""Collect fresh, terminal-inclusive source executions for visual world models."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from nablamimic.source_world_model_data import (
    SCHEMA, STATE_KEYS, STATE_SHAPES, TRANSITION_KEYS, TRANSITION_SHAPES,
    episode_schedule, split_windows, validate_episode, validity_masks,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "live"), default="ema")
    parser.add_argument("--backend", choices=("engine", "mujoco"), default="engine")
    parser.add_argument("--ray-backend", choices=("native", "mujoco"), default="native")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sensor-noise", choices=("off", "on"), required=True)
    parser.add_argument("--action-noise-std", type=float, default=0.)
    parser.add_argument("--episode-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--split-type", choices=("temporal", "episode"), required=True)
    parser.add_argument("--heldout-seeds", type=int, nargs="+")
    parser.add_argument("--window-controls", type=int, default=32)
    parser.add_argument("--heldout-fraction", type=float)
    parser.add_argument("--split-gap-controls", type=int)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--checkpoint-artifact-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--engine-revision", required=True)
    parser.add_argument("--environment-artifact-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def collection_protocol(args):
    """Resolve one explicit split protocol before constructing a simulator."""
    schedule = episode_schedule(args.episode_seeds, 10)
    protocol = dict(split_type=args.split_type, window_controls=args.window_controls)
    if args.split_type == "episode":
        if args.heldout_fraction is not None or args.split_gap_controls is not None:
            raise ValueError("Whole-episode splitting uses heldout seeds without temporal fraction or gap")
        protocol["heldout_seeds"] = args.heldout_seeds
    else:
        if args.heldout_seeds is not None or args.heldout_fraction is None or args.split_gap_controls is None:
            raise ValueError("Temporal splitting requires fraction and gap without heldout seeds")
        protocol.update(heldout_fraction=args.heldout_fraction, gap_controls=args.split_gap_controls)
    split_windows([dict(seed=seed, transitions=0) for seed, _ in schedule], protocol)
    return schedule, protocol


def observe(env, state, sensor_noise):
    """One noisy observation per state, plus a clean image without RNG draws."""
    import torch
    finite = all(bool(torch.isfinite(state[name]).all()) for name in
                 ("qpos", "qvel", "warm", "xpos", "xquat", "cvel"))
    if not finite:
        # Unavailable observations remain explicit NaNs with false validity masks.
        observation = torch.full((1, 1320), float("nan"), device=env.device)
        goal = env.source_goal(state).to(torch.float32)
        image = observation[:, 744:].reshape(1, 1, 18, 32)
        return observation, goal, image, image
    if sensor_noise == "on":
        camera_noise = env.camera_noise
        env.camera_noise = None
        try:
            clean = env.depth_image(state).to(torch.float32)
        finally:
            env.camera_noise = camera_noise
    goal_noise = torch.zeros((1, env.goal_dim), device=env.device)
    observation, goal = env.policy_obs(state, goal_noise)
    observation, goal = observation.to(torch.float32), goal.to(torch.float32)
    noisy = observation[:, 744:].reshape(1, 1, 18, 32)
    if sensor_noise == "off":
        clean = noisy
    return observation, goal, clean, noisy


def snapshot(env, state, observed, control):
    """Record body-origin world velocities through the environment's binding."""
    import torch
    observation, goal, clean, noisy = observed
    linear, angular = env.body_velocity(state)
    torso = torch.cat((state["xpos"][:, env.torso_id], state["xquat"][:, env.torso_id],
                       linear[:, env.anchor_k], angular[:, env.anchor_k]), -1)
    values = dict(qpos=state["qpos"], qvel=state["qvel"], warm=state["warm"],
                  torso_state=torso, body_pos=state["xpos"][:, env.body_ids],
                  body_quat=state["xquat"][:, env.body_ids], history=state["hist"].to(torch.float32),
                  policy_observation=observation, goal=goal, depth_clean=clean, depth_noisy=noisy,
                  frame=state["frame"], tracking_terminated=env.terminated(state), clip_ended=env.clip_ended(state))
    result = {name: value[0].detach().cpu().numpy().copy() for name, value in values.items()}
    result["time_s"] = np.asarray(control * .02, dtype=np.float64)
    return result


def episode_arrays(states, transitions):
    arrays = {name: np.stack([row[name] for row in states]) for name in STATE_KEYS
              if name not in ("state_valid", "observation_valid")}
    for name, shape in TRANSITION_SHAPES.items():
        if name == "transition_valid":
            continue
        dtype = np.int32 if name.endswith("contact_count") or name == "substep_info" else np.float32
        arrays[name] = (np.stack([row[name] for row in transitions]) if transitions
                        else np.empty((0, *shape), dtype=dtype))
    arrays.update(validity_masks(arrays))
    validate_episode(arrays)
    return arrays


def write_manifest(path, manifest):
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    args = build_parser().parse_args()
    schedule, protocol = collection_protocol(args)
    if not np.isfinite(args.action_noise_std) or args.action_noise_std < 0:
        raise ValueError("Action exploration standard deviation must be finite and nonnegative")
    for name in ("artifact_id", "checkpoint_artifact_id", "code_revision", "engine_revision", "environment_artifact_id"):
        if not getattr(args, name).strip():
            raise ValueError(f"Declared {name} must be nonempty")
    import torch
    from nablamimic import torch_nets as nets
    from nablamimic.torch_train import build_environment
    from evaluate_source_perceptive import set_evaluation_conditions

    original_config = json.loads(args.config.read_text())
    config = dict(original_config)
    if config["task_inputs"] != "instinct_perceptive_native" or config["clips"] != ["all"]:
        raise ValueError("World-model collection requires all original source-task motions")
    set_evaluation_conditions(config, "source-reset")
    for key in ("terrain_model", "terrain_model_contract"):
        config[key] = Path(config[key])
    env, bindings = build_environment(argparse.Namespace(**config), evaluation_backend=args.backend,
                                      ray_backend=args.ray_backend, device=args.device)
    if env.m.nbody != 31 or env.hist_dim != 1320 or env.goal_dim != 670:
        raise ValueError("Collector requires the declared source torso-root G1 and observation dimensions")
    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if checkpoint["observation"] != env.observation_contract or checkpoint["perception"] != env.perception_contract:
        raise ValueError("Teacher checkpoint and constructed source observation/scene differ")
    actor = nets.tree_from_jax(checkpoint["actor_ema" if args.weights == "ema" else "actor"], env.device)
    frozen = [value.clone() for value in nets.tree_leaves(actor)]
    reference_root = Path(config["motion_lib"])
    reference = json.loads((reference_root / "index.json").read_text())
    screening = json.loads((reference_root / "screening.json").read_text())
    entries = []
    for shard in sorted({entry["shard"] for entry in reference["clips"].values()}):
        with np.load(reference_root / f"{shard}.npz", allow_pickle=False) as archive:
            entries.extend((str(name), reference["clips"][str(name)]) for name in archive["names"])
    if len(entries) != 10 or len(env.starts) != 10 or screening["completed_motion_count"] != 10:
        raise ValueError("Collection must retain all ten source motions")
    for (_, entry), length in zip(entries, env.lengths):
        if entry["length"] != int(length):
            raise ValueError("Collector motion identities differ from the loaded reference order")
    if screening["target_artifact_id"] != reference["scene_artifact_id"]:
        raise ValueError("Reference producer and library scene identities differ")
    if args.sensor_noise == "on" and env.camera_noise is None:
        raise ValueError("Noise-on collection requires the declared source camera noise")
    if args.sensor_noise == "off":
        env.camera_noise = None
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "episodes").mkdir()
    manifest = dict(schema=SCHEMA, status="collecting", artifact_id=args.artifact_id,
        code_revision=args.code_revision, engine_revision=args.engine_revision,
        environment_artifact_id=args.environment_artifact_id,
        producer="collect_source_world_model.py", episode_seeds=args.episode_seeds, motion_count=len(entries),
        randomness="At each episode reset, reseed the global camera RNG and an independent action-noise generator with the episode seed; every seed visits all motions",
        config=dict(path=str(args.config), input_resolved=original_config,
            collection_resolved={key: str(value) if isinstance(value, Path) else value for key, value in config.items()}),
        collection_conditions=dict(physics_backend=args.backend, ray_backend=args.ray_backend,
            reset="source-reset", sensor_noise=args.sensor_noise, action_noise_std=args.action_noise_std,
            reset_proprioception_goal_noise="zero", depth_pair="Same physical input state; clean rendering consumes no noise draws",
            unavailable_observation="NaN with observation_valid false, retained in its actual episode"),
        teacher=dict(artifact_id=args.checkpoint_artifact_id, path=str(args.checkpoint), weights=args.weights,
            completed_updates=checkpoint["completed_updates"], normalizer="Frozen source_norm in the declared checkpoint",
            observation=env.observation_contract, perception=env.perception_contract),
        reference=dict(artifact_id=screening["artifact_id"], path=str(reference_root),
            velocity_definition=reference["velocity_definition"]),
        bindings=bindings, clock=dict(control_dt_s=.02, physics_dt_s=float(env.m.opt.timestep), substeps=4),
        coordinates=dict(torso_state="world xyz, WXYZ quaternion, world body-origin linear velocity, world angular velocity",
            qvel="World root linear velocity, torso-body-frame root angular velocity, then declared joints",
            history="Unnormalized float32 eight-frame proprioception supplied to the teacher",
            depth="Normalized optical depth in [0,1], shape [1,18,32]",
            transition="State i and actual PD target i produce state i+1; terminal row has no executed action"),
        body_names=[env.m.body(int(index)).name for index in env.body_ids],
        measurement_body_names=[env.m.body(index).name for index in range(env.m.nbody)],
        joint_names=[env.m.joint(int(index)).name for index in env.m.actuator_trnid[:, 0]],
        measured_forces=dict(sampling_hz=200., point="Each physics input state",
            values="Actual world contact forces; terrain-only and all-contact fields retain distinct meanings",
            backend=bindings["evaluation_physics"] if args.backend == "mujoco" else bindings["engine"]),
        window_split=protocol,
        split_bounds="Half-open transition intervals; a window's state interval includes its terminal row; episode splits assign every executed transition to exactly one split",
        episodes=[], failed_conditions=[])
    write_manifest(args.output / "manifest.json", manifest)
    with torch.no_grad():
        for episode, (seed, clip_index) in enumerate(schedule):
            torch.manual_seed(seed)
            np.random.seed(seed)
            exploration = torch.Generator(device=env.device).manual_seed(seed)
            name, entry = entries[clip_index]
            frame = torch.tensor([int(env.starts[clip_index])], device=env.device, dtype=torch.int32)
            noise = {key: torch.zeros_like(value) for key, value in env.sample_reset_noise(1).items()}
            if args.backend == "mujoco":
                env.mujoco_adapter.reset_clock()
            state = env.reset(frame, noise)
            observed = observe(env, state, args.sensor_noise)
            states = [snapshot(env, state, observed, 0)]
            transitions = []
            reason = "reference_complete"
            requested = int(env.lengths[clip_index]) - 1
            for control in range(requested):
                if bool(env.terminated(state).item()):
                    reason = "tracking_termination"
                    break
                if not bool(torch.isfinite(observed[0]).all() & torch.isfinite(observed[1]).all()):
                    reason = "nonfinite_observation"
                    break
                raw = nets.actor_apply(actor, observed[0], observed[1])
                perturbation = args.action_noise_std * torch.randn(raw.shape, device=env.device, generator=exploration)
                proposed = raw + perturbation
                if not bool(torch.isfinite(proposed).all()):
                    reason = "nonfinite_policy_action"
                    break
                executed = env.bound_action(proposed)
                target = env.ctrl_of(executed, state["frame"])
                proprio = torch.zeros((1, 93), device=env.device)
                stepped, measured = env.step_recorded(state, proposed, proprio,
                    measurement_backend=env.mujoco_adapter if args.backend == "mujoco" else None)
                if not torch.equal(executed, stepped["prev_a"]):
                    raise ValueError("Recorded executed action differs from the source environment")
                values = dict(actor_action=raw, exploration_noise=perturbation, action_proposed=proposed,
                    action=stepped["prev_a"], pd_target=target,
                    substep_terrain_force=measured["terrain_wrench"][..., :3],
                    substep_all_force=measured["all_contact_wrench"][..., :3],
                    substep_terrain_contact_count=measured["terrain_contact_count"],
                    substep_all_contact_count=measured["all_contact_count"],
                    substep_actuator_torque=measured["actuator_torque"], substep_info=measured["substep_info"])
                transitions.append({key: value[0].detach().cpu().numpy().copy() for key, value in values.items()})
                state = stepped
                observed = observe(env, state, args.sensor_noise)
                states.append(snapshot(env, state, observed, control + 1))
                if bool(env.terminated(state).item()):
                    reason = "tracking_termination"
                    break
                if bool(env.clip_ended(state).item()):
                    break
            arrays = episode_arrays(states, transitions)
            relative_path = f"episodes/episode_{episode:02d}.npz"
            np.savez_compressed(args.output / relative_path, **arrays)
            record = dict(episode_index=episode, seed=seed, clip_index=clip_index, reference_clip=name,
                source_motion_file=entry["motion_file"], terrain_id=entry["terrain_id"], file=relative_path,
                requested_controls=requested, transitions=len(transitions), states=len(states), stop_reason=reason,
                reference_complete=len(transitions) == requested and not bool(arrays["tracking_terminated"][-1]),
                invalid_states=int((~arrays["state_valid"]).sum()),
                invalid_observations=int((~arrays["observation_valid"]).sum()),
                valid_transitions=int(arrays["transition_valid"].sum()),
                clipped_action_components=int(np.count_nonzero(arrays["action_proposed"] != arrays["action"])))
            manifest["episodes"].append(record)
            if reason != "reference_complete":
                manifest["failed_conditions"].append(dict(episode_index=episode, stop_reason=reason))
            if record["invalid_states"] or record["invalid_observations"] or record["valid_transitions"] != len(transitions):
                manifest["failed_conditions"].append(dict(episode_index=episode, issue="nonfinite_recorded_values"))
            write_manifest(args.output / "manifest.json", manifest)
    if not all(torch.equal(before, after) for before, after in zip(frozen, nets.tree_leaves(actor))):
        raise ValueError("Collection changed teacher weights or normalization")
    windows, splits = split_windows(manifest["episodes"], protocol)
    for episode, split in zip(manifest["episodes"], splits):
        episode["split"] = split
    np.savez(args.output / "windows.npz", **windows)
    manifest.update(status="complete", teacher_weights_and_normalizer_unchanged=True,
                    window_counts={name: len(rows) for name, rows in windows.items()})
    write_manifest(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
