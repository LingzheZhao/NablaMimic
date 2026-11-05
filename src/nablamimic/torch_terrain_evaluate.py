"""Finite closed-loop reads of a bound terrain policy, retaining partial trajectories."""
import json
from pathlib import Path

import numpy as np
import torch

from . import torch_nets as nets


def evaluate(env, actor, output, maximum_controls, start_frame=0, reset_pose=None):
    """Execute one nominal physical chain, ending at tracking termination or clip end.

    The caller supplies the already constructed physical environment and EMA or
    live actor explicitly. Tracking termination is recorded by its own name.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if env.terrain is None:
        raise ValueError("Terrain evaluation requires the bound physical sensor")
    frame = torch.tensor([start_frame], device=env.device, dtype=torch.int32)
    reset_noise = env.sample_reset_noise(1)
    reset_noise = {name: torch.zeros_like(value) for name, value in reset_noise.items()}
    if reset_pose is not None:
        reset_noise['position'][:] = reset_pose['translation_m'] / env.reset_position_limits
        reset_noise['rotation'][:] = reset_pose['rotation_rad'] / env.reset_rotation_limits
    state = env.reset(frame, reset_noise)
    rows = []
    reason = 'requested_controls'
    with torch.no_grad():
        for control in range(maximum_controls):
            observation_noise = env.sample_observation_noise(1)
            observation_noise = {name: torch.zeros_like(value) for name, value in observation_noise.items()}
            history, goal = env.policy_obs(state, observation_noise['goal'])
            action = nets.actor_apply(actor, history, goal)
            scan = env.terrain_scan(state)
            next_state = env.step(state, action, observation_noise['proprio'])
            terminated = bool(env.terminated(next_state).item())
            complete = bool(env.clip_ended(next_state).item())
            physical_body = state['xpos'][:, env.body_ids]
            reference_body = env.ref_body_pos[state['frame']]
            world_error = torch.linalg.vector_norm(physical_body-reference_body, dim=-1)
            rows.append(dict(world_body_error_m=world_error[0].cpu().numpy(),
                torso_error_m=float(world_error[0, env.anchor_k].item()), qpos=state['qpos'][0].cpu().numpy(),
                qvel=state['qvel'][0].cpu().numpy(), action=action[0].cpu().numpy(),
                scan=scan[0].cpu().numpy(), frame=int(state['frame'][0].item()),
                tracking_terminated=terminated, clip_ended=complete))
            state = next_state
            if terminated or complete:
                reason = 'tracking_termination' if terminated else 'reference_complete'
                break
    if rows:
        np.savez(output/'trajectory.npz', **{key: np.array([row[key] for row in rows]) for key in rows[0]},
                 final_qpos=state['qpos'].cpu().numpy(), final_qvel=state['qvel'].cpu().numpy())
    report = dict(controls=len(rows), requested_controls=maximum_controls,
                  start_frame=start_frame, stop_reason=reason, perception=env.perception_contract,
                  control_hz=50, stochastic_action_noise=0, physical_fall='unavailable_without_force_classifier')
    (output/'result.json').write_text(json.dumps(report, indent=2))
    return report


def evaluate_checkpoint(env, checkpoint_path, output):
    """Read EMA/live separately at the fixed twenty paired physical starts."""
    import pickle
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with Path(checkpoint_path).open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['perception'] != env.perception_contract:
        raise ValueError('Evaluation sensor layout differs from checkpoint binding')
    if len(env.starts) != 1 or int(env.lengths[0]) != 537:
        raise ValueError('This pilot requires the selected complete 537-frame reference')
    generator = torch.Generator(device='cpu').manual_seed(0)
    positions = .01*(2*torch.rand(20, 3, generator=generator)-1)
    rotations = .02*(2*torch.rand(20, 3, generator=generator)-1)
    positions[0] = 0
    rotations[0] = 0
    identity = dict(checkpoint_path=str(checkpoint_path), completed_updates=checkpoint['completed_updates'])
    reports = []
    original = env.sensor_ablation
    try:
        for weights in ('ema', 'live'):
            key = 'actor_ema' if weights == 'ema' else 'actor'
            actor = nets.tree_from_jax(checkpoint[key], env.device)
            for index in range(20):
                pose = dict(translation_m=positions[index].to(env.device),
                            rotation_rad=rotations[index].to(env.device))
                for sensor in ('measured', 'constant'):
                    env.sensor_ablation = sensor
                    report = evaluate(env, actor, output/f'{weights}_start{index:02d}_{sensor}',
                                      maximum_controls=536, start_frame=int(env.starts[0]), reset_pose=pose)
                    reports.append(dict(**identity, weights=weights, start=index,
                                        sensor_intervention=sensor, **report))
    finally:
        env.sensor_ablation = original
    (output/'panel.json').write_text(json.dumps(dict(identity=identity, reports=reports), indent=2))
    return reports
