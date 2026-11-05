"""Finite closed-loop reads of a bound terrain policy, retaining partial trajectories."""
import json
from pathlib import Path

import numpy as np
import torch

from . import torch_nets as nets


def posture_fall(updot, height, reference_complete):
    """Classify >=0.2 s observed bad posture at 50 Hz, including terminal state."""
    bad = (np.asarray(updot) < .5) | (np.asarray(height) < .25)
    run = 0
    for value in bad:
        run = run + 1 if value else 0
        if run >= 11:  # Eleven state samples span ten 20 ms intervals.
            return True
    if not reference_complete and (len(bad) < 11 or run > 0):
        return 'unavailable_truncated_before_0.2s'
    return False


def evaluate(env, actor, output, maximum_controls, start_frame=0, reset_pose=None):
    """Execute one nominal physical chain, ending at tracking termination or clip end.

    The caller supplies the already constructed physical environment and EMA or
    live actor explicitly. Tracking termination is recorded by its own name.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not hasattr(env, 'perception_body_names'):
        from .env import BM_BODIES
        env.perception_body_names = BM_BODIES
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
            action = next_state['prev_a']
            terminated = bool(env.terminated(next_state).item())
            complete = bool(env.clip_ended(next_state).item())
            physical_body = state['xpos'][:, env.body_ids]
            reference_body = env.ref_body_pos[state['frame']]
            world_error = torch.linalg.vector_norm(physical_body-reference_body, dim=-1)
            q = state['qpos'][0]
            updot = 1 - 2*(q[4]**2 + q[5]**2)
            pelvis_index = 0
            ankle_indices = [i for i, name in enumerate(env.perception_body_names) if 'ankle_roll' in name]
            rows.append(dict(pelvis_updot=float(updot.item()), pelvis_height_m=float(q[2].item()),
                pelvis_error_m=float(world_error[0, pelvis_index].item()),
                ankle_height_m=physical_body[0, ankle_indices, 2].cpu().numpy(),
                world_body_error_m=world_error[0].cpu().numpy(),
                torso_error_m=float(world_error[0, env.anchor_k].item()), qpos=state['qpos'][0].cpu().numpy(),
                qvel=state['qvel'][0].cpu().numpy(), action=action[0].cpu().numpy(),
                scan=scan[0].cpu().numpy(), frame=int(state['frame'][0].item()),
                tracking_terminated=terminated, clip_ended=complete))
            state = next_state
            if terminated or complete:
                reason = 'tracking_termination' if terminated else 'reference_complete'
                break
    return finish_trial(env, state, rows, output, maximum_controls, start_frame, reason)


def finish_trial(env, state, rows, output, maximum_controls, start_frame, reason):
    """Write the same scientific readings for serial and batched execution."""
    terminal_error = torch.linalg.vector_norm(
        state['xpos'][:, env.body_ids]-env.ref_body_pos[state['frame']], dim=-1)[0].cpu().numpy()
    final_qpos = state['qpos'][0].cpu().numpy()
    posture_qpos = np.stack([row['qpos'] for row in rows] + [final_qpos])
    posture_updot = 1-2*(posture_qpos[:, 4]**2+posture_qpos[:, 5]**2)
    posture_height = posture_qpos[:, 2]
    posture_result = posture_fall(posture_updot, posture_height, reason == 'reference_complete')
    if rows:
        np.savez(output/'trajectory.npz', **{key: np.array([row[key] for row in rows]) for key in rows[0]},
                 final_qpos=state['qpos'].cpu().numpy(), final_qvel=state['qvel'].cpu().numpy(),
                 final_frame=state['frame'].cpu().numpy(), final_world_body_error_m=terminal_error,
                 final_body_position=state['xpos'][:, env.body_ids].cpu().numpy(),
                 posture_updot=posture_updot, posture_height_m=posture_height)
    report = dict(controls=len(rows), requested_controls=maximum_controls,
                  start_frame=start_frame, stop_reason=reason, perception=env.perception_contract,
                  control_hz=50, stochastic_action_noise=0, posture_fall='unobserved')
    report['posture_fall'] = posture_result
    report['posture_states'] = len(posture_qpos)
    if rows:
        report['mean_world_body_error_m'] = float(np.mean([row['world_body_error_m'].mean() for row in rows] + [terminal_error.mean()]))
        report['mean_torso_error_m'] = float(np.mean([row['torso_error_m'] for row in rows] + [terminal_error[env.anchor_k]]))
        report['passing_trial'] = (reason == 'reference_complete' and
            report['mean_world_body_error_m'] <= .10 and report['mean_torso_error_m'] <= .10
            and posture_result is False)
        seated = [row for row in rows if 4.5 <= row['frame']/50 <= 5.5]
        report['seat_interval'] = dict(requested_s=[4.5, 5.5], controls=len(seated))
        for key in ('torso_error_m', 'pelvis_error_m', 'ankle_height_m'):
            report['seat_interval'][key] = np.mean([row[key] for row in seated], axis=0).tolist() if seated else None
    (output/'result.json').write_text(json.dumps(report, indent=2))
    return report


def evaluate_batch(env, actor, outputs, maximum_controls, start_frame, reset_poses):
    """Advance active chains only, removing each terminal chain before the next step."""
    count = len(outputs)
    for output in outputs:
        Path(output).mkdir(parents=True, exist_ok=False)
    frames = torch.full((count,), start_frame, device=env.device, dtype=torch.int32)
    noise = {name: torch.zeros_like(value) for name, value in env.sample_reset_noise(count).items()}
    noise['position'][:] = reset_poses['translation_m'] / env.reset_position_limits
    noise['rotation'][:] = reset_poses['rotation_rad'] / env.reset_rotation_limits
    state = env.reset(frames, noise)
    active = list(range(count))
    rows = [[] for _ in outputs]
    reports = [None] * count
    ankle_indices = [i for i, name in enumerate(env.perception_body_names) if 'ankle_roll' in name]
    with torch.no_grad():
        for control in range(maximum_controls):
            noise = {name: torch.zeros_like(value) for name, value in env.sample_observation_noise(len(active)).items()}
            history, goal = env.policy_obs(state, noise['goal'])
            actions = nets.actor_apply(actor, history, goal)
            scan = env.terrain_scan(state)
            stepped = env.step(state, actions, noise['proprio'])
            actions = stepped['prev_a']
            terminated = env.terminated(stepped).cpu().numpy()
            complete = env.clip_ended(stepped).cpu().numpy()
            body = state['xpos'][:, env.body_ids]
            errors = torch.linalg.vector_norm(body-env.ref_body_pos[state['frame']], dim=-1)
            arrays = {key: value.detach().cpu().numpy() for key, value in dict(
                qpos=state['qpos'], qvel=state['qvel'], action=actions, scan=scan,
                frame=state['frame'], world_body_error_m=errors, body=body).items()}
            keep = []
            for slot, trial in enumerate(active):
                q = arrays['qpos'][slot]
                rows[trial].append(dict(qpos=q, qvel=arrays['qvel'][slot], action=arrays['action'][slot],
                    scan=arrays['scan'][slot], frame=int(arrays['frame'][slot]),
                    world_body_error_m=arrays['world_body_error_m'][slot],
                    torso_error_m=float(arrays['world_body_error_m'][slot, env.anchor_k]),
                    pelvis_error_m=float(arrays['world_body_error_m'][slot, 0]),
                    ankle_height_m=arrays['body'][slot, ankle_indices, 2],
                    pelvis_updot=float(1-2*(q[4]**2+q[5]**2)), pelvis_height_m=float(q[2]),
                    tracking_terminated=bool(terminated[slot]), clip_ended=bool(complete[slot])))
                if terminated[slot] or complete[slot] or control == maximum_controls-1:
                    final_state = {key: value[slot:slot+1] for key, value in stepped.items()}
                    reason = ('tracking_termination' if terminated[slot] else
                              'reference_complete' if complete[slot] else 'requested_controls')
                    reports[trial] = finish_trial(env, final_state, rows[trial], Path(outputs[trial]),
                                                 maximum_controls, start_frame, reason)
                else:
                    keep.append(slot)
            if not keep:
                break
            indices = torch.tensor(keep, device=env.device, dtype=torch.long)
            state = {key: value.index_select(0, indices) for key, value in stepped.items()}
            active = [active[slot] for slot in keep]
    return reports


def evaluate_checkpoint(env, checkpoint_path, output, execution="serial", starts=20):
    """Read EMA/live separately at the fixed twenty paired physical starts."""
    import pickle
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with Path(checkpoint_path).open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['perception'] != env.perception_contract:
        raise ValueError('Evaluation sensor layout differs from checkpoint binding')
    if checkpoint['observation'] != env.observation_contract:
        raise ValueError('Evaluation policy observation differs from checkpoint binding')
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
            for sensor in ('measured', 'constant'):
                env.sensor_ablation = sensor
                outputs = [output/f'{weights}_start{index:02d}_{sensor}' for index in range(starts)]
                if execution == 'batch':
                    poses = dict(translation_m=positions[:starts].to(env.device),
                                 rotation_rad=rotations[:starts].to(env.device))
                    group = evaluate_batch(env, actor, outputs, 536, int(env.starts[0]), poses)
                else:
                    group = []
                    for index in range(starts):
                        pose = dict(translation_m=positions[index].to(env.device),
                                    rotation_rad=rotations[index].to(env.device))
                        group.append(evaluate(env, actor, outputs[index], 536, int(env.starts[0]), pose))
                for index, report in enumerate(group):
                    reports.append(dict(**identity, weights=weights, start=index,
                                        sensor_intervention=sensor, execution=execution, **report))
    finally:
        env.sensor_ablation = original
    decisions = []
    for weights in ('ema', 'live'):
        for sensor in ('measured', 'constant'):
            selected = [row for row in reports if row['weights'] == weights and row['sensor_intervention'] == sensor]
            passing = sum(row.get('passing_trial', False) for row in selected)
            decisions.append(dict(weights=weights, sensor_intervention=sensor,
                                  passing_trials=passing, trials=len(selected), passes=starts == 20 and passing >= 18))
    (output/'panel.json').write_text(json.dumps(dict(identity=identity, reports=reports, decisions=decisions), indent=2))
    return reports


def main():
    import argparse
    from .torch_train import build_environment
    from .env import BM_BODIES
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True, help='Resolved training config.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execution', choices=('serial', 'batch'), default='serial')
    parser.add_argument('--starts', type=int, choices=range(1, 21), default=20)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('terrain_model', 'terrain_model_contract', 'terrain_sensor'):
        config[key] = Path(config[key])
    env, bindings = build_environment(argparse.Namespace(**config))
    env.perception_body_names = BM_BODIES
    evaluate_checkpoint(env, args.checkpoint, args.output, args.execution, args.starts)


if __name__ == '__main__':
    main()
