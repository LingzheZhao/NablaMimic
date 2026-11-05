"""Finite closed-loop reads of a bound terrain policy, retaining partial trajectories."""
import json
from pathlib import Path

import numpy as np
import torch

from . import torch_nets as nets


def evaluate(env, actor, output, maximum_controls, start_frame=0):
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
            rows.append(dict(qpos=state['qpos'][0].cpu().numpy(),
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
                  control_hz=50, stochastic_action_noise=0)
    (output/'result.json').write_text(json.dumps(report, indent=2))
    return report
