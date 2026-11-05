"""Bounded full-library source-task physical reads with explicit saved actor weights."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from nablamimic.torch_train import build_environment
from nablamimic import torch_nets as nets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=('ema','live'), required=True)
    parser.add_argument('--maximum-controls', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for key in ('terrain_model','terrain_model_contract'):
        config[key] = Path(config[key])
    env, bindings = build_environment(argparse.Namespace(**config))
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
        raise ValueError('Source actor observation or physical geometry binding differs')
    actor = nets.tree_from_jax(checkpoint['actor_ema' if args.weights=='ema' else 'actor'], env.device)
    reports = []
    with torch.no_grad():
        for clip, start in enumerate(env.starts):
            frame = torch.tensor([int(start)],device=env.device,dtype=torch.int32)
            noise = {name:torch.zeros_like(value) for name,value in env.sample_reset_noise(1).items()}
            state = env.reset(frame,noise)
            rows=[]
            reason='requested_controls'
            for control in range(args.maximum_controls):
                observation_noise = {name:torch.zeros_like(value) for name,value in env.sample_observation_noise(1).items()}
                history,goal = env.policy_obs(state,observation_noise['goal'])
                action = nets.actor_apply(actor,history,goal)
                stepped = env.step(state,action,observation_noise['proprio'])
                done = bool(env.terminated(stepped).item())
                ended = bool(env.clip_ended(stepped).item())
                rows.append(dict(qpos=state['qpos'][0].cpu().numpy(),qvel=state['qvel'][0].cpu().numpy(),
                    action=stepped['prev_a'][0].cpu().numpy(),depth=history[0,744:].cpu().numpy(),
                    frame=int(state['frame'][0].item()),tracking_terminated=done,clip_ended=ended))
                state=stepped
                if done or ended:
                    reason='tracking_termination' if done else 'reference_complete'
                    break
            output=args.output/f'clip_{clip:02d}'
            output.mkdir()
            np.savez(output/'trajectory.npz',**{key:np.asarray([row[key] for row in rows]) for key in rows[0]},
                final_qpos=state['qpos'].cpu().numpy(),final_qvel=state['qvel'].cpu().numpy())
            reports.append(dict(clip=clip,start_frame=int(start),controls=len(rows),stop_reason=reason))
    (args.output/'result.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),weights=args.weights,
        completed_updates=checkpoint['completed_updates'],perception=env.perception_contract,
        observation=env.observation_contract,reports=reports),indent=2))


if __name__=='__main__':
    main()
