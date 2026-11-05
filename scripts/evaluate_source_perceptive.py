"""Bounded full-library source-task physical reads with explicit saved actor weights."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from nablamimic.torch_train import build_environment
from nablamimic import torch_nets as nets
from nablamimic.source_evaluation_metrics import summarize_source_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=('ema','live','official'), required=True)
    parser.add_argument('--maximum-controls', type=int, help='Default is each complete reference length')
    parser.add_argument('--sensor-noise', choices=('on','off'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--record-substeps', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_controls is not None and args.maximum_controls < 1:
        raise ValueError("Requested source rollout must contain at least one control")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for key in ('terrain_model','terrain_model_contract'):
        config[key] = Path(config[key])
    env, bindings = build_environment(argparse.Namespace(**config))
    if args.weights == 'official':
        from nablamimic.official_source_policy import OfficialSourcePolicy
        if env.observation_contract['task'] != 'instinct_perceptive_native':
            raise ValueError('Official source policy requires the source perceptive task')
        if env.cfg.residual_action:
            raise ValueError('Official source policy requires absolute actions')
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        actor = OfficialSourcePolicy(checkpoint).to(env.device)
    else:
        actor = None
    if args.weights != 'official':
        with args.checkpoint.open('rb') as handle:
            checkpoint = pickle.load(handle)
        if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
            raise ValueError('Source actor observation or physical geometry binding differs')
        actor = nets.tree_from_jax(checkpoint['actor_ema' if args.weights=='ema' else 'actor'], env.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.sensor_noise == 'on' and env.camera_noise is None:
        raise ValueError('Noise-on evaluation requires the declared source camera noise pipeline')
    if args.sensor_noise == 'off':
        env.camera_noise = None
    reports = []
    body_names = config['physical_model']['tracked_body_names']
    torso_index = body_names.index('torso_link')
    with torch.no_grad():
        for clip, start in enumerate(env.starts):
            frame = torch.tensor([int(start)],device=env.device,dtype=torch.int32)
            noise = {name:torch.zeros_like(value) for name,value in env.sample_reset_noise(1).items()}
            state = env.reset(frame,noise)
            initial_state_error=dict(
                qpos_reference_absolute_max=float((state['qpos']-env.ref_qpos[frame]).abs().max()),
                qvel_reference_absolute_max=float((state['qvel']-env.ref_qvel[frame]).abs().max()),
                tracked_body_world_position_error_max_m=float(torch.linalg.vector_norm(
                    state['xpos'][:,env.body_ids]-env.ref_body_pos[frame],dim=-1).max()))
            rows=[]
            substep_rows=[]
            world_errors=[]
            terminations=[]
            body_positions=[]
            up_dots=[]
            requested = int(env.lengths[clip])-1 if args.maximum_controls is None else args.maximum_controls
            reason='requested_controls'
            for control in range(requested):
                actual = state['xpos'][:,env.body_ids]
                reference = env.ref_body_pos[state['frame']]
                world_errors.append(torch.linalg.vector_norm(actual-reference,dim=-1)[0].cpu().numpy())
                body_positions.append(actual[0].cpu().numpy())
                terminations.append(bool(env.terminated(state).item()))
                quaternion = state['xquat'][0,env.torso_id]
                up_dots.append(float((1-2*(quaternion[1]**2+quaternion[2]**2)).item()))
                observation_noise = {name:torch.zeros_like(value) for name,value in env.sample_observation_noise(1).items()}
                history,goal = env.policy_obs(state,observation_noise['goal'])
                action = actor(history,goal) if args.weights == 'official' else nets.actor_apply(actor,history,goal)
                if args.record_substeps:
                    stepped,substeps=env.step_recorded(state,action,observation_noise['proprio'])
                    substep_rows.append({name:value[0].cpu().numpy() for name,value in substeps.items()})
                else:
                    stepped = env.step(state,action,observation_noise['proprio'])
                done = bool(env.terminated(stepped).item())
                ended = bool(env.clip_ended(stepped).item())
                rows.append(dict(qpos=state['qpos'][0].cpu().numpy(),qvel=state['qvel'][0].cpu().numpy(),
                    action=stepped['prev_a'][0].cpu().numpy(),policy_action=action[0].cpu().numpy(),depth=history[0,744:].cpu().numpy(),
                    frame=int(state['frame'][0].item()),tracking_terminated=done,clip_ended=ended))
                state=stepped
                if done or ended:
                    reason='tracking_termination' if done else 'reference_complete'
                    break
            actual = state['xpos'][:,env.body_ids]
            reference = env.ref_body_pos[state['frame']]
            world_errors.append(torch.linalg.vector_norm(actual-reference,dim=-1)[0].cpu().numpy())
            body_positions.append(actual[0].cpu().numpy())
            terminations.append(bool(env.terminated(state).item()))
            quaternion = state['xquat'][0,env.torso_id]
            up_dots.append(float((1-2*(quaternion[1]**2+quaternion[2]**2)).item()))
            reading = summarize_source_chain(world_errors,torso_index,terminations,
                reason=='reference_complete',len(rows),requested)
            output=args.output/f'clip_{clip:02d}' 
            output.mkdir()
            substep_arrays={}
            if args.record_substeps:
                substep_arrays={f'substep_{name}':np.stack([row[name] for row in substep_rows])
                                for name in substep_rows[0]}
                substep_arrays['substep_time_s']=(np.arange(len(rows))[:,None]*4+np.arange(4)[None])*env.m.opt.timestep
            np.savez(output/'trajectory.npz',**substep_arrays,**{key:np.asarray([row[key] for row in rows]) for key in rows[0]},
                final_qpos=state['qpos'].cpu().numpy(),final_qvel=state['qvel'].cpu().numpy(),
                final_frame=state['frame'].cpu().numpy(),world_body_error_m=np.asarray(world_errors),
                torso_error_m=np.asarray(world_errors)[:,torso_index],body_world_position=np.asarray(body_positions),
                torso_updot=np.asarray(up_dots),state_tracking_terminated=np.asarray(terminations))
            reports.append(dict(action_clipped_components=sum(int(np.count_nonzero(row['policy_action'] != row['action'])) for row in rows),clip=clip,start_frame=int(start),stop_reason=reason,initial_state_error=initial_state_error,**reading))
    (args.output/'result.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),weights=args.weights,
        policy_implementation='official release actor with stored std' if args.weights == 'official' else 'native actor',
        action_execution='Environment applies its declared normalized action domain; raw policy_action and executed action are recorded',
        completed_updates=checkpoint['iter'] if args.weights == 'official' else checkpoint['completed_updates'],sensor_noise=args.sensor_noise,seed=args.seed,
        reset=bindings['reset'],reset_noise='zero',proprioception_noise='zero',goal_noise='zero',
        tracked_body_names=body_names,
        substep_recording=dict(enabled=args.record_substeps,sampling_hz=1/env.m.opt.timestep,
            time_support='Each physics input t_k before its implicitfast update',
            body_names=[env.m.body(index).name for index in range(env.m.nbody)],
            torque_joint_names=[env.m.joint(int(env.m.actuator_trnid[index,0])).name for index in range(env.nu)],
            wrench_units='world force N and moment about world origin N m',torque_units='executed generalized actuator torque N m'),
        strict_passes=sum(row['strict_pass'] for row in reports),
        perception=env.perception_contract,
        observation=env.observation_contract,reports=reports),indent=2))


if __name__=='__main__':
    main()
