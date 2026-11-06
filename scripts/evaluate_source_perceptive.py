"""Terminal-inclusive full-reference and local-window source-task physical reads."""
import argparse
from dataclasses import replace
import json
import pickle
from pathlib import Path

import numpy as np
from nablamimic.source_evaluation_metrics import summarize_source_chain


CAMERA_INPUT_DEFINITIONS = {
    'measured': 'Generated measured depth with the selected camera noise',
    'constant-far': ('The environment generates the complete measured camera and noise, then supplies a constant '
        'processed image of 1.0 (source normalized 2m/far), before the frozen checkpoint observation normalizer. '
        'Actor and recorded depth consume this same observation'),
    'self-only': ('Raycast only the original visible geoms with body_id != 0 using actual dynamic body poses. '
        'Static world camera occluders are excluded. Physical collision geometry, geom IDs, and the original '
        'checkpoint perception binding are retained. Miss conversion, noise and image processing run normally'),
}


def self_only_camera_geometry(geometry, device):
    """Construct a separate sensor owner retaining original non-world geom IDs."""
    from nablamimic.mujoco_source_camera import MuJoCoRayCamera
    if isinstance(geometry,MuJoCoRayCamera):
        return MuJoCoRayCamera(geometry.adapter,include_static=False)
    geoms = tuple(geom for geom in geometry.geoms if geom.body_id != 0)
    if not geoms:
        raise ValueError('Self-only camera requires visible non-world geometry')
    backend = None
    if geometry.cuda_backend is not None:
        from nablaengine.depth_cuda import CudaDepthGeometry
        backend = CudaDepthGeometry(geoms,geometry.body_count,device)
    return replace(geometry,geoms=geoms,cuda_backend=backend)


def set_evaluation_conditions(config, reset_condition):
    """Own inference-only gradient and explicitly selected reset conditions."""
    if not {'source_motion_start_height_offset','source_ensure_link_below_zero_ground'} <= config.keys():
        raise ValueError('Evaluation requires a current resolved source reset configuration')
    config.update(source_depth_gradient=False,
                  source_motion_start_height_offset=.1 if reset_condition == 'source-reset' else 0.,
                  source_ensure_link_below_zero_ground=reset_condition == 'source-reset')


def resolve_camera_input(config, checkpoint_observation, override):
    """Validate training provenance before selecting an explicit inference condition."""
    trained=config['source_camera_input']
    if trained not in ('measured','constant-far'):
        raise ValueError('Training camera input must be measured or constant-far')
    if checkpoint_observation['camera_input'] != trained:
        raise ValueError('Checkpoint training camera input differs from the resolved configuration')
    if override is not None and override not in CAMERA_INPUT_DEFINITIONS:
        raise ValueError('Unknown inference camera input override')
    effective=trained if override is None else override
    return dict(training_camera_input=trained,inference_camera_input_override=override,
                camera_input=effective,source_camera_input='measured' if effective=='self-only' else effective)


def select_motion_indices(indices, motion_count):
    """Select original clip IDs in the complete library's traversal order."""
    if indices is None:
        return list(range(motion_count))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Motion selection must contain distinct clip indices')
    if any(index < 0 or index >= motion_count for index in indices):
        raise ValueError('Motion index is outside the complete reference library')
    return sorted(indices)


def source_window(length, start_fraction, maximum_controls):
    """Select one supported control interval within a reference clip."""
    if not np.isfinite(start_fraction) or not 0 <= start_fraction < 1:
        raise ValueError('Start fraction must be finite and in [0,1)')
    if length < 2 or (maximum_controls is not None and maximum_controls < 1):
        raise ValueError('A reference and requested window must support at least one control')
    offset = int(np.floor((length-1)*start_fraction))
    remaining = length-1-offset
    requested = remaining if maximum_controls is None else min(maximum_controls, remaining)
    return dict(start_offset=offset, remaining_controls=remaining, requested_controls=requested)


def summarize_window(world_errors, torso_index, terminations, clip_end_reached, controls, window):
    reading = summarize_source_chain(world_errors, torso_index, terminations,
        clip_end_reached and window['start_offset'] == 0 and not terminations[-1],
        controls, window['requested_controls'])
    complete = controls == window['requested_controls']
    reading.update(clip_end_reached=bool(clip_end_reached), requested_window_complete=complete,
        window_fidelity_pass=bool(complete and not reading['tracking_terminated']
            and reading['mean_world_body_error_m'] <= .1 and reading['mean_torso_error_m'] <= .1))
    return reading


def reference_window_activity(positions, torso_index, control_dt_s=.02):
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[1:] != (14,3) or len(positions) < 2 or not np.isfinite(positions).all():
        raise ValueError('Reference activity requires finite fourteen-body positions across a control interval')
    return dict(reference_world_body_speed_mean_m_s=float(
        np.linalg.norm(np.diff(positions, axis=0)/control_dt_s, axis=-1).mean()),
        reference_torso_displacement_m=float(np.linalg.norm(positions[-1,torso_index]-positions[0,torso_index])))


def offset_initial_state(env, state, xy_offset, yaw_offset):
    """Apply an evaluation-only world pose offset, then rebuild FK and history."""
    if tuple(xy_offset)==(0.,0.) and yaw_offset==0.:
        return state
    import torch
    from nablamimic.torch_env import quat_mul
    qpos=state['qpos'].clone()
    qpos[:,:2]+=qpos.new_tensor(xy_offset)
    half=qpos.new_tensor(yaw_offset/2.)
    yaw=torch.stack((half.cos(),half*0,half*0,half.sin()))
    qpos[:,3:7]=quat_mul(yaw,qpos[:,3:7])
    compact={**env.compact(state),'qpos':qpos}
    shifted=env.restore(compact)
    shifted['hist']=env.proprio(shifted)[:,None].expand_as(state['hist']).clone()
    return shifted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backend',choices=('engine','mujoco'),default='engine')
    parser.add_argument('--ray-backend',choices=('native','mujoco'),default='native')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--weights', choices=('ema','live','official'), required=True)
    parser.add_argument('--maximum-controls', type=int, help='Default is all controls remaining after the selected start')
    parser.add_argument('--start-fraction', type=float, default=0., help='Fraction of each clip control span at reset')
    parser.add_argument('--motion-index', type=int, action='append',
                        help='Original clip ID to evaluate, repeat to select several, default all in library order')
    parser.add_argument('--sensor-noise', choices=('on','off'), required=True)
    parser.add_argument('--camera-input', choices=tuple(CAMERA_INPUT_DEFINITIONS), default=None,
                        help='Explicit inference override. Default preserves the checkpoint training input mode')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--record-substeps', action='store_true')
    parser.add_argument('--push-force-world', type=float, nargs=3, default=[0.,0.,0.], metavar=('X','Y','Z'))
    parser.add_argument('--push-start-control', type=int, default=150)
    parser.add_argument('--push-duration-controls', type=int, default=5)
    parser.add_argument('--initial-xy-offset',type=float,nargs=2,default=[0.,0.],metavar=('X','Y'))
    parser.add_argument('--initial-yaw-offset',type=float,default=0.,help='World Z rotation in radians, left-multiplied onto the root quaternion')
    parser.add_argument('--reset-condition', choices=('exact-reference','source-reset'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_controls is not None and args.maximum_controls < 1:
        raise ValueError("Requested source rollout must contain at least one control")
    if not np.isfinite(args.start_fraction) or not 0 <= args.start_fraction < 1:
        raise ValueError('Start fraction must be finite and in [0,1)')
    if not np.isfinite([*args.initial_xy_offset,args.initial_yaw_offset]).all():
        raise ValueError('Initial pose offsets must be finite')
    from nablamimic.source_force_pulse import TorsoWorldForcePulse
    # Validate before loading any policy or physics artifact.
    pulse = TorsoWorldForcePulse(1, tuple(args.push_force_world),
                                args.push_start_control, args.push_duration_controls)
    push_enabled = any(pulse.force_world)
    if push_enabled and args.backend != 'mujoco':
        raise ValueError('Nonzero world-force pulses require the MuJoCo backend')
    if push_enabled and not args.record_substeps:
        raise ValueError('World-force pulses require --record-substeps for 200 Hz force evidence')
    import torch
    from nablamimic.torch_train import build_environment
    from nablamimic import torch_nets as nets

    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    set_evaluation_conditions(config,args.reset_condition)
    for key in ('terrain_model','terrain_model_contract'):
        config[key] = Path(config[key])
    env, bindings = build_environment(argparse.Namespace(**config),evaluation_backend=args.backend,
        ray_backend=args.ray_backend,device=args.device)
    if args.backend == 'mujoco':
        env.mujoco_adapter.applied_body_wrench_world = TorsoWorldForcePulse(
            int(env.torso_id), pulse.force_world, pulse.start_control, pulse.duration_controls)
    selected_motion_indices = select_motion_indices(args.motion_index,len(env.starts))
    if args.weights == 'official':
        from nablamimic.official_source_policy import OfficialSourcePolicy
        if env.observation_contract['task'] != 'instinct_perceptive_native':
            raise ValueError('Official source policy requires the source perceptive task')
        if env.cfg.residual_action:
            raise ValueError('Official source policy requires absolute actions')
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        actor = OfficialSourcePolicy(checkpoint).to(env.device)
        camera_condition=resolve_camera_input(config,dict(camera_input='measured'),args.camera_input)
    else:
        actor = None
    if args.weights != 'official':
        with args.checkpoint.open('rb') as handle:
            checkpoint = pickle.load(handle)
        camera_condition=resolve_camera_input(config,checkpoint['observation'],args.camera_input)
        if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
            raise ValueError('Source actor observation or physical geometry binding differs')
        actor = nets.tree_from_jax(checkpoint['actor_ema' if args.weights=='ema' else 'actor'], env.device)
    training_observation=env.observation_contract
    if args.camera_input is not None:
        env.cfg=replace(env.cfg,source_camera_input=camera_condition['source_camera_input'])
    effective_camera_input=camera_condition['camera_input']
    excluded_geom_ids = []
    camera_input_definition = CAMERA_INPUT_DEFINITIONS[effective_camera_input]
    if effective_camera_input == 'self-only':
        original_geoms = env.depth_geometry.geoms
        env.depth_geometry = self_only_camera_geometry(env.depth_geometry,env.device)
        retained_ids = {geom.geom_id for geom in env.depth_geometry.geoms}
        excluded_geom_ids = [geom.geom_id for geom in original_geoms if geom.geom_id not in retained_ids]
        if args.ray_backend == 'mujoco':
            camera_input_definition = ('MuJoCo camera mj_multiRay/mj_ray with flg_static=False excludes '
                'camera geoms whose body_weldid is zero. Physical collision geometry and original geom IDs '
                'are retained. Actual dynamic poses, miss conversion, noise and image processing run normally.')
    camera_geometry = dict(retained_geom_ids=[geom.geom_id for geom in env.depth_geometry.geoms],
        excluded_static_geom_ids=excluded_geom_ids,
        body_count=env.depth_geometry.body_count,physical_geoms=env.m.ngeom,physical_pairs=env.m.npair)
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
        for clip in selected_motion_indices:
            start = env.starts[clip]
            window = source_window(int(env.lengths[clip]), args.start_fraction, args.maximum_controls)
            actual_start = int(start)+window['start_offset']
            reference_activity = reference_window_activity(env.ref_body_pos[
                actual_start:actual_start+window['requested_controls']+1].cpu().numpy(),torso_index)
            frame = torch.tensor([actual_start],device=env.device,dtype=torch.int32)
            noise = {name:torch.zeros_like(value) for name,value in env.sample_reset_noise(1).items()}
            if args.backend=='mujoco':
                env.mujoco_adapter.reset_clock()
            state = env.reset(frame,noise)
            state = offset_initial_state(env,state,args.initial_xy_offset,args.initial_yaw_offset)
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
            requested = window['requested_controls']
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
                policy_observation,goal = env.policy_obs(state,observation_noise['goal'])
                policy_observation,goal=policy_observation.to(torch.float32),goal.to(torch.float32)
                action = actor(policy_observation,goal) if args.weights == 'official' else nets.actor_apply(actor,policy_observation,goal)
                if args.record_substeps:
                    stepped,substeps=env.step_recorded(state,action,observation_noise['proprio'],
                        measurement_backend=env.mujoco_adapter if args.backend=='mujoco' else None)
                    substep_rows.append({name:value[0].cpu().numpy() for name,value in substeps.items()})
                else:
                    stepped = env.step(state,action,observation_noise['proprio'])
                done = bool(env.terminated(stepped).item())
                ended = bool(env.clip_ended(stepped).item())
                rows.append(dict(qpos=state['qpos'][0].cpu().numpy(),qvel=state['qvel'][0].cpu().numpy(),
                    action=stepped['prev_a'][0].cpu().numpy(),policy_action=action[0].cpu().numpy(),depth=policy_observation[0,744:].cpu().numpy(),
                    frame=int(state['frame'][0].item()),tracking_terminated=done,clip_ended=ended))
                state=stepped
                if done or ended:
                    reason='tracking_termination' if done else (
                        'reference_complete' if window['start_offset'] == 0 else 'clip_end_reached')
                    break
            actual = state['xpos'][:,env.body_ids]
            reference = env.ref_body_pos[state['frame']]
            world_errors.append(torch.linalg.vector_norm(actual-reference,dim=-1)[0].cpu().numpy())
            body_positions.append(actual[0].cpu().numpy())
            terminations.append(bool(env.terminated(state).item()))
            quaternion = state['xquat'][0,env.torso_id]
            up_dots.append(float((1-2*(quaternion[1]**2+quaternion[2]**2)).item()))
            reading = summarize_window(world_errors,torso_index,terminations,
                ended,len(rows),window)
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
            reports.append(dict(action_clipped_components=sum(int(np.count_nonzero(row['policy_action'] != row['action'])) for row in rows),
                clip=clip,clip_start_frame=int(start),start_frame=actual_start,start_fraction=args.start_fraction,
                start_offset=window['start_offset'],remaining_controls=window['remaining_controls'],
                stop_reason=reason,initial_state_error=initial_state_error,**reference_activity,**reading))
    (args.output/'result.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),weights=args.weights,
        physics_backend=args.backend,ray_backend=args.ray_backend,
        push_force=dict(enabled=push_enabled,force_world_N=list(pulse.force_world),
            torque_world_Nm=[0.,0.,0.],body_name=env.m.body(int(env.torso_id)).name,
            application_point='torso body center of mass',
            clock='Episode control index from reset, independent of reference start frame; no reset at pulse',
            start_control=pulse.start_control,duration_controls=pulse.duration_controls,
            start_substep=4*pulse.start_control,duration_substeps=4*pulse.duration_controls,
            physics_dt_s=float(env.m.opt.timestep),
            requested_impulse_world_Ns=[v*4*pulse.duration_controls*env.m.opt.timestep for v in pulse.force_world],
            applied_wrench_measured=args.backend=='mujoco' and args.record_substeps,
            actual_time_support='Executed substeps only; an early termination may precede or truncate pulse',
            applied_wrench_definition='World force N then torque about body COM N m; substep_applied_body_wrench_world and substep_body_com_world',
            unavailable_reason=None if args.backend=='mujoco' and args.record_substeps else 'Applied wrench was not measured'),
        physics=bindings['evaluation_physics'] if args.backend=='mujoco' else bindings['engine'],
        policy_implementation='official release actor with stored std' if args.weights == 'official' else 'native actor',
        action_execution='Environment applies its declared normalized action domain; raw policy_action and executed action are recorded',
        completed_updates=checkpoint['iter'] if args.weights == 'official' else checkpoint['completed_updates'],sensor_noise=args.sensor_noise,seed=args.seed,
        camera_input=effective_camera_input,training_camera_input=camera_condition['training_camera_input'],
        inference_camera_input_override=camera_condition['inference_camera_input_override'],
        camera_input_definition=camera_input_definition,camera_geometry=camera_geometry,
        initial_pose_offset=dict(world_xy_m=args.initial_xy_offset,world_yaw_rad=args.initial_yaw_offset,
            definition='After reset, translate root world XY and left-multiply root WXYZ quaternion by world Z yaw; preserve qvel coordinates, rebuild FK and zero-noise proprioception history'),
        reset=bindings['reset'],reset_noise='zero',proprioception_noise='zero',goal_noise='zero',
        selected_motion_indices=selected_motion_indices,
        start_fraction=args.start_fraction, maximum_controls=args.maximum_controls,
        window_protocol=dict(start_offset='floor((clip_length-1)*start_fraction)',
            requested_controls='min(maximum_controls or remaining_controls, remaining_controls)',
            requested_window_complete='All requested controls executed, including terminal state',
            window_fidelity_pass='Requested window complete, no source tracking termination, mean fourteen-body and torso error each <= .1m',
            reference_activity_support='All requested reference controls and both interval endpoints, even if execution terminates early',
            reference_complete='Executed from clip frame zero through its final reference frame'),
        tracked_body_names=body_names,
        substep_recording=dict(enabled=args.record_substeps,sampling_hz=1/env.m.opt.timestep,
            time_support='Each physics input t_k before its implicitfast update',
            body_names=[env.m.body(index).name for index in range(env.m.nbody)],
            torque_joint_names=[env.m.joint(int(env.m.actuator_trnid[index,0])).name for index in range(env.nu)],
            wrench_units='world force N and moment about world origin N m',torque_units='executed generalized actuator torque N m'),
        strict_passes=sum(row['strict_pass'] for row in reports),
        perception=env.perception_contract,
        observation=training_observation,inference_observation=env.observation_contract,reports=reports),indent=2))


if __name__=='__main__':
    main()
