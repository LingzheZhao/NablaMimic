"""Read same-reference-state source observations across physics and ray backends.

Each selected global frame gets one batch-one normal source reset per physics
backend. The two MuJoCo cameras consume the same double physical state and
restored Torch RNG. Numerical differences are measured without acceptance
thresholds. No physical control is executed.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np

from evaluate_source_perceptive import set_evaluation_conditions


def delta_statistics(left, right, units, mask=None):
    left,right = np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError('Compared observations must have matching shapes')
    delta = right-left
    if mask is not None:
        delta = delta[mask]
    finite = bool(np.isfinite(delta).all())
    return dict(units=units,elements=int(delta.size),finite=finite,
        max_absolute=float(np.abs(delta).max()) if delta.size and finite else None,
        rms=float(np.sqrt(np.mean(delta*delta))) if delta.size and finite else None)


def compare_observations(left, right):
    """Report each physical or policy quantity in its own units."""
    fields = {
        'qpos_root_position': ('qpos',slice(0,3),'m'),
        'qpos_root_quaternion': ('qpos',slice(3,7),'dimensionless WXYZ components'),
        'qpos_joints': ('qpos',slice(7,None),'rad'),
        'qvel_root_linear': ('qvel',slice(0,3),'m/s world'),
        'qvel_root_angular': ('qvel',slice(3,6),'rad/s torso'),
        'qvel_joints': ('qvel',slice(6,None),'rad/s'),
        'fk_position': ('fk_position',slice(None),'m world'),
        'fk_quaternion': ('fk_quaternion',slice(None),'dimensionless WXYZ components'),
        'subtree_com': ('subtree_com',slice(None),'m world'),
        'cvel_angular': ('cvel',slice(0,3),'rad/s spatial cvel angular component'),
        'cvel_linear': ('cvel',slice(3,6),'m/s spatial cvel linear component'),
        'proprio_gravity': ('proprio',slice(0,3),'dimensionless torso'),
        'proprio_angular_velocity': ('proprio',slice(3,6),'rad/s torso'),
        'proprio_joint_position': ('proprio',slice(6,35),'rad relative to defaults'),
        'proprio_joint_velocity': ('proprio',slice(35,64),'rad/s'),
        'proprio_last_action': ('proprio',slice(64,93),'normalized action'),
        'goal_joint_position': ('goal',slice(0,290),'rad relative to defaults'),
        'goal_joint_velocity': ('goal',slice(290,580),'rad/s'),
        'goal_position': ('goal',slice(580,610),'m current reference torso'),
        'goal_rotation': ('goal',slice(610,670),'dimensionless tangent-normal components'),
        'processed_depth': ('processed_depth',slice(None),'normalized optical depth [0,1]'),
        'policy_action': ('policy_action',slice(None),'normalized action'),
        'bounded_action': ('bounded_action',slice(None),'normalized action'),
        'control_target': ('control_target',slice(None),'rad PD target'),
        'ray_origin': ('ray_origins',slice(None),'m world'),
        'ray_direction': ('ray_directions',slice(None),'dimensionless world unit vector'),
    }
    for name,(selection,units) in {
        'gravity': (slice(0,24),'dimensionless torso'),
        'angular_velocity': (slice(24,48),'rad/s torso'),
        'joint_position': (slice(48,280),'rad relative to defaults'),
        'joint_velocity': (slice(280,512),'rad/s'),
        'last_action': (slice(512,744),'normalized action'),
    }.items():
        fields['actor_history_'+name] = ('policy_observation',selection,units)
    for name in ('joint_position','joint_velocity','position','rotation'):
        _,selection,units = fields['goal_'+name]
        fields['actor_goal_'+name] = ('policy_goal',selection,units)
    measurements = {name:delta_statistics(left[key][...,selection],right[key][...,selection],units)
                    for name,(key,selection,units) in fields.items()}
    a,b = left['ray_geometry_ids'],right['ray_geometry_ids']
    both = (a>=0)&(b>=0)
    same = both&(a==b)
    changed = both&(a!=b)
    measurements['ray_selection'] = dict(rays=int(a.size),
        hit_mismatches=int(np.count_nonzero((a>=0)!=(b>=0))),
        geometry_id_mismatches=int(np.count_nonzero(a!=b)),
        different_geom_both_hit=int(changed.sum()),same_geom_both_hit=int(same.sum()),
        both_hit_range=delta_statistics(left['ray_ranges'],right['ray_ranges'],'m',both),
        same_geom_range=delta_statistics(left['ray_ranges'],right['ray_ranges'],'m',same),
        different_geom_range=delta_statistics(left['ray_ranges'],right['ray_ranges'],'m',changed),
        surface_selection_measure='Original geom IDs. Triangle changes within one geom are not observed.')
    return measurements


class RecordingRays:
    """Observe the existing camera call without regenerating rays or image processing."""
    def __init__(self, geometry):
        self.geometry = geometry
        self.arrays = None

    def raycast(self, body_positions, body_quaternions, origins, directions, **limits):
        ranges,ids = self.geometry.raycast(body_positions,body_quaternions,origins,directions,**limits)
        self.arrays = {key:value.detach().cpu().numpy().copy() for key,value in (
            ('ray_ranges',ranges),('ray_geometry_ids',ids),('ray_origins',origins),('ray_directions',directions))}
        return ranges,ids


def capture_observation(env, state, geometry, actor, rng):
    import torch
    from nablamimic import torch_nets as nets

    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state_all(rng[1])
    recorder = RecordingRays(geometry)
    previous = env.depth_geometry
    env.depth_geometry = recorder
    try:
        noise = {name:torch.zeros_like(value) for name,value in env.sample_observation_noise(1).items()}
        observation,goal = env.policy_obs(state,noise['goal'])
        policy_observation,policy_goal = observation.to(torch.float32),goal.to(torch.float32)
        action = nets.actor_apply(actor,policy_observation,policy_goal)
        bounded = env.bound_action(action)
        tensors = dict(qpos=state['qpos'],qvel=state['qvel'],fk_position=state['xpos'],
            fk_quaternion=state['xquat'],cvel=state['cvel'],subtree_com=state['subtree_com'],
            proprio=env.proprio(state),goal=goal,policy_goal=policy_goal,
            policy_observation=policy_observation,processed_depth=policy_observation[:,744:],
            policy_action=action,bounded_action=bounded,control_target=env.ctrl_of(bounded,state['frame']))
        arrays = {name:value.detach().cpu().numpy().copy() for name,value in tensors.items()}
        arrays.update(recorder.arrays)
        if hasattr(env,'mujoco_adapter'):
            arrays.update({f'mujoco_data_{name}':np.stack([getattr(data,name).copy()
                for data in env.mujoco_adapter.data_batch]) for name in ('qpos','qvel','qacc_warmstart')})
        return arrays
    finally:
        env.depth_geometry = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[0,100,500,1000,2000,3000,4000])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--code-revision',required=True)
    parser.add_argument('--engine-code-revision',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for name in ('config','checkpoint','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    os.chdir(args.output)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key,relative in dict(DK_CACHE_DIR='cache/native',XDG_CACHE_HOME='cache',
                             CUDA_CACHE_PATH='cache/cuda',TMPDIR='scratch').items():
        path = args.output/relative
        path.mkdir(parents=True,exist_ok=True)
        os.environ[key] = str(path)
    status = dict(state='started',code_revision=args.code_revision,engine_code_revision=args.engine_code_revision,
        arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},physics_controls=0)
    def write_status():
        (args.output/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
    def fail(kind,value,tb):
        status.update(state='failed',failure_type=kind.__name__,failure=str(value),
                      traceback=''.join(traceback.format_exception(kind,value,tb)))
        write_status()
        sys.__excepthook__(kind,value,tb)
    write_status()
    sys.excepthook = fail
    import torch
    from nablamimic import torch_nets as nets
    from nablamimic.torch_train import build_environment
    from nablamimic.mujoco_source_camera import MuJoCoRayCamera

    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('The engine/native comparison requires a provisioned CUDA environment')
    config = json.loads(args.config.read_text())
    set_evaluation_conditions(config,'source-reset')
    for key in ('terrain_model','terrain_model_contract'):
        config[key] = Path(config[key])
    if config['task_inputs'] != 'instinct_perceptive_native' or config['clips'] != ['all']:
        raise ValueError('Checker requires the complete prepared source reference library')
    (args.output/'config.runtime.json').write_text(json.dumps({
        key:str(value) if isinstance(value,Path) else value for key,value in config.items()},indent=2)+'\n')
    reference_contract = json.loads((Path(config['motion_lib'])/'screening.json').read_text())
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    environments,bindings = {},{}
    for backend in ('engine','mujoco'):
        env,binding = build_environment(argparse.Namespace(**config),evaluation_backend=backend,
            ray_backend='native',device=args.device)
        if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
            raise ValueError('Frozen checkpoint binding differs from the constructed environment')
        env.camera_noise = None
        environments[backend],bindings[backend] = env,binding
    engine,mjc = environments['engine'],environments['mujoco']
    if len(set(args.frames)) != len(args.frames) or any(frame < 0 or frame >= engine.n_frames for frame in args.frames):
        raise ValueError('Requested global reference frames must be distinct and within the complete library')
    if not np.array_equal(engine.starts,mjc.starts) or engine.n_frames != mjc.n_frames:
        raise ValueError('Physics backends must use the same reference frame binding')
    if any(reference_contract['target_artifact_id'] != value['physical_model']['artifact_id']
           for value in bindings.values()):
        raise ValueError('Reference artifact contract differs from the constructed physical scene')
    native_geometry = mjc.depth_geometry
    mujoco_geometry = MuJoCoRayCamera(mjc.mujoco_adapter)
    mjc.camera_state_sync = lambda state:mjc.mujoco_adapter.synchronized_data(state['qpos'],state['qvel'],state['warm'])
    actor = nets.tree_from_jax(checkpoint['actor_ema'],args.device)
    frozen = [value.clone() for value in nets.tree_leaves(actor)]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    normalizer = checkpoint['actor_ema']['source_norm']
    protocol = dict(config=str(args.config),checkpoint=str(args.checkpoint),weights='ema',
        model=str(config['terrain_model']),model_contract=str(config['terrain_model_contract']),
        reference_library=str(config['motion_lib']),reference_artifact_id=reference_contract['artifact_id'],
        completed_updates=checkpoint['completed_updates'],code_revision=args.code_revision,
        engine_code_revision=args.engine_code_revision,frames=args.frames,seed=args.seed,
        torch_version=torch.__version__,gpu_model=torch.cuda.get_device_name(args.device),
        torch_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        torch_cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        reset='Batch-one source reset separately at each selected global frame',
        reset_conditions=bindings['engine']['reset'],additive_noise='zero',camera_noise='off',physics_controls=0,
        actor_normalizer=dict(mean_shape=list(normalizer['mean'].shape),variance_shape=list(normalizer['variance'].shape),
            count=int(normalizer['count']),epsilon=.01),
        variants=['engine_native','mujoco_native','mujoco_mujoco'],
        camera_pair='Same MuJoCo environment, same double state object, restored CPU/CUDA Torch RNG before each observation',
        observation_generation='Existing SourcePerceptiveEnv.policy_obs and frozen actor. Same explicit float32 actor boundary as evaluation.',
        deltas='Right minus left, componentwise max absolute and RMS, with separate units. No numerical acceptance thresholds.',
        bindings=bindings)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows = []
    with torch.no_grad():
        for frame in args.frames:
            states = {}
            for backend,env in environments.items():
                frame_tensor = torch.tensor([frame],dtype=torch.int32,device=env.device)
                noise = {name:torch.zeros_like(value) for name,value in env.sample_reset_noise(1).items()}
                states[backend] = env.reset(frame_tensor,noise)
            rng = (torch.get_rng_state().clone(),[value.clone() for value in torch.cuda.get_rng_state_all()])
            snapshots = {}
            for name,env,state,geometry in (
                ('engine_native',engine,states['engine'],engine.depth_geometry),
                ('mujoco_native',mjc,states['mujoco'],native_geometry),
                ('mujoco_mujoco',mjc,states['mujoco'],mujoco_geometry)):
                snapshots[name] = capture_observation(env,state,geometry,actor,rng)
                np.savez(args.output/f'frame_{frame:06d}_{name}.npz',**snapshots[name])
            a,b = snapshots['mujoco_native'],snapshots['mujoco_mujoco']
            same_state = all(a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
                for key in ('qpos','qvel','mujoco_data_qpos','mujoco_data_qvel','mujoco_data_qacc_warmstart'))
            if not same_state:
                raise ValueError('MuJoCo cameras did not consume the same double physical state')
            comparisons = {name:compare_observations(snapshots[left],snapshots[right])
                for name,left,right in (
                    ('physics_engine_to_mujoco_native','engine_native','mujoco_native'),
                    ('camera_native_to_mujoco_same_physics','mujoco_native','mujoco_mujoco'),
                    ('full_engine_to_mujoco','engine_native','mujoco_mujoco'))}
            row = dict(global_frame=frame,clip=int(np.searchsorted(engine.starts,frame,side='right')-1),
                same_mujoco_physical_state_bytewise=True,
                dtypes={name:{key:str(value.dtype) for key,value in snapshot.items()} for name,snapshot in snapshots.items()},
                comparisons=comparisons)
            rows.append(row)
            (args.output/'result.json').write_text(json.dumps(dict(protocol=protocol,frames=rows),indent=2,allow_nan=False)+'\n')
            status.update(completed_frames=[row['global_frame'] for row in rows])
            write_status()
    if not all(torch.equal(before,after) for before,after in zip(frozen,nets.tree_leaves(actor))):
        raise ValueError('Frozen actor or empirical normalization changed during observation checks')
    status.update(state='completed',completed_frames=args.frames,frozen_actor_and_normalizer_unchanged=True)
    write_status()


if __name__=='__main__':
    main()
