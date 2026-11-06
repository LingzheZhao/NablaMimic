"""Generated matched-camera checkpoints and terminal-inclusive physical reads."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_source_camera_training import (
    GROUPS, compare_arrays, compare_configurations, compare_evaluations, load_checkpoint,
)
sys.path.pop(0)
from nablamimic.source_evaluation_metrics import summarize_source_chain


def generated_checkpoint(mode, updates=2250):
    actor = dict(enc={'weight':np.arange(6,dtype=np.float32).reshape(2,3)},
                 source_norm=dict(mean=np.zeros(4,np.float32),variance=np.ones(4,np.float32)))
    return dict(actor=actor,actor_ema=deepcopy(actor),critic=[(np.ones((3,1),np.float32),np.zeros(1,np.float32))],
        cnorm=(np.zeros(3,np.float32),np.ones(3,np.float32)),completed_updates=updates,iter=updates,
        ema_updates=updates,training_framework='torch',observation=dict(camera_input=mode),
        perception=dict(physical_scene_artifact_id='generated-scene',terrain_artifact_id='generated-terrain'))


def generated_readings(tmp_path):
    bodies = tuple(f'body_{i}' for i in range(13))+('torso_link',)
    reference = SimpleNamespace(names=[f'motion_{i:02d}' for i in range(10)],bodies=bodies,
        fps=50.,starts=np.arange(10)*6,last=np.repeat(np.arange(10)*6+5,6),
        body_pos=np.arange(60*14*3,dtype=np.float32).reshape(60,14,3)/1000)
    roots = {group:tmp_path/group for group in GROUPS}
    paths = {group:tmp_path/f'{group}.pkl' for group in GROUPS}
    checkpoints = {group:generated_checkpoint(group) for group in GROUPS}
    for group in GROUPS:
        for clip in range(10):
            root = roots[group]/f'motion-{clip:02d}'
            trajectory = root/f'clip_{clip:02d}'
            trajectory.mkdir(parents=True)
            count = (3 if group=='measured' else 4) if clip==2 else 5
            start = clip*6
            order = np.arange(14)[::-1]
            positions = reference.body_pos[start:start+count+1,order].copy()
            offset = np.full(count+1,.02 if group=='measured' else .08)
            if clip==2 and group=='measured':
                offset[-1] = .3
            positions[:,:,0] += offset[:,None]
            errors = np.linalg.norm(positions-reference.body_pos[start:start+count+1,order],axis=-1)
            terminated = np.zeros(count+1,bool)
            terminated[-1] = count<5
            reading = summarize_source_chain(errors,0,terminated,count==5,count,5)
            reading.update(clip=clip,clip_start_frame=start,start_frame=start,start_fraction=0.,start_offset=0,
                remaining_controls=5,action_clipped_components=0,
                stop_reason='reference_complete' if count==5 else 'tracking_termination')
            checkpoint = checkpoints[group]
            evaluation = dict(checkpoint=str(paths[group]),completed_updates=2250,weights='ema',physics_backend='engine',
                ray_backend='native',observation=checkpoint['observation'],perception=checkpoint['perception'],
                training_camera_input=group,camera_input=group,inference_camera_input_override=None,
                selected_motion_indices=[clip],tracked_body_names=list(bodies[::-1]),
                seed=2,sensor_noise='on',start_fraction=0.,maximum_controls=None,
                reset_noise='zero',proprioception_noise='zero',goal_noise='zero',
                camera_geometry=dict(retained_geom_ids=[0,1],physical_geoms=2),
                reset=dict(profile='source_pose',motion_start_height_offset_m=.1,ensure_link_below_zero_ground=True),
                initial_pose_offset=dict(world_xy_m=[0.,0.],world_yaw_rad=0.),
                push_force=dict(enabled=False,force_world_N=[0.,0.,0.],torque_world_Nm=[0.,0.,0.],
                    requested_impulse_world_Ns=[0.,0.,0.]),
                physics=dict(timestep=.005,iterations=10,path=str(root/'library.so'),artifact=str(root/'contract.json')),
                reports=[reading])
            (root/'result.json').write_text(json.dumps(evaluation))
            np.savez(trajectory/'trajectory.npz',frame=np.arange(start,start+count),final_frame=np.array([start+count]),
                body_world_position=positions,world_body_error_m=errors,torso_error_m=errors[:,0],
                state_tracking_terminated=terminated,qpos=np.zeros((count,36)),qvel=np.zeros((count,35)),
                final_qpos=np.zeros((1,36)),final_qvel=np.zeros((1,35)),policy_action=np.zeros((count,29)),
                action=np.zeros((count,29)),depth=np.full((count,576),1. if group=='constant-far' else .5,np.float32))
    return roots,paths,checkpoints,reference


def test_initial_tensor_representation_and_values_are_checked():
    checkpoint = generated_checkpoint('measured',0)
    assert compare_arrays(checkpoint['actor'],checkpoint['actor_ema'])['equal']
    changed = deepcopy(checkpoint['actor'])
    changed['enc']['weight'] = changed['enc']['weight'].astype(np.float64)
    assert not compare_arrays(checkpoint['actor'],changed)['equal']
    changed = deepcopy(checkpoint['actor'])
    changed['enc']['weight'][0,0] = 1
    assert not compare_arrays(checkpoint['actor'],changed)['equal']
    changed['enc']['weight'] = changed['enc']['weight'].reshape(3,2)
    assert not compare_arrays(checkpoint['actor'],changed)['equal']
    assert not compare_arrays(checkpoint['actor'],{'missing':np.zeros(1)})['equal']


def test_checkpoint_is_bound_to_requested_updates_and_configuration(tmp_path):
    path = tmp_path/'initial.pkl'
    checkpoint = generated_checkpoint('measured',0)
    with path.open('wb') as handle:
        pickle.dump(checkpoint,handle)
    config = {key:checkpoint[key] for key in ('observation','perception')}
    assert load_checkpoint(path,config,0)['completed_updates']==0
    with pytest.raises(ValueError,match='update counts'):
        load_checkpoint(path,config,2250)
    config['observation'] = dict(camera_input='constant-far')
    with pytest.raises(ValueError,match='resolved training'):
        load_checkpoint(path,config,0)
    checkpoint['cnorm'][0][0] = np.nan
    with path.open('wb') as handle:
        pickle.dump(checkpoint,handle)
    config['observation'] = checkpoint['observation']
    with pytest.raises(ValueError,match='nonfinite'):
        load_checkpoint(path,config,0)


def test_allowed_camera_and_locator_differences_are_explicit():
    configs = {group:dict(seed=0,num_envs=128,horizon=64,collect_envs=128,replay_steps=256,regression_batch=2048,
        source_camera_input=group,out=group,engine=dict(path=group,artifact=group,timestep=.005),
        observation=dict(camera_input=group,camera_input_definition=group)) for group in GROUPS}
    result = compare_configurations(configs)
    assert result['match_except_declared_fields']
    configs['constant-far']['seed'] = 1
    result = compare_configurations(configs)
    assert not result['match_except_declared_fields']
    assert [row['field'] for row in result['differences'] if not row['allowed']]==['.seed']


def test_common_support_includes_earlier_terminal_and_keeps_own_readings(tmp_path):
    roots,paths,checkpoints,reference = generated_readings(tmp_path)
    result = compare_evaluations(roots,paths,checkpoints,'engine',reference,'generated-reference',1e-6)
    pair = result['paired_common_supports'][2]
    support = pair['common_support']
    assert support['controls']==3 and support['states']==4 and support['duration_s']==.06
    assert support['start_frame']==12 and support['terminal_inclusive']
    assert support['means']['measured']['mean_world_body_error_m']==pytest.approx(.09,abs=1e-6)
    assert support['means']['constant-far']['mean_world_body_error_m']==pytest.approx(.08,abs=1e-6)
    assert result['conditions']['constant-far']['reports'][2]['controls']==4
    assert result['conditions']['measured']['reference_completions']==9
    assert result['conditions']['measured']['reports'][2]['tracking_terminated']
    assert pair['initial_qpos_equal'] and pair['initial_qvel_equal']


@pytest.mark.parametrize('field,value,error',[
    ('checkpoint','different.pkl','selected EMA'),('seed',0,'seed 2'),
    ('inference_camera_input_override','measured','trained camera'),
])
def test_unmatched_checkpoint_and_inference_conditions_are_rejected(tmp_path,field,value,error):
    roots,paths,checkpoints,reference = generated_readings(tmp_path)
    path = roots['measured']/'motion-00'/'result.json'
    evaluation = json.loads(path.read_text())
    evaluation[field] = value
    path.write_text(json.dumps(evaluation))
    with pytest.raises(ValueError,match=error):
        compare_evaluations(roots,paths,checkpoints,'engine',reference,'generated-reference',1e-6)


@pytest.mark.parametrize('field,subfield,value,error',[
    ('sensor_noise',None,'off','camera noise on'),
    ('initial_pose_offset','world_xy_m',[.05,0.],'zero initial world'),
    ('initial_pose_offset','world_yaw_rad',.1,'zero initial world'),
    ('push_force','enabled',True,'disabled zero-force'),
    ('push_force','force_world_N',[50.,0.,0.],'disabled zero-force'),
    ('push_force','torque_world_Nm',[0.,0.,1.],'disabled zero-force'),
    ('push_force','requested_impulse_world_Ns',[5.,0.,0.],'disabled zero-force'),
])
def test_matching_non_target_physical_conditions_are_rejected(tmp_path,field,subfield,value,error):
    roots,paths,checkpoints,reference = generated_readings(tmp_path)
    for group in GROUPS:
        path = roots[group]/'motion-00'/'result.json'
        evaluation = json.loads(path.read_text())
        if subfield is None:
            evaluation[field] = value
        else:
            evaluation[field][subfield] = value
        path.write_text(json.dumps(evaluation))
    with pytest.raises(ValueError,match=error):
        compare_evaluations(roots,paths,checkpoints,'engine',reference,'generated-reference',1e-6)


def test_saved_error_tampering_and_false_constant_depth_are_rejected(tmp_path):
    roots,paths,checkpoints,reference = generated_readings(tmp_path)
    path = roots['constant-far']/'motion-00'/'clip_00'/'trajectory.npz'
    with np.load(path) as archive:
        arrays = {key:archive[key] for key in archive.files}
    arrays['world_body_error_m'][0,0] += .01
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='Saved errors'):
        compare_evaluations(roots,paths,checkpoints,'engine',reference,'generated-reference',1e-6)
    arrays['world_body_error_m'][0,0] -= .01
    arrays['depth'][0,0] = .5
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='actual constant'):
        compare_evaluations(roots,paths,checkpoints,'engine',reference,'generated-reference',1e-6)
