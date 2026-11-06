"""Read matched camera-training checkpoints and per-motion physical evaluations.

Each evaluation root contains motion-00/result.json through motion-09/result.json
and the current evaluator's clip_XX/trajectory.npz. The reader uses the declared
checkpoint pairs in order, retaining every terminal state and each own support.
It performs no policy inference or simulation.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from inspect_terrain_checkpoints import leaves
from render_source_policy_comparison import comparison_support
from render_source_trajectory import reference_frame_indices
from nablamimic.source_evaluation_metrics import summarize_source_chain


GROUPS = ('measured', 'constant-far')
INITIAL_COMPONENTS = ('actor', 'critic', 'cnorm', 'actor_ema')
ALLOWED_CONFIG_DIFFERENCES = {
    '.source_camera_input', '.observation.camera_input', '.observation.camera_input_definition',
    '.out', '.engine.path', '.engine.artifact',
}


def compare_arrays(first, second):
    """Compare each named tensor's representation and values without hashing."""
    a, b = dict(leaves(first)), dict(leaves(second))
    tensors = {}
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b:
            tensors[name] = dict(equal=False, reason='Missing tensor')
            continue
        x, y = a[name], b[name]
        tensors[name] = dict(shape_first=list(x.shape), shape_second=list(y.shape),
            dtype_first=str(x.dtype), dtype_second=str(y.dtype),
            equal=bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)))
    return dict(equal=bool(tensors) and all(row['equal'] for row in tensors.values()), tensors=tensors)


def compare_configurations(configs):
    first, second = [dict(leaves(configs[group])) for group in GROUPS]
    differences = []
    for name in sorted(first.keys() | second.keys()):
        if name not in first or name not in second:
            differences.append(dict(field=name, allowed=False, reason='Missing configuration field'))
        elif not np.array_equal(first[name], second[name]):
            differences.append(dict(field=name, allowed=name in ALLOWED_CONFIG_DIFFERENCES,
                measured=first[name].tolist(), constant_far=second[name].tolist()))
    return dict(match_except_declared_fields=all(row['allowed'] for row in differences),
        allowed_difference_fields=sorted(ALLOWED_CONFIG_DIFFERENCES), differences=differences)


def load_checkpoint(path, configuration, updates):
    with path.open('rb') as handle:
        checkpoint = pickle.load(handle)
    if (checkpoint['completed_updates'] != updates or checkpoint['iter'] != updates
            or checkpoint['ema_updates'] != updates or checkpoint['training_framework'] != 'torch'):
        raise ValueError('Checkpoint must declare the requested completed Torch and EMA update counts')
    if checkpoint['observation'] != configuration['observation'] or checkpoint['perception'] != configuration['perception']:
        raise ValueError('Checkpoint observation or perception differs from its resolved training configuration')
    for component in INITIAL_COMPONENTS:
        if not all(np.isfinite(array).all() for _, array in leaves(checkpoint[component])):
            raise ValueError('Checkpoint contains nonfinite actor, critic or normalization tensors')
    return checkpoint


def read_chain(root, clip, evaluation, reference, reference_artifact_id, tolerance):
    """Recompute the existing source metrics with name-bound reference bodies."""
    rows = evaluation['reports']
    if len(rows) != 1 or rows[0]['clip'] != clip or evaluation['selected_motion_indices'] != [clip]:
        raise ValueError('Each evaluation process must contain exactly its declared motion')
    report = rows[0]
    names = evaluation['tracked_body_names']
    if len(names) != 14 or len(set(names)) != 14 or set(names) != set(reference.bodies):
        raise ValueError('Evaluation and reference must bind the same fourteen named bodies')
    order = [reference.bodies.index(name) for name in names]
    torso = names.index('torso_link')
    with np.load(root / f'clip_{clip:02d}' / 'trajectory.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('frame', 'final_frame', 'body_world_position',
            'world_body_error_m', 'torso_error_m', 'state_tracking_terminated', 'qpos', 'qvel',
            'final_qpos', 'final_qvel', 'policy_action', 'action', 'depth')}
    count = report['controls']
    expected_shapes = dict(frame=(count,), final_frame=(1,), body_world_position=(count+1,14,3),
        world_body_error_m=(count+1,14), torso_error_m=(count+1,), state_tracking_terminated=(count+1,),
        qpos=(count,36), qvel=(count,35), final_qpos=(1,36), final_qvel=(1,35),
        policy_action=(count,29), action=(count,29), depth=(count,576))
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Current trajectory field {key} has invalid shape or nonfinite values')
    clip_base = int(reference.starts[clip])
    clip_length = int(reference.last[clip_base])-clip_base+1
    local_frames = reference_frame_indices(report, clip_base, clip_length, arrays['frame'], arrays['final_frame'])
    positions = reference.body_pos[clip_base+local_frames][:,order]
    errors = np.linalg.norm(arrays['body_world_position'].astype(np.float64)-positions, axis=-1)
    if (not np.allclose(errors, arrays['world_body_error_m'], rtol=0, atol=tolerance)
            or not np.allclose(errors[:,torso], arrays['torso_error_m'], rtol=0, atol=tolerance)):
        raise ValueError('Saved errors differ from named-body world-position/reference errors')
    reference_complete = (local_frames[0] == 0 and local_frames[-1] == clip_length-1
                          and not arrays['state_tracking_terminated'][-1])
    reading = summarize_source_chain(errors, torso, arrays['state_tracking_terminated'],
        reference_complete, count, report['requested_controls'], 1/reference.fps)
    for key, value in reading.items():
        if isinstance(value, float):
            if not np.isclose(value, report[key], rtol=0, atol=tolerance):
                raise ValueError(f'Saved report {key} differs from the trajectory reading')
        elif value != report[key]:
            raise ValueError(f'Saved report {key} differs from the trajectory reading')
    clipped = int(np.count_nonzero(arrays['policy_action'] != arrays['action']))
    if clipped != report['action_clipped_components']:
        raise ValueError('Report clipping count differs from raw/executed actions')
    depth = arrays['depth']
    if np.any((depth < 0) | (depth > 1)):
        raise ValueError('Source camera input depth must be normalized to [0,1]')
    if evaluation['camera_input'] == 'constant-far' and not np.all(depth == 1):
        raise ValueError('Constant-far execution must store the actual constant actor depth input')
    metadata = dict(report, scene_artifact_id=evaluation['perception']['physical_scene_artifact_id'],
        terrain_artifact_id=evaluation['perception']['terrain_artifact_id'],
        reference_artifact_id=reference_artifact_id, control_hz=reference.fps,
        final_frame=int(arrays['final_frame'][0]),
        **{key:evaluation[key] for key in ('physics_backend','ray_backend','sensor_noise','seed',
            'camera_input','camera_geometry','reset','initial_pose_offset','physics')})
    reading.update(clip=clip, stop_reason=report['stop_reason'], action_clipped_components=clipped,
        start_frame=report['start_frame'], final_frame=metadata['final_frame'],
        saved_error_max_difference_m=float(abs(errors-arrays['world_body_error_m']).max()))
    return dict(reading=reading, metadata=metadata, errors=errors, torso_index=torso,
        terminations=arrays['state_tracking_terminated'], initial_qpos=arrays['qpos'][0], initial_qvel=arrays['qvel'][0])


def compare_evaluations(roots, checkpoint_paths, checkpoints, backend, reference, reference_artifact_id, tolerance):
    """Retain per-motion own supports and the shared terminal-inclusive prefix."""
    conditions = {group:[] for group in GROUPS}
    pairs = []
    for clip in range(10):
        chains = {}
        for group in GROUPS:
            root = roots[group] / f'motion-{clip:02d}'
            evaluation = json.loads((root / 'result.json').read_text())
            checkpoint = checkpoints[group]
            if (Path(evaluation['checkpoint']).resolve() != checkpoint_paths[group].resolve()
                    or evaluation['completed_updates'] != checkpoint['completed_updates']
                    or evaluation['weights'] != 'ema' or evaluation['physics_backend'] != backend
                    or evaluation['ray_backend'] != ('native' if backend == 'engine' else 'mujoco')
                    or evaluation['observation'] != checkpoint['observation']
                    or evaluation['perception'] != checkpoint['perception']):
                raise ValueError('Evaluation differs from its selected EMA checkpoint or backend bindings')
            if (evaluation['training_camera_input'] != group or evaluation['camera_input'] != group
                    or evaluation['inference_camera_input_override'] is not None):
                raise ValueError('Training comparison requires each actor to retain its trained camera condition')
            if evaluation['sensor_noise'] != 'on':
                raise ValueError('Training comparison requires source camera noise on')
            offset = evaluation['initial_pose_offset']
            if not np.array_equal(offset['world_xy_m'], [0.,0.]) or offset['world_yaw_rad'] != 0.:
                raise ValueError('Training comparison requires zero initial world XY and yaw offsets')
            push = evaluation['push_force']
            if push['enabled'] is not False or any(not np.array_equal(push[key], [0.,0.,0.])
                    for key in ('force_world_N','torque_world_Nm','requested_impulse_world_Ns')):
                raise ValueError('Training comparison requires an explicitly disabled zero-force pulse')
            if (evaluation['seed'] != 2 or evaluation['start_fraction'] != 0
                    or evaluation['maximum_controls'] is not None
                    or any(evaluation[key] != 'zero' for key in ('reset_noise','proprioception_noise','goal_noise'))):
                raise ValueError('Evaluation requires independent per-motion seed 2, full references and zero additive noise')
            reset = evaluation['reset']
            if (reset['profile'] != 'source_pose' or reset['motion_start_height_offset_m'] != .1
                    or not reset['ensure_link_below_zero_ground']):
                raise ValueError('Evaluation requires the declared source-reset physical condition')
            chains[group] = read_chain(root, clip, evaluation, reference, reference_artifact_id, tolerance)
            conditions[group].append(chains[group]['reading'])
        comparison_support([chains[group]['metadata'] for group in GROUPS], 'measured-vs-constant-far')
        count = min(chains[group]['reading']['controls'] for group in GROUPS)
        common = {}
        for group, chain in chains.items():
            summary = summarize_source_chain(chain['errors'][:count+1], chain['torso_index'],
                chain['terminations'][:count+1], False, count, count, 1/reference.fps)
            common[group] = {key:summary[key] for key in ('mean_world_body_error_m','mean_torso_error_m')}
        pairs.append(dict(clip=clip, initial_qpos_equal=bool(np.array_equal(chains['measured']['initial_qpos'],chains['constant-far']['initial_qpos'])),
            initial_qvel_equal=bool(np.array_equal(chains['measured']['initial_qvel'],chains['constant-far']['initial_qvel'])),
            common_support=dict(controls=count,states=count+1,duration_s=count/reference.fps,
                start_frame=chains['measured']['reading']['start_frame'],
                terminal_inclusive=True, means=common)))
    return dict(physics_backend=backend, camera_rng='Independent seed 2 in each motion process',
        conditions={group:dict(reference_completions=sum(row['reference_complete'] for row in rows),
            strict_passes=sum(row['strict_pass'] for row in rows), reports=rows) for group,rows in conditions.items()},
        paired_common_supports=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-checkpoints',type=Path,nargs=2,required=True,metavar=('MEASURED','CONSTANT_FAR'))
    parser.add_argument('--training-configs',type=Path,nargs=2,required=True,metavar=('MEASURED','CONSTANT_FAR'))
    parser.add_argument('--checkpoint-pair',nargs=3,action='append',required=True,metavar=('UPDATES','MEASURED','CONSTANT_FAR'))
    parser.add_argument('--evaluation-pair',nargs=4,action='append',required=True,metavar=('UPDATES','BACKEND','MEASURED_ROOT','CONSTANT_FAR_ROOT'))
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--error-absolute-tolerance-m',type=float,default=1e-6)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not np.isfinite(args.error_absolute_tolerance_m) or args.error_absolute_tolerance_m <= 0:
        parser.error('Saved-error consistency tolerance must be positive and finite')
    configs = {group:json.loads(path.read_text()) for group,path in zip(GROUPS,args.training_configs)}
    for group, config in configs.items():
        if (config['source_camera_input'] != group or config['observation']['camera_input'] != group
                or Path(config['motion_lib']).resolve() != args.reference.resolve()):
            raise ValueError('Training configuration differs from its declared camera condition or reference input')
    initial = {group:load_checkpoint(path,configs[group],0) for group,path in zip(GROUPS,args.initial_checkpoints)}
    initialization = {key:compare_arrays(initial['measured'][key],initial['constant-far'][key]) for key in INITIAL_COMPONENTS}
    initialization['ema_matches_actor'] = {group:compare_arrays(initial[group]['actor'],initial[group]['actor_ema']) for group in GROUPS}
    # Reuse the producer's reference reader once, sharing the checked body/frame tables across readings.
    from nablamimic.motion_lib import ShardedLibrary
    source = ShardedLibrary(args.reference,filtered=False)
    reference = source.working_set(source.shards,sum(row['length'] for row in source.clips.values()))
    if reference.names != [f'motion_{clip:02d}' for clip in range(10)] or reference.fps != 50.:
        raise ValueError('Comparison requires the current full ten-motion 50 Hz source library')
    screening = json.loads((args.reference/'screening.json').read_text())
    if any(config['physical_model']['artifact_id'] != screening['target_artifact_id'] for config in configs.values()):
        raise ValueError('Training physical scene and reference producer target artifacts differ')
    snapshots = {}
    for update, *paths in args.checkpoint_pair:
        count = int(update)
        if count < 1 or count in snapshots:
            raise ValueError('Declare unique positive checkpoint update counts')
        bound_paths = dict(zip(GROUPS,map(Path,paths)))
        snapshots[count] = dict(paths=bound_paths,checkpoints={
            group:load_checkpoint(path,configs[group],count) for group,path in bound_paths.items()})
    readings = []
    seen = set()
    for update, backend, *roots in args.evaluation_pair:
        count = int(update)
        if backend not in ('engine','mujoco') or count not in snapshots or (count,backend) in seen:
            raise ValueError('Each backend/update reading must bind one declared checkpoint pair exactly once')
        seen.add((count,backend))
        snapshot = snapshots[count]
        reading = compare_evaluations(dict(zip(GROUPS,map(Path,roots))),snapshot['paths'],snapshot['checkpoints'],
            backend,reference,screening['artifact_id'],args.error_absolute_tolerance_m)
        reading.update(completed_updates=count,checkpoint_paths={group:str(path) for group,path in snapshot['paths'].items()})
        readings.append(reading)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(dict(
        initial_checkpoint_paths=list(map(str,args.initial_checkpoints)), initialization=initialization,
        initialization_arrays_equal=all(initialization[key]['equal'] for key in INITIAL_COMPONENTS)
            and all(row['equal'] for row in initialization['ema_matches_actor'].values()),
        configuration_comparison=compare_configurations(configs),resolved_training_configurations=configs,
        reference_artifact_id=screening['artifact_id'],error_absolute_tolerance_m=args.error_absolute_tolerance_m,
        readings=readings,scope='Input initialization equality and checkpoint-specific physical readings; no training-seed replication or pooled error means'),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
