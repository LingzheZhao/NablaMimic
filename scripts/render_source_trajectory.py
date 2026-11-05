"""Render a source-policy evaluation chain with its aligned complete-library reference.

Consumes evaluate_source_perceptive result.json and one recorded clip trajectory.
The camera follows the actual free-root body from the scene contract. Captions
identify actual/reference geometry and elapsed time. Both are in the declared
world frame, with source visible groups 0 and 2. Rendering uses kinematics only.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True

from render_terrain_trajectory import render_poses
from evaluate_source_perceptive import source_window


def reference_frame_indices(reading, clip_base, clip_length, frames, final_frame):
    import numpy as np

    window = source_window(clip_length, reading['start_fraction'], reading['requested_controls'])
    start = clip_base+window['start_offset']
    count = reading['controls']
    if (reading['clip_start_frame'] != clip_base or reading['start_frame'] != start
            or reading['start_offset'] != window['start_offset']
            or reading['remaining_controls'] != window['remaining_controls']
            or reading['requested_controls'] != window['requested_controls']
            or not 1 <= count <= window['requested_controls']):
        raise ValueError('Evaluation start or requested support differs from its reference window')
    recorded = np.concatenate((frames, final_frame))
    if not np.array_equal(recorded, np.arange(start,start+count+1)):
        raise ValueError('Recorded reference frames must describe one continuous window including its terminal state')
    return recorded-clip_base


def load_source_reading(args):
    import mujoco
    import numpy as np

    scene = json.loads(args.scene_contract.read_text())
    evaluation = json.loads(args.evaluation_result.read_text())
    index = json.loads((args.reference/'index.json').read_text())
    reference_contract = json.loads((args.reference/'screening.json').read_text())
    ordered_names = [f'motion_{i:02d}' for i in range(10)]
    if sorted(index['clips']) != ordered_names or args.clip not in range(10):
        raise ValueError('Expected the current full ten-motion source library and a declared clip index')
    readings = [row for row in evaluation['reports'] if row['clip'] == args.clip]
    if len(readings) != 1:
        raise ValueError('Evaluation must bind the selected clip exactly once')
    reading = readings[0]
    root = scene['free_root_body']
    if root != 'torso_link' or evaluation['observation']['free_root_body'] != root:
        raise ValueError('Source evaluation and model must bind the torso free root')
    perception = evaluation['perception']
    if perception['physical_scene_artifact_id'] != scene['artifact_id'] or perception['terrain_artifact_id'] != scene['terrain_artifact_id']:
        raise ValueError('Evaluation and selected physical geometry artifacts differ')
    if perception['physical_frame'] != scene['frame']:
        raise ValueError('Evaluation and selected physical world frames differ')
    if reference_contract['target_artifact_id'] != scene['artifact_id'] or reference_contract['binding']['scene_artifact_id'] != scene['artifact_id']:
        raise ValueError('Reference and physical scene artifacts differ')
    model = mujoco.MjModel.from_binary_path(str(args.model))
    order = [model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu)]
    if model.body(model.jnt_bodyid[0]).name != root or order != scene['actuator_joint_names'] or order != reference_contract['binding']['target_joint_order']:
        raise ValueError('Constructed root or actuator binding differs from the saved contracts')
    if model.ngeom != scene['counts']['geoms'] or model.npair != scene['counts']['pairs']:
        raise ValueError('Constructed model geometry counts differ from the saved contract')
    control_hz = index['fps']
    if not np.isfinite(control_hz) or control_hz <= 0 or not np.isclose(4*model.opt.timestep*control_hz, 1., rtol=0, atol=1e-12):
        raise ValueError('Reference clock differs from four physical substeps per control')
    name = ordered_names[args.clip]
    clip = index['clips'][name]
    clip_base = sum(index['clips'][key]['length'] for key in ordered_names[:args.clip])
    terrain = next(row for row in scene['terrains'] if row['terrain_index'] == clip['terrain_id'])
    if clip['world_translation_m'] != terrain['input_to_world_translation_m']:
        raise ValueError('Selected reference and physical terrain translations differ')
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        qpos = trajectory['qpos']
        final_qpos = trajectory['final_qpos']
        frames = trajectory['frame']
        final_frame = trajectory['final_frame']
    count = reading['controls']
    if qpos.shape != (count, model.nq) or final_qpos.shape != (1, model.nq) or frames.shape != (count,) or final_frame.shape != (1,) or count < 1:
        raise ValueError('Expected the current single-chain source-policy trajectory schema')
    local_frames = reference_frame_indices(reading,clip_base,clip['length'],frames,final_frame)
    with np.load(args.reference/f"{clip['shard']}.npz", allow_pickle=False) as reference:
        if reference['names'].tolist() != [name] or reference['starts'].tolist() != [0] or reference['lengths'].tolist() != [clip['length']]:
            raise ValueError('Selected reference must use the current single-motion shard schema')
        reference_qpos = reference['qpos']
    if reference_qpos.shape != (clip['length'], model.nq) or local_frames[-1] >= len(reference_qpos):
        raise ValueError('Recorded chain extends beyond its declared reference support')
    poses = np.concatenate((qpos, final_qpos))
    reference_poses = reference_qpos[local_frames]
    if not np.isfinite(poses).all() or not np.isfinite(reference_poses).all():
        raise ValueError('Recorded or reference qpos contains nonfinite values')
    metadata = dict(scene_artifact_id=scene['artifact_id'], terrain_artifact_id=scene['terrain_artifact_id'],
                    reference_artifact_id=reference_contract['artifact_id'], checkpoint=evaluation['checkpoint'],
                    completed_updates=evaluation['completed_updates'], weights=evaluation['weights'],
                    sensor_noise=evaluation['sensor_noise'], clip=args.clip, reference_motion=name,
                    free_root_body=root, control_hz=control_hz, controls=count, states=len(poses),
                    elapsed_s=count/control_hz, stop_reason=reading['stop_reason'],
                    clip_start_frame=clip_base, start_frame=reading['start_frame'],
                    start_fraction=reading['start_fraction'], start_offset=reading['start_offset'],
                    requested_controls=reading['requested_controls'],remaining_controls=reading['remaining_controls'],
                    final_frame=int(final_frame[0]), terminal_state_included=True,
                    geometry_groups=[0, 2], comparison='Actual blue and reference translucent cream in the same world frame',
                    camera_follows='Actual '+root, physics_steps=0)
    return model, poses, reference_poses, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'scene-contract', 'evaluation-result', 'trajectory', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--clip', type=int, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--camera', choices=('follow', 'fixed'), default='follow')
    parser.add_argument('--distance', type=float, default=4.)
    parser.add_argument('--azimuth', type=float, default=135.)
    parser.add_argument('--elevation', type=float, default=-20.)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    if args.width < 430 or args.height < 30 or args.width % 2 or args.height % 2 or args.distance <= 0:
        parser.error('Comparison requires even dimensions at least 430 by 30 and a positive camera distance')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    status = dict(state='started', code_revision=args.code_revision,
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    def write_status():
        (args.output/'execution.json').write_text(json.dumps(status, indent=2))

    def fail(kind, value, tb):
        status.update(state='failed', failure_type=kind.__name__, failure=str(value),
                      traceback=''.join(traceback.format_exception(kind, value, tb)))
        write_status()
        sys.__excepthook__(kind, value, tb)

    write_status()
    sys.excepthook = fail
    model, poses, reference_poses, metadata = load_source_reading(args)
    status.update(metadata)
    write_status()
    rendered = render_poses(model, poses, reference_poses, args, metadata['control_hz'],
                            follow_body=metadata['free_root_body'], geom_groups=(0, 2), show_elapsed=True)
    status.update(state='completed', **rendered)
    write_status()
    print(json.dumps(dict(output=str(args.output), format=status['format'], states=len(poses))))


if __name__ == '__main__':
    main()
