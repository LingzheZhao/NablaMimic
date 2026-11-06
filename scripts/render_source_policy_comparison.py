"""Compare recorded source-policy chains on synchronized display time.

Each physical terminal state is retained. After a condition stops, its panel is
blank with STOP and its actual support, while other conditions continue. Those
display frames contain no physical poses. Depth insets show the saved actor
input, with no depth invented for an unobserved terminal state. No policy
inference, physics stepping, resets or trajectory concatenation occurs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from render_source_trajectory import load_source_reading
from render_terrain_trajectory import caption, render_poses, write_png


def comparison_support(metadata):
    """Require one shared requested window, retaining each executed support."""
    if len(metadata) < 2:
        raise ValueError('A comparison requires at least two recorded conditions')
    shared = ('scene_artifact_id', 'terrain_artifact_id', 'reference_artifact_id',
              'physics_backend', 'ray_backend', 'control_hz', 'start_frame',
              'clip_start_frame', 'start_fraction', 'start_offset', 'requested_controls',
              'sensor_noise', 'seed', 'camera_input', 'reset', 'initial_pose_offset')
    first = metadata[0]
    # These two Engine fields locate per-job compiled libraries and their JSON.
    # Keep the complete original metadata in the saved condition records.
    first_physics = {key: value for key, value in first['physics'].items() if key not in ('path', 'artifact')}
    for row in metadata:
        physics = {key: value for key, value in row['physics'].items() if key not in ('path', 'artifact')}
        if any(row[key] != first[key] for key in shared) or physics != first_physics:
            raise ValueError('Conditions must share physical bindings, backends, clock and requested reference window')
        if (row['states'] != row['controls']+1 or not 1 <= row['controls'] <= row['requested_controls']
                or row['final_frame'] != row['start_frame']+row['controls']):
            raise ValueError('Each physical chain must retain its contiguous terminal state')
    return first['requested_controls']+1


def depth_for_state(depth, index, controls):
    """Depth is recorded only for action-input states, excluding terminal."""
    if depth is None or index == controls:
        return None
    return depth[index].reshape(18, 32)


def metrics_text(report):
    return (f"MEAN BODY {report['mean_world_body_error_m']:.3f}M "
            f"MEAN TORSO {report['mean_torso_error_m']:.3f}M "
            f"COMPLETE {int(report['reference_complete'])} STRICT {int(report['strict_pass'])} N {report['controls']}")


def frame_labels(label, report, depth):
    def transform(pixels, index):
        pixels[30:90] = [20, 24, 30]
        caption(pixels[30:], label, 10, [235, 235, 235])
        caption(pixels[60:], metrics_text(report), 10, [235, 235, 235])
        image = depth_for_state(depth, index, report['controls'])
        y, x = pixels.shape[0]-172, pixels.shape[1]-266
        pixels[y:y+164, x:x+256] = [20, 24, 30]
        if image is not None and np.isfinite(image).all():
            caption(pixels[y:], 'DEPTH INPUT', x+4, [235, 235, 235])
            gray = np.rint(image.clip(0, 1)*255).astype(np.uint8)
            pixels[y+20:y+164, x:x+256] = np.repeat(np.repeat(gray, 8, 0), 8, 1)[..., None]
        else:
            caption(pixels[y:], 'NO DEPTH', x+4, [235, 235, 235])
        if index == report['controls']:
            pixels[90:120] = [60, 20, 20]
            caption(pixels[90:], 'STOP', 10, [255, 180, 150])
        return pixels
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'scene-contract', 'reference', 'output', 'ffmpeg'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', nargs=2, action='append', required=True, metavar=('LABEL', 'EVALUATION_DIR'))
    parser.add_argument('--clip', type=int, action='append')
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--distance', type=float, default=3.)
    parser.add_argument('--azimuth', type=float, default=90.)
    parser.add_argument('--elevation', type=float, default=-55.)
    args = parser.parse_args()
    clips = [2, 3, 5, 9] if args.clip is None else args.clip
    labels = [label.upper() for label, _ in args.condition]
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-')
    if (len(labels) < 2 or len(set(labels)) != len(labels)
            or any(not label or len(label) > 75 or set(label)-allowed for label in labels)
            or len(set(clips)) != len(clips) or any(clip not in range(10) for clip in clips)
            or not np.isfinite([args.distance, args.azimuth, args.elevation]).all() or args.distance <= 0):
        parser.error('Use unique clips 0..9, at least two unique ASCII labels up to 75 characters, and finite camera parameters')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    conditions = [(label, Path(path).resolve()) for label, (_, path) in zip(labels, args.condition)]
    if not args.ffmpeg.is_file() or not os.access(args.ffmpeg, os.X_OK):
        parser.error('The selected ffmpeg must be an executable file')
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    status = dict(state='started', code_revision=args.code_revision, clips=clips,
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        camera=dict(mode='follow', distance=args.distance, azimuth=args.azimuth, elevation=args.elevation,
                    width=960, height=720, follows='Each recorded actual torso independently'),
        display_after_stop='Blank STOP panel only; no continuation or repeated terminal physical pose',
        depth='Recorded processed actor input [0,1], far white; terminal state has no recorded depth',
        metrics_support='Each condition own terminal-inclusive physical support; N is its executed controls',
        physics_steps=0, motions=[], commands=[])

    def save():
        (args.output/'execution.json').write_text(json.dumps(status, indent=2))

    def run(command, log):
        status['commands'].append([str(value) for value in command])
        save()
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    save()
    try:
        for clip in clips:
            directory = args.output/f'motion-{clip:02d}'
            directory.mkdir()
            loaded, reports, metadata = [], [], []
            for label, root in conditions:
                evaluation = json.loads((root/'result.json').read_text())
                report = next(row for row in evaluation['reports'] if row['clip'] == clip)
                trajectory = root/f'clip_{clip:02d}/trajectory.npz'
                reading_args = argparse.Namespace(model=args.model, scene_contract=args.scene_contract,
                    evaluation_result=root/'result.json', trajectory=trajectory, reference=args.reference,
                    clip=clip, reference_overlay='off')
                model, poses, _, row = load_source_reading(reading_args)
                with np.load(trajectory, allow_pickle=False) as arrays:
                    depth = arrays['depth']
                if depth.shape != (report['controls'], 576):
                    raise ValueError('Recorded actor depth must have controls by 576 shape')
                loaded.append((label, model, poses, depth))
                reports.append(report)
                metadata.append(dict(row, label=label, evaluation_result=str(root/'result.json'),
                    trajectory=str(trajectory), depth_recorded=depth is not None, report=report,
                    **{key: evaluation[key] for key in ('seed', 'camera_input', 'reset', 'physics', 'initial_pose_offset')}))
            total = comparison_support(metadata)
            videos, stops = [], []
            for index, ((label, model, poses, depth), report, row) in enumerate(zip(loaded, reports, metadata)):
                output = directory/f'condition-{index:02d}'
                output.mkdir()
                rendering = argparse.Namespace(output=output, width=960, height=720, camera='follow', ffmpeg=args.ffmpeg,
                    distance=args.distance, azimuth=args.azimuth, elevation=args.elevation)
                row['rendered'] = render_poses(model, poses, None, rendering, row['control_hz'],
                    follow_body=row['free_root_body'], geom_groups=(0, 2), show_elapsed=True,
                    frame_transform=frame_labels(label, report, depth))
                stop = np.full((720, 960, 3), [20, 24, 30], dtype=np.uint8)
                caption(stop, 'STOP', 10, [255, 180, 150])
                caption(stop[40:], label, 10, [235, 235, 235])
                caption(stop[80:], f"ACTUAL SUPPORT {report['controls']/row['control_hz']:.2f}S", 10, [235, 235, 235])
                caption(stop[120:], metrics_text(report), 10, [235, 235, 235])
                stop_path = output/'stop.png'
                write_png(stop_path, stop)
                videos.append(output/'policy.mp4')
                stops.append(stop_path)
            command = [str(args.ffmpeg), '-nostdin', '-hide_banner', '-loglevel', 'warning', '-n']
            for path in (*videos, *stops):
                command.extend(('-i', str(path)))
            filters = []
            count = len(conditions)
            for index, row in enumerate(metadata):
                filters.append(f'[{index}:v]tpad=stop_mode=add:stop={total-row["states"]}:color=black[p{index}]')
                filters.append(f'[p{index}][{count+index}:v]overlay=enable=\'gt(t,{row["states"]-.5}/{row["control_hz"]})\':'
                               f'repeatlast=1:eof_action=repeat[v{index}]')
            filters.append(''.join(f'[v{index}]' for index in range(count))+f'hstack=inputs={count},format=yuv420p[out]')
            video = directory/'comparison.mp4'
            command.extend(('-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-c:v', 'libx264',
                            '-crf', '18', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-movflags', '+faststart', str(video)))
            run(command, directory/'comparison-encode.log')
            decode = subprocess.run([str(args.ffmpeg), '-nostdin', '-hide_banner', '-i', str(video),
                '-map', '0:v:0', '-an', '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'], capture_output=True, text=True)
            (directory/'decode.log').write_text(decode.stderr+'\n'+decode.stdout)
            decode.check_returncode()
            progress = dict(line.split('=', 1) for line in decode.stdout.splitlines() if '=' in line)
            if int(progress['frame']) != total:
                raise ValueError('Decoded comparison must retain exactly the requested display states including terminal')
            selected = np.rint(np.linspace(0, total-1, 6)).astype(int).tolist()
            selection = '+'.join(f'eq(n,{frame})' for frame in selected)
            run([str(args.ffmpeg), '-nostdin', '-hide_banner', '-loglevel', 'warning', '-n', '-i', str(video),
                 '-vf', f"select='{selection}',scale=960:-1,tile=2x3", '-frames:v', '1',
                 str(directory/'contact-sheet.png')], directory/'contact-sheet.log')
            status['motions'].append(dict(clip=clip, display_states=total, display_hz=metadata[0]['control_hz'],
                conditions=metadata, contact_sheet_frames=selected, video=str(video)))
            save()
        status['state'] = 'completed'
        save()
    except BaseException as error:
        status.update(state='failed', failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
