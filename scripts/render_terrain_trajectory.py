"""Render the recorded physical qpos chain written by torch_terrain_evaluate.

Requires the pinned environment's MuJoCo rendering backend. Uses an existing
ffmpeg executable when available, otherwise writes PNG frames at control_hz.
No physics stepping or policy inference occurs during rendering.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback
import zlib

sys.dont_write_bytecode = True


def write_png(path, pixels):
    """Encode RGB frames using the standard library for the ffmpeg-free route."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    height, width, _ = pixels.shape
    scanlines = b''.join(b'\x00' + row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera', choices=('follow', 'fixed'), default='follow')
    parser.add_argument('--distance', type=float, default=4.0)
    parser.add_argument('--azimuth', type=float, default=135.0)
    parser.add_argument('--elevation', type=float, default=-20.0)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2 or args.distance <= 0:
        parser.error('Positive even dimensions and a positive camera distance are required')
    args.trajectory = args.trajectory.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(state='started', code_revision=args.code_revision,
                  arguments={key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()})

    def write_status():
        (args.output / 'execution.json').write_text(json.dumps(status, indent=2))

    def record_failure(kind, value, tb):
        status.update(state='failed', failure_type=kind.__name__, failure=str(value),
                      traceback=''.join(traceback.format_exception(kind, value, tb)))
        write_status()
        sys.__excepthook__(kind, value, tb)

    write_status()
    sys.excepthook = record_failure
    os.chdir(args.output)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import numpy as np

    result = json.loads(args.trajectory.with_name('result.json').read_text())
    control_hz = result['control_hz']
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('Evaluation control_hz must be positive and finite')
    model = mujoco.MjModel.from_binary_path(str(args.model))
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        qpos = trajectory['qpos']
        final_qpos = trajectory['final_qpos']
        reference_frames = trajectory['frame']
        final_frame = trajectory['final_frame']
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) == 0:
        raise ValueError('Recorded qpos must be a nonempty controls by model.nq array')
    if final_qpos.shape != (1, model.nq) or reference_frames.shape != (len(qpos),) or final_frame.shape != (1,):
        raise ValueError('Expected the single-chain torch_terrain_evaluate trajectory schema')
    if result['controls'] != len(qpos):
        raise ValueError('Evaluation result and recorded physical chain lengths differ')
    poses = np.concatenate((qpos, final_qpos), axis=0)
    if not np.isfinite(poses).all():
        raise ValueError('Recorded physical qpos contains nonfinite values')
    reference_frames = np.concatenate((reference_frames, final_frame))
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    # The physical fixture contains collision geoms. Give the body and terrain
    # distinct display colours while using their recorded compiled shapes.
    robot_geoms = model.geom_bodyid != 0
    model.geom_rgba[robot_geoms] = [.18, .45, .80, 1.0]
    model.geom_rgba[~robot_geoms] = [.62, .66, .51, 1.0]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    camera.distance = args.distance
    robot_bodies = np.flatnonzero(model.body_rootid != 0)
    if args.camera == 'fixed':
        centers = []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            centers.append((data.xpos[robot_bodies].min(0) + data.xpos[robot_bodies].max(0)) / 2)
        centers = np.asarray(centers)
        camera.lookat[:] = (centers.min(0) + centers.max(0)) / 2
        camera.distance = max(args.distance, float(np.linalg.norm(np.ptp(centers, axis=0))) + 3)
    ffmpeg = shutil.which('ffmpeg')
    status.update(control_hz=control_hz, states=len(poses), reference_frames=reference_frames.tolist(),
                  stop_reason=result['stop_reason'], posture_fall=result['posture_fall'],
                  mujoco_version=mujoco.__version__, mujoco_gl=os.environ['MUJOCO_GL'],
                  geometry='Compiled physical collision geometry; robot blue and terrain olive',
                  ffmpeg=ffmpeg, format='mp4' if ffmpeg else 'png_frames',
                  terminal_state_included=True, physics_steps=0)
    write_status()
    process = None
    frames = args.output / 'frames'
    if not ffmpeg:
        frames.mkdir()
    with (args.output / 'ffmpeg.log').open('wb') as encoder_log:
        if ffmpeg:
            process = subprocess.Popen([
                ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'warning', '-n',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{args.width}x{args.height}',
                '-framerate', str(control_hz), '-i', 'pipe:0', '-an',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(args.output / 'policy.mp4')], stdin=subprocess.PIPE, stderr=encoder_log)
        try:
            with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
                for index, pose in enumerate(poses):
                    data.qpos[:] = pose
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_camlight(model, data)
                    if args.camera == 'follow':
                        camera.lookat[:] = (data.xpos[robot_bodies].min(0) + data.xpos[robot_bodies].max(0)) / 2
                    renderer.update_scene(data, camera=camera)
                    pixels = renderer.render()
                    if process:
                        process.stdin.write(pixels.tobytes())
                    else:
                        write_png(frames / f'{index:06d}.png', pixels)
        finally:
            if process:
                try:
                    process.stdin.close()
                finally:
                    returncode = process.wait()
        if process and returncode:
            raise RuntimeError(f'ffmpeg exited with {returncode}; see ffmpeg.log')
    status.update(state='completed', video=str(args.output / 'policy.mp4') if ffmpeg else None,
                  frames_directory=str(frames) if not ffmpeg else None)
    if not ffmpeg:
        status['encoding_dependency'] = 'ffmpeg with libx264 is required to encode the saved PNG frames to MP4'
    write_status()
    print(json.dumps(dict(output=str(args.output), format=status['format'], states=len(poses),
                          control_hz=control_hz)))


if __name__ == '__main__':
    main()
