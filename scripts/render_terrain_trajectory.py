"""Render the recorded physical qpos chain written by torch_terrain_evaluate.

Requires the pinned environment's MuJoCo rendering backend. Uses an existing
ffmpeg executable when available, otherwise writes PNG frames at control_hz.
Optional --reference overlays the prepared robot at each recorded reference frame,
with ACTUAL and REFERENCE colour captions and one copy of the terrain.
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


def caption(pixels, text, left, color):
    """Draw comparison labels and elapsed seconds without a font dependency."""
    glyphs = {
        'A': ('01110', '10001', '10001', '11111', '10001', '10001', '10001'),
        'C': ('01111', '10000', '10000', '10000', '10000', '10000', '01111'),
        'T': ('11111', '00100', '00100', '00100', '00100', '00100', '00100'),
        'U': ('10001', '10001', '10001', '10001', '10001', '10001', '01110'),
        'L': ('10000', '10000', '10000', '10000', '10000', '10000', '11111'),
        'R': ('11110', '10001', '10001', '11110', '10100', '10010', '10001'),
        'E': ('11111', '10000', '10000', '11110', '10000', '10000', '11111'),
        'F': ('11111', '10000', '10000', '11110', '10000', '10000', '10000'),
        'N': ('10001', '11001', '11001', '10101', '10011', '10011', '10001'),
        'S': ('01111', '10000', '10000', '01110', '00001', '00001', '11110'),
        '0': ('01110', '10001', '10011', '10101', '11001', '10001', '01110'),
        '1': ('00100', '01100', '00100', '00100', '00100', '00100', '01110'),
        '2': ('01110', '10001', '00001', '00010', '00100', '01000', '11111'),
        '3': ('11110', '00001', '00001', '01110', '00001', '00001', '11110'),
        '4': ('00010', '00110', '01010', '10010', '11111', '00010', '00010'),
        '5': ('11111', '10000', '10000', '11110', '00001', '00001', '11110'),
        '6': ('01110', '10000', '10000', '11110', '10001', '10001', '01110'),
        '7': ('11111', '00001', '00010', '00100', '01000', '01000', '01000'),
        '8': ('01110', '10001', '10001', '01110', '10001', '10001', '01110'),
        '9': ('01110', '10001', '10001', '01111', '00001', '00001', '01110'),
        '.': ('00000', '00000', '00000', '00000', '00000', '00100', '00100'),
        ' ': ('00000',)*7,
    }
    scale = 2
    for letter in text:
        for row, values in enumerate(glyphs[letter]):
            for column, value in enumerate(values):
                if value == '1':
                    pixels[8 + row*scale:8 + (row+1)*scale,
                           left + column*scale:left + (column+1)*scale] = color
        left += 6*scale


def render_poses(model, poses, reference_poses, args, control_hz, *, follow_body=None, geom_groups=None, show_elapsed=False):
    """Render one saved state chain and its aligned reference using a shared camera."""
    import mujoco
    import numpy as np

    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    # The physical fixture contains collision geoms. Give the body and terrain
    # distinct display colours while using their recorded compiled shapes.
    robot_geoms = model.geom_bodyid != 0
    model.geom_rgba[robot_geoms] = [.18, .45, .80, 1.0]
    model.geom_matid[robot_geoms] = -1
    model.geom_rgba[~robot_geoms] = [.62, .66, .51, 1.0]
    data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model) if reference_poses is not None else None
    reference_option = mujoco.MjvOption()
    if geom_groups is not None:
        reference_option.geomgroup[:] = 0
        reference_option.geomgroup[list(geom_groups)] = 1
        reference_option.sitegroup[:] = 0
    root_id = model.body(follow_body).id if follow_body is not None else None
    reference_perturb = mujoco.MjvPerturb()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    camera.distance = args.distance
    robot_bodies = np.flatnonzero(model.body_rootid != 0)
    if args.camera == 'fixed':
        centers = []
        framing_poses = poses if reference_poses is None else np.concatenate((poses, reference_poses))
        for pose in framing_poses:
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            centers.append((data.xpos[robot_bodies].min(0) + data.xpos[robot_bodies].max(0)) / 2)
        centers = np.asarray(centers)
        camera.lookat[:] = (centers.min(0) + centers.max(0)) / 2
        camera.distance = max(args.distance, float(np.linalg.norm(np.ptp(centers, axis=0))) + 3)
    if args.ffmpeg is not None and not (args.ffmpeg.is_file() and os.access(args.ffmpeg, os.X_OK)):
        raise ValueError('The explicitly selected ffmpeg must be an executable file')
    ffmpeg = str(args.ffmpeg) if args.ffmpeg is not None else shutil.which('ffmpeg')
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
                    body_positions = data.xpos[robot_bodies]
                    if reference_data is not None:
                        reference_data.qpos[:] = reference_poses[index]
                        mujoco.mj_kinematics(model, reference_data)
                        body_positions = np.concatenate((body_positions, reference_data.xpos[robot_bodies]))
                    if args.camera == 'follow':
                        if root_id is None:
                            camera.lookat[:] = (body_positions.min(0) + body_positions.max(0)) / 2
                        else:
                            camera.lookat[:] = data.xpos[root_id]
                        if reference_data is not None:
                            if root_id is None:
                                camera.distance = max(args.distance, float(np.linalg.norm(np.ptp(body_positions, axis=0)))+1)
                            else:
                                extent = float(np.max(np.linalg.norm(body_positions-camera.lookat, axis=1)))
                                camera.distance = max(args.distance, extent*2+1)
                    renderer.update_scene(data, camera=camera, scene_option=reference_option)
                    if reference_data is not None:
                        actual_geoms = renderer.scene.ngeom
                        mujoco.mjv_addGeoms(model, reference_data, reference_option, reference_perturb,
                                           mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
                        for geom in renderer.scene.geoms[actual_geoms:renderer.scene.ngeom]:
                            geom.rgba[:] = [.95, .88, .68, .45]
                    pixels = renderer.render()
                    if reference_data is not None or show_elapsed:
                        pixels[:30] = [20, 24, 30]
                        caption(pixels, 'ACTUAL', 10, [70, 150, 255])
                        if reference_data is not None:
                            caption(pixels, 'REFERENCE', 120, [245, 225, 175])
                        if show_elapsed:
                            caption(pixels, f'T {index/control_hz:.2f}S', 260, [235, 235, 235])
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
    result = dict(format='mp4' if ffmpeg else 'png_frames', ffmpeg=ffmpeg,
                  video=str(args.output/'policy.mp4') if ffmpeg else None,
                  frames_directory=str(frames) if not ffmpeg else None,
                  mujoco_version=mujoco.__version__, mujoco_gl=os.environ['MUJOCO_GL'])
    if not ffmpeg:
        result['encoding_dependency'] = 'ffmpeg with libx264 is required to encode the saved PNG frames to MP4'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Prepared single-motion NPZ to overlay at recorded reference frames')
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, help='Explicit executable from the pinned encoder artifact')
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
    if args.reference is not None:
        args.reference = args.reference.resolve()
    if args.ffmpeg is not None:
        args.ffmpeg = args.ffmpeg.resolve()
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
    reference_poses = None
    if args.reference is not None:
        screening = json.loads(args.reference.with_name('screening.json').read_text())
        index = json.loads(args.reference.with_name('index.json').read_text())
        joint_names = [model.joint(i).name for i in range(1, model.njnt)]
        if screening['binding']['target_joint_order'] != joint_names or screening['binding']['target_root'] != 'pelvis':
            raise ValueError('Reference joint order or root differs from the rendered physical model')
        if index['fps'] != control_hz:
            raise ValueError('Reference and evaluation sample rates differ')
        with np.load(args.reference, allow_pickle=False) as reference:
            reference_qpos = reference['qpos']
            if reference['starts'].tolist() != [0] or reference['lengths'].tolist() != [len(reference_qpos)]:
                raise ValueError('Reference must use the prepared single-motion NPZ schema')
            if reference['names'].tolist() != [args.reference.stem]:
                raise ValueError('Reference name differs from the selected prepared clip')
        if reference_qpos.ndim != 2 or reference_qpos.shape[1] != model.nq or not np.isfinite(reference_qpos).all():
            raise ValueError('Reference qpos must be finite and match the physical model width')
        if not np.issubdtype(reference_frames.dtype, np.integer) or reference_frames.min() < 0 or reference_frames.max() >= len(reference_qpos):
            raise ValueError('Recorded reference frames must index the selected complete reference')
        reference_poses = reference_qpos[reference_frames]
        if args.width < 300 or args.height < 30:
            raise ValueError('Reference comparison captions require at least 300 by 30 pixels')
    status.update(control_hz=control_hz, states=len(poses), reference_frames=reference_frames.tolist(),
                  stop_reason=result['stop_reason'], posture_fall=result['posture_fall'],
                  geometry='Compiled physical geometry; robot blue and terrain olive',
                  terminal_state_included=True, physics_steps=0,
                  reference_artifact_id=screening['artifact_id'] if reference_poses is not None else None)
    write_status()
    rendered = render_poses(model, poses, reference_poses, args, control_hz)
    status.update(state='completed', **rendered)
    write_status()
    print(json.dumps(dict(output=str(args.output), format=status['format'], states=len(poses),
                          control_hz=control_hz)))


if __name__ == '__main__':
    main()
