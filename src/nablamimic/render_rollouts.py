"""Videos of the records of ``nablamimic.engine_rollouts``, made by replaying the recorded states: no physics, no
policy and no second rollout. ROLLOUTS.md, "Videos of a record", gives the command and what a video shows.

  MUJOCO_GL=egl python -m nablamimic.render_rollouts --record OUT [--record OUT2] --trials 0 3 --out VIDEOS

For each trial: an H.264 MP4, a contact sheet (PNG, 8 frames evenly spaced) and a JSON file naming the sources and
the options. Needs the package's dependencies (mujoco and numpy; jax and mujoco-mjx through nablamimic.env, whose
body list names the processed clips), an OpenGL backend that MuJoCo opens offscreen (MUJOCO_GL=egl or osmesa) and an
ffmpeg executable with the libx264 encoder on PATH.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

import numpy as np

LIBRARY_INPUTS = ("LAFAN1_G1_DIR", "MOTION_CACHE_DIR", "MENAGERIE_DIR")  # as engine_rollouts compares them
RECORD_FIELDS = ("qpos", "frame_after", "clip_last_row", "available", "done", "command_row", "body_row", "start_qpos",
                 "start_body_pos_w")
DESCRIPTION_KEYS = ("controls", "scene_content", "settings")
SETTINGS_KEYS = ("clips", "inputs")
NQ = 36
BODIES = 14  # the tracked bodies of start_body_pos_w (env.BM_BODIES)
START_TOLERANCE = 1e-5  # largest difference of two records' start_qpos shown as one start
BODY_TOLERANCE = 1e-3  # m: largest difference of the model's tracked-body positions at a start from the record's
SHEET_FRAMES, SHEET_COLUMNS = 8, 4
GHOST_RGBA = (0.25, 0.65, 1.0, 0.35)
GHOST_PLACEMENTS = ("world", "robot")
TAIL_S = 2.0  # default seconds shown after a panel's end point
GL_HELP = ("set MUJOCO_GL to an offscreen OpenGL backend that this machine provides: egl (an EGL library, from a GPU "
           "driver or Mesa) or osmesa (libOSMesa)")


class Record:
    """One record folder of engine_rollouts: ``description.json`` and the arrays of ``rollouts.npz`` that a replay
    reads (RECORD_FIELDS)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        description_path, arrays_path = self.folder/"description.json", self.folder/"rollouts.npz"
        for path in (description_path, arrays_path):
            if not path.is_file():
                raise ValueError(f"{self.folder} is not a rollout record: {path.name} is missing")
        self.description = json.loads(description_path.read_text())
        missing = [key for key in DESCRIPTION_KEYS if key not in self.description]
        missing += [f"settings.{key}" for key in SETTINGS_KEYS if key not in self.description.get("settings", {})]
        if missing:
            raise ValueError(f"{description_path} lacks {', '.join(missing)}")
        with np.load(arrays_path) as arrays:
            missing = [key for key in RECORD_FIELDS if key not in arrays.files]
            if missing:
                raise ValueError(f"{arrays_path} lacks the field(s) {', '.join(missing)}")
            self.arrays = {key: arrays[key] for key in RECORD_FIELDS}
        a = self.arrays
        controls, trials = a["qpos"].shape[:2]
        expected = dict(qpos=(controls, trials, NQ), frame_after=(controls, trials), clip_last_row=(controls, trials),
                        available=(controls, trials), done=(controls, trials), command_row=(trials,),
                        body_row=(trials,), start_qpos=(trials, NQ), start_body_pos_w=(trials, BODIES, 3))
        wrong = [f"{key} {a[key].shape} (expected {shape})" for key, shape in expected.items() if a[key].shape != shape]
        if wrong:
            raise ValueError(f"{arrays_path}: shapes {'; '.join(wrong)}")
        if controls != self.description["controls"]:
            raise ValueError(f"{arrays_path} holds {controls} controls, description.json says "
                             f"{self.description['controls']}")
        self.controls, self.trials = controls, trials
        self.inputs = self.description["settings"]["inputs"]
        self.clips = self.description["settings"]["clips"]


def clip_names(clips):
    """The clip names of a run's ``clips`` setting in library order (every processed clip for ['all'])."""
    from . import motion
    names = motion.all_clip_names() if clips == ["all"] else list(clips)
    if clips == ["all"] and len(names) != 40:
        raise ValueError(f"expected the complete 40-clip library in {motion.LAFAN1_DIR}, found {len(names)} CSVs")
    return names


def same_trials(first, second):
    """Refuses two records that do not hold the same trials: the same scene content, trial count, body and command
    rows, clips (the rows index the library of these clips), start states (``start_qpos`` within START_TOLERANCE) and
    controls (the two panels share one time axis)."""
    differences = []
    if first.description["scene_content"] != second.description["scene_content"]:
        differences.append("scene content")
    if first.clips != second.clips and clip_names(first.clips) != clip_names(second.clips):
        differences.append(f"clip names ({first.clips} and {second.clips})")
    if first.trials != second.trials:
        differences.append(f"trial count ({first.trials} and {second.trials})")
    else:
        for key in ("command_row", "body_row"):
            if not np.array_equal(first.arrays[key], second.arrays[key]):
                differences.append(key)
        start, other = first.arrays["start_qpos"], second.arrays["start_qpos"]
        if not np.allclose(start, other, rtol=0, atol=START_TOLERANCE):
            differences.append(f"start_qpos (largest difference {np.nanmax(np.abs(start - other)):.3g}, "
                               f"tolerance {START_TOLERANCE:g})")
    if first.controls != second.controls:
        differences.append(f"controls ({first.controls} and {second.controls})")
    if differences:
        raise ValueError(f"{first.folder} and {second.folder} do not hold the same trials: they differ in "
                         f"{', '.join(differences)}")


def select_trials(requested, count):
    """The trial indices to render, in the order requested (all trials in record order when None)."""
    if requested is None:
        return list(range(count))
    if not requested:
        raise ValueError("no trial index given")
    outside = [t for t in requested if not 0 <= t < count]
    if outside:
        raise ValueError(f"trial indices {outside} are not trials of the record (0 to {count - 1})")
    if len(set(requested)) != len(requested):
        raise ValueError(f"trial indices {requested} repeat an index")
    return list(requested)


def replay_steps(controls, stride):
    """The time steps of the frames: step 0 is the placed start, step s the state after control s - 1; every
    ``stride``-th step from 0 to ``controls``, so ``controls // stride + 1`` frames."""
    if type(stride) is not int or stride < 1:
        raise ValueError(f"the control stride must be a positive integer, got {stride!r}")
    return list(range(0, controls + 1, stride))


def tail_seconds(text):
    """The ``--tail`` value: seconds, or None (every recorded control) for "all" or a negative number."""
    if text == "all":
        return None
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds or 'all', got {text!r}") from None
    if not np.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected finite seconds or 'all', got {text!r}")
    return None if value < 0 else value


def tail_controls(tail_s, rate):
    """The tail in controls (``tail_s`` seconds at ``rate`` controls per second, rounded; None: every control)."""
    return None if tail_s is None else int(round(tail_s*rate))


def trial_end(record, trial):
    """The control that ends ``trial`` (None if none does) and the last control whose state the record holds (-1:
    only the placed start)."""
    done = np.flatnonzero(record.arrays["done"][:, trial])
    if done.size == 0:
        last = record.controls - 1
        end = None
    else:
        end = int(done[0])
        last = end if record.arrays["available"][end, trial] else end - 1
    qpos = record.arrays["qpos"][:last + 1, trial]
    if not np.isfinite(qpos).all() or not np.isfinite(record.arrays["start_qpos"][trial]).all():
        raise ValueError(f"{record.folder}: trial {trial} holds a non-finite state before its last recorded control "
                         f"{last}")
    return end, last


def clip_end(record, trial):
    """The first control after which the command is at its clip's last row (``frame_after == clip_last_row``, over the
    controls whose ``frame_after`` is recorded, i.e. not -1); the command row no longer advances from it. None if the
    command does not reach it within the record."""
    a = record.arrays
    row, last = a["frame_after"][:, trial], a["clip_last_row"][:, trial]
    reached = np.flatnonzero((row >= 0) & (row == last))
    return int(reached[0]) if reached.size else None


def end_point(record, trial):
    """The panel's end point: the earlier of the control that ends the trial and the clip end (the trial end on a
    tie), with its reason ("trial ended" or "clip ended"); (None, "all controls") if neither is in the record."""
    end, clip = trial_end(record, trial)[0], clip_end(record, trial)
    if end is not None and (clip is None or end <= clip):
        return end, "trial ended"
    if clip is not None:
        return clip, "clip ended"
    return None, "all controls"


def video_stop(records, trial, tail):
    """The last control a video of ``trial`` shows, its reason and each panel's end point: ``tail`` controls after the
    later of the panels' end points (the first panel's on a tie), and never beyond the recorded controls. Every
    recorded control (reason "all controls") when ``tail`` is None or a panel has no end point."""
    ends = [end_point(record, trial) for record in records]
    last = records[0].controls - 1
    if tail is None or any(control is None for control, _ in ends):
        return last, "all controls", ends
    control, reason = max(ends, key=lambda point: point[0])
    return min(control + tail, last), reason, ends


def ghost_qpos(reference, robot_qpos, placement):
    """The ghost's ``qpos``: the command row's ``reference`` pose, with its root moved horizontally to the robot's
    root (``robot_qpos[:2]``) for placement "robot"; height, orientation and joints are the reference's."""
    if placement not in GHOST_PLACEMENTS:
        raise ValueError(f"ghost placement {placement!r} is not one of {GHOST_PLACEMENTS}")
    ghost = np.array(reference, np.float64)
    if placement == "robot":
        ghost[:2] = robot_qpos[:2]
    return ghost


def bottom_line(panels, placement):
    """The text lines at the bottom of every panel."""
    parts = ["translucent command pose drawn at the robot's position"] if placement == "robot" else []
    if panels == 2:
        parts.append("two runs from one start")
    return "\n".join(parts)


def panel_states(record, trial, steps):
    """Per step: the robot's ``qpos`` [36], the command row of the ghost, the control the state follows (-1: the
    placed start) and the control that ended the trial (None before the end). A trial that ended is held at its last
    recorded state and its command row."""
    a = record.arrays
    end, last = trial_end(record, trial)
    states = []
    for step in steps:
        control = step - 1
        shown = min(control, last)
        if shown < 0:
            qpos, row = a["start_qpos"][trial], int(a["command_row"][trial])
        else:
            qpos, row = a["qpos"][shown, trial], int(a["frame_after"][shown, trial])
        ended = end if end is not None and control >= end else None
        states.append(dict(qpos=np.asarray(qpos, np.float64), row=row, control=control, shown_control=shown,
                           ended=ended))
    return states


def check_library_inputs(record):
    """Refuses a record whose library paths differ from this environment's (as engine_rollouts does): its rows
    index the motion library these paths locate."""
    here = {key: os.environ.get(key) for key in LIBRARY_INPUTS}
    differing = [key for key in LIBRARY_INPUTS if record.inputs.get(key) != here[key]]
    if differing:
        raise ValueError(f"{record.folder}: the motion library inputs differ from the record's: " + "; ".join(
            f"{key} {here[key]!r} here, {record.inputs.get(key)!r} in the record" for key in differing))


class ReferenceRows:
    """The reference ``qpos`` of the record's clips, read from the processed-clip cache (MOTION_CACHE_DIR) without
    writing to it, with each row's clip and the last row of its clip."""

    def __init__(self, clips):
        from . import motion
        from .env import BM_BODIES
        names = clip_names(clips)
        parts = []
        for name in names:
            path = motion.CACHE_DIR/motion.cache_file_name(name, bodies=BM_BODIES)
            if not path.is_file():
                raise ValueError(f"no processed clip {path}; build the cache with python -m nablamimic.motion")
            with np.load(path) as cached:
                qpos = cached["qpos"]
            if qpos.ndim != 2 or qpos.shape[1] != NQ or not len(qpos):
                raise ValueError(f"{path}: qpos of shape {qpos.shape}, expected [rows, {NQ}]")
            parts.append(qpos.astype(np.float64))
        self.names = names
        self.qpos = np.concatenate(parts)
        lengths = np.array([len(p) for p in parts])
        self.starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        self.last = np.repeat(self.starts + lengths - 1, lengths)

    def clip(self, row):
        return self.names[int(np.searchsorted(self.starts, row, side="right")) - 1]

    def check(self, record):
        """Refuses a record whose rows are not rows of this library or whose clip ends differ from it."""
        a = record.arrays
        rows = np.concatenate((a["command_row"], a["body_row"], a["frame_after"][a["frame_after"] >= 0]))
        if rows.size and (rows.min() < 0 or rows.max() >= len(self.qpos)):
            raise ValueError(f"{record.folder}: rows outside the motion library of {len(self.qpos)} rows")
        valid = a["frame_after"] >= 0
        if not np.array_equal(self.last[a["frame_after"][valid]], a["clip_last_row"][valid]):
            raise ValueError(f"{record.folder}: the record's clip ends differ from the motion library's")


def caption(label, trial, record, library, state, step, rate):
    """The text lines of one panel."""
    command, body = int(record.arrays["command_row"][trial]), int(record.arrays["body_row"][trial])
    lines = [f"{label}  trial {trial}", f"command clip {library.clip(command)}"]
    if body != command:
        lines.append(f"body placed at {library.clip(body)} row {body}")
    lines.append(("start" if state["control"] < 0 else f"control {state['control']}") + f"  {step / rate:.2f} s")
    if state["ended"] is not None:
        lines.append(f"ended at control {state['ended']}")
    return lines


def png_bytes(rgb):
    """An 8-bit RGB PNG of ``rgb`` [H, W, 3] uint8."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def sheet_indices(frames):
    """The frames of a contact sheet: SHEET_FRAMES evenly spaced (all frames when there are fewer)."""
    return sorted(set(np.linspace(0, frames - 1, min(frames, SHEET_FRAMES)).round().astype(int).tolist()))


def contact_sheet(images):
    """The images at half size on a grid of SHEET_COLUMNS columns."""
    small = [img.reshape(img.shape[0]//2, 2, img.shape[1]//2, 2, 3).mean((1, 3)).astype(np.uint8) for img in images]
    height, width = small[0].shape[:2]
    rows = -(-len(small)//SHEET_COLUMNS)
    sheet = np.zeros((rows*height, SHEET_COLUMNS*width, 3), np.uint8)
    for k, img in enumerate(small):
        r, c = divmod(k, SHEET_COLUMNS)
        sheet[r*height:(r + 1)*height, c*width:(c + 1)*width] = img
    return sheet


def find_encoder():
    """The ffmpeg executable on PATH if it has the libx264 encoder, else a refusal."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("no ffmpeg on PATH; the videos are encoded by ffmpeg with libx264")
    encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if " libx264 " not in encoders:
        raise RuntimeError(f"{ffmpeg} has no libx264 encoder")
    return ffmpeg


class VideoWriter:
    """An H.264 MP4 (yuv420p) written by ffmpeg from RGB frames [H, W, 3] uint8."""

    def __init__(self, ffmpeg, path, width, height, fps):
        self.process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.shape = (height, width, 3)

    def write(self, rgb):
        if rgb.shape != self.shape:
            raise ValueError(f"frame of shape {rgb.shape}, expected {self.shape}")
        self.process.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error.strip()}")


def gl_refusal(error):
    return (f"no OpenGL backend could be opened with MUJOCO_GL={os.environ.get('MUJOCO_GL')!r} "
            f"({type(error).__name__}: {error}); {GL_HELP}")


def import_mujoco():
    """The mujoco module; MUJOCO_GL selects its OpenGL backend when it is imported, and an import that fails on it is
    refused with the same message as a context that cannot be opened."""
    try:
        import mujoco
    except Exception as error:
        raise RuntimeError(gl_refusal(error)) from error
    return mujoco


def build_model():
    """The G1 scene as the trainer builds it (robot.build of the Menagerie scene in MENAGERIE_DIR)."""
    import_mujoco()
    from . import motion, robot
    return robot.build(motion.SCENE)


def check_model(model, record, trials):
    """Refuses a model that does not place the tracked bodies at the starts of ``trials`` where the record holds them
    (``start_body_pos_w``, within BODY_TOLERANCE). The record holds no joint names and its ``qpos`` is copied to the
    model's by position, so this compares the model with the one that wrote the record."""
    mujoco = import_mujoco()
    from .env import BM_BODIES
    if len(BM_BODIES) != BODIES:
        raise ValueError(f"{len(BM_BODIES)} tracked bodies, the record holds {BODIES}")
    ids = [model.body(name).id for name in BM_BODIES]
    data = mujoco.MjData(model)
    largest = 0.0
    for trial in trials:
        data.qpos[:] = record.arrays["start_qpos"][trial]
        mujoco.mj_kinematics(model, data)
        largest = max(largest, float(np.abs(data.xpos[ids] - record.arrays["start_body_pos_w"][trial]).max()))
    if not largest <= BODY_TOLERANCE:
        raise ValueError(f"{record.folder}: the model places the tracked bodies at the starts up to {largest:.3g} m "
                         f"from the record's start_body_pos_w (tolerance {BODY_TOLERANCE:g} m); it is not the model "
                         f"that wrote the record")


class PanelRenderer:
    """Offscreen frames of the model: the robot at a ``qpos``, the ghost (a translucent second robot) at another, a
    floor and a camera that follows the robot's root at a fixed azimuth, elevation and distance, with text lines."""

    def __init__(self, model, width, height, distance, azimuth, elevation):
        mujoco = import_mujoco()
        try:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self.gl = mujoco.GLContext(width, height)
            self.gl.make_current()
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        except Exception as error:
            raise RuntimeError(gl_refusal(error)) from error
        self.mujoco, self.model = mujoco, model
        self.data, self.ghost = mujoco.MjData(model), mujoco.MjData(model)
        self.scene = mujoco.MjvScene(model, maxgeom=4*model.ngeom + 100)
        self.option, self.perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, azimuth, elevation
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.pixels = np.empty((height, width, 3), np.uint8)

    def render(self, qpos, ghost_qpos, top_lines, bottom_line=""):
        mj = self.mujoco
        self.data.qpos[:], self.ghost.qpos[:] = qpos, ghost_qpos
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)
        mj.mj_camlight(self.model, self.data)  # the scene's lights (the ghost adds geoms only)
        mj.mj_kinematics(self.model, self.ghost)
        self.camera.lookat[:] = self.data.qpos[:3]
        mj.mjv_updateScene(self.model, self.data, self.option, self.perturb, self.camera, mj.mjtCatBit.mjCAT_ALL,
                           self.scene)
        robot_geoms = self.scene.ngeom
        mj.mjv_addGeoms(self.model, self.ghost, self.option, self.perturb, mj.mjtCatBit.mjCAT_DYNAMIC, self.scene)
        for k in range(robot_geoms, self.scene.ngeom):
            self.scene.geoms[k].rgba[:] = GHOST_RGBA
            self.scene.geoms[k].matid = -1
        mj.mjr_render(self.viewport, self.scene, self.context)
        font = mj.mjtFont.mjFONT_NORMAL
        mj.mjr_overlay(font, mj.mjtGridPos.mjGRID_TOPLEFT, self.viewport, "\n".join(top_lines), "", self.context)
        if bottom_line:
            mj.mjr_overlay(font, mj.mjtGridPos.mjGRID_BOTTOMLEFT, self.viewport, bottom_line, "", self.context)
        mj.mjr_readPixels(self.pixels, None, self.viewport, self.context)
        return np.flipud(self.pixels).copy()

    def close(self):
        self.context.free()
        self.gl.free()


def render_trial(records, libraries, labels, trial, renderer, ffmpeg, out, fps, rate, options):
    """Writes trial_NNNN.mp4, trial_NNNN.png (the contact sheet) and trial_NNNN.json for one trial, up to the control
    of video_stop."""
    stem = out/f"trial_{trial:04d}"
    stop, reason, ends = video_stop(records, trial, tail_controls(options["tail_s"], rate))
    steps = replay_steps(stop + 1, options["stride"])
    states = [panel_states(record, trial, steps) for record in records]
    ghosts = [[ghost_qpos(library.qpos[state["row"]], state["qpos"], options["ghost"]) for state in panel]
              for library, panel in zip(libraries, states)]
    width, height = renderer.viewport.width, renderer.viewport.height
    writer = VideoWriter(ffmpeg, stem.with_suffix(".mp4"), width*len(records), height, fps)
    keep = set(sheet_indices(len(steps)))
    kept = []
    bottom = bottom_line(len(records), options["ghost"])
    try:
        for k, step in enumerate(steps):
            panels = [renderer.render(s[k]["qpos"], ghost[k], caption(label, trial, record, library, s[k], step, rate),
                                      bottom)
                      for record, library, label, s, ghost in zip(records, libraries, labels, states, ghosts)]
            frame = np.concatenate(panels, axis=1)
            writer.write(frame)
            if k in keep:
                kept.append(frame)
    finally:
        writer.close()
    stem.with_suffix(".png").write_bytes(png_bytes(contact_sheet(kept)))
    summary = dict(
        records=[str(record.folder.resolve()) for record in records], labels=labels, trial=trial,
        command_row=int(records[0].arrays["command_row"][trial]), body_row=int(records[0].arrays["body_row"][trial]),
        ended_at_control=[trial_end(record, trial)[0] for record in records],
        end_points=[dict(control=control, reason=why) for control, why in ends], stop_control=stop, stop_reason=reason,
        frames=len(steps), stride=options["stride"], first_step=steps[0], last_step=steps[-1],
        contact_sheet_frames=sorted(keep), playback_speed=fps*options["stride"]/rate, options=options)
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m nablamimic.render_rollouts",
                                     description="Videos of engine_rollouts records by replaying the recorded states.")
    parser.add_argument("--record", type=Path, action="append", required=True,
                        help="record folder (description.json, rollouts.npz); give it twice for two panels side by "
                             "side, which must hold the same trials and the same number of controls")
    parser.add_argument("--trials", type=int, nargs="+", help="trial indices in the order to render (default all)")
    parser.add_argument("--out", type=Path, required=True, help="output folder; must not exist")
    parser.add_argument("--width", type=int, default=480, help="panel width in pixels (even)")
    parser.add_argument("--height", type=int, default=360, help="panel height in pixels (even)")
    parser.add_argument("--fps", type=float, help="video frames per second (default the control rate)")
    parser.add_argument("--stride", type=int, default=1, help="controls per video frame")
    parser.add_argument("--ghost", choices=GHOST_PLACEMENTS, default="world",
                        help="where the command's pose is drawn: at its own world pose (world) or with its root moved "
                             "horizontally to the robot's root at every frame (robot)")
    parser.add_argument("--tail", type=tail_seconds, default=TAIL_S,
                        help="seconds shown after the earlier of the trial's end and the command reaching its clip's "
                             "last row (the later of the two panels'); 'all' or a negative value: every recorded "
                             f"control (default {TAIL_S:g})")
    parser.add_argument("--labels", nargs="+", help="one label per record (default the record folder names)")
    parser.add_argument("--distance", type=float, default=3.5, help="camera distance from the root (m)")
    parser.add_argument("--azimuth", type=float, default=135.0, help="camera azimuth (degrees)")
    parser.add_argument("--elevation", type=float, default=-15.0, help="camera elevation (degrees)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if len(args.record) > 2:
        raise ValueError("give one record, or two to show side by side")
    if args.width < 16 or args.height < 16 or args.width % 2 or args.height % 2:
        raise ValueError(f"the panel size must be even and at least 16 pixels, got {args.width} x {args.height}")
    labels = args.labels or [folder.name for folder in args.record]
    if len(labels) != len(args.record):
        raise ValueError(f"{len(labels)} labels for {len(args.record)} records")
    if args.out.exists():
        raise ValueError(f"the output folder {args.out} exists")
    import_mujoco()
    from . import motion
    rate = motion.REFERENCE_FPS  # controls per second: the references are resampled to the control rate
    fps = rate if args.fps is None else args.fps
    if not 0 < fps < 1000:
        raise ValueError(f"frames per second must be in (0, 1000), got {fps}")
    records = [Record(folder) for folder in args.record]
    if len(records) == 2:
        same_trials(*records)
    trials = select_trials(args.trials, records[0].trials)
    replay_steps(records[0].controls, args.stride)  # refuses an invalid stride before any output
    for record in records:
        check_library_inputs(record)
    library = ReferenceRows(records[0].clips)  # same_trials: both records hold the same clips
    for record in records:
        library.check(record)
        for trial in trials:
            trial_end(record, trial)
    libraries = [library]*len(records)
    model = build_model()
    if model.nq != NQ:
        raise ValueError(f"the model has {model.nq} positions, the record {NQ}")
    for record in records:
        check_model(model, record, trials)
    renderer = PanelRenderer(model, args.width, args.height, args.distance, args.azimuth, args.elevation)
    try:
        ffmpeg = find_encoder()
        args.out.mkdir(parents=True)
        options = dict(width=args.width, height=args.height, fps=fps, stride=args.stride, ghost=args.ghost,
                       tail_s=args.tail, distance=args.distance,
                       azimuth=args.azimuth, elevation=args.elevation, mujoco_gl=os.environ.get("MUJOCO_GL"),
                       scene=str(motion.SCENE))
        for trial in trials:
            summary = render_trial(records, libraries, labels, trial, renderer, ffmpeg, args.out, fps, rate, options)
            print(f"trial {trial}: {summary['frames']} frames to control {summary['stop_control']} "
                  f"({summary['stop_reason']}) -> {args.out/f'trial_{trial:04d}.mp4'}", flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
