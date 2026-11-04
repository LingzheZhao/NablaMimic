"""Write a manifest of a reference cache (nablamimic.motion's per-clip npz files) for readers outside this package.

A reader of the cache needs the clip order (it sets the global frame ids and the reset bins) and the file of each clip.
The file names carry a tag computed from the body list (motion.cache_file_name); the manifest states them, so a reader
takes the names from the manifest and never recomputes the tag or lists the directory.

The manifest is JSON, written once (an existing --out is refused). Its fields:

  format              "nablamimic-reference-manifest-v1"
  artifact_id         the identity the caller publishes this cache under (--artifact-id)
  cache_directory     the cache directory: relative to the manifest's directory when the cache lies inside it,
                      otherwise absolute; a reader opens <manifest directory>/<cache_directory>/<cache_file>
  frame_rate_hz       the rate of the processed clips (motion.REFERENCE_FPS)
  bodies              the tracked bodies, in the order of the body axis (env.BM_BODIES)
  arrays              per array: dtype on disk, shape after the frame axis, frame and convention
  clip_selection      "all" (every clip of scripts/lafan1_g1.sha256) or "listed" (the names given)
  clip_order          "sorted_clip_names" for "all" (motion.all_clip_names), "as_listed" otherwise (train.py keeps an
                      explicit --clips list in its given order)
  source_dataset      the download identity of the CSVs (motion.LAFAN1_DATASET, motion.LAFAN1_REVISION)
  clips               in library order: name, cache_file, frames, first_row and last_row (global frame ids of the
                      concatenated library, as MotionLibrary.starts and .last), source_file in the dataset
  total_frames        the sum of the clips' frames

  python -m nablamimic.export_reference_manifest --cache-dir DIR --clips all --artifact-id ID --out DIR/../manifest.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from .motion import CACHE_KEYS, LAFAN1_DATASET, LAFAN1_REVISION, REFERENCE_FPS, cache_file_name
from .params import REPO

FORMAT = "nablamimic-reference-manifest-v1"
CLIP_CHECKSUMS = REPO / "scripts" / "lafan1_g1.sha256"  # the fork's record of the 40 dataset files
DOWNLOAD_SCRIPT = REPO / "scripts" / "download_lafan1.sh"

# Per array: trailing shape after the frame axis (None: the model's nq / nv, equal across clips; "bodies": the number of
# tracked bodies), and its meaning (motion.process_clip).
ARRAYS = {
    "qpos": ((None,), "generalized position: pelvis free joint (position, quaternion w x y z), then the joints in "
                      "actuator order; floor-aligned"),
    "qvel": ((None,), "generalized velocity: pelvis free joint (world linear velocity, body-frame angular velocity), "
                      "then the joint velocities; mj_differentiatePos over neighbouring frames"),
    "body_pos": (("bodies", 3), "world position of each tracked body origin"),
    "body_quat": (("bodies", 4), "world orientation of each tracked body, quaternion w x y z"),
    "body_lin_vel": (("bodies", 3), "world-frame linear velocity of each tracked body origin"),
    "body_ang_vel": (("bodies", 3), "world-frame angular velocity of each tracked body"),
}


def recorded_clip_names(path=CLIP_CHECKSUMS):
    """The clip names of the dataset files this repository records, sorted (what clips=all trains on: the download
    script refuses a directory that does not hold exactly these files)."""
    names = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = re.fullmatch(r"[0-9a-f]{64}  (\S+)\.csv", line.strip())
        if m is None:
            raise ValueError(f"{path}: unexpected line {line!r}")
        names.append(m.group(1))
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{path}: no clip names or repeated names")
    return sorted(names)


def check_download_identity(script=DOWNLOAD_SCRIPT):
    """The download script must fetch the dataset revision the manifest declares."""
    text = Path(script).read_text()
    for line in (f"REPO={LAFAN1_DATASET}", f"REV={LAFAN1_REVISION}"):
        if line not in text.splitlines():
            raise ValueError(f"{script} does not contain the line {line!r}")


def read_clip(path, bodies):
    """Frames, nq and nv of one cached clip; refuses other keys, shapes and non-finite values."""
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(CACHE_KEYS):
            raise ValueError(f"{path}: arrays {sorted(z.files)}, expected {sorted(CACHE_KEYS)}")
        arrays = {k: z[k] for k in CACHE_KEYS}
    frames = arrays["qpos"].shape[0]
    nq, nv = arrays["qpos"].shape[1:], arrays["qvel"].shape[1:]
    if len(nq) != 1 or len(nv) != 1 or frames < 2:
        raise ValueError(f"{path}: qpos {arrays['qpos'].shape}, qvel {arrays['qvel'].shape}")
    for k, ((*trailing,), _) in ARRAYS.items():
        want = tuple({None: (nq if k == "qpos" else nv)[0], "bodies": len(bodies)}.get(d, d) for d in trailing)
        if arrays[k].shape != (frames, *want):
            raise ValueError(f"{path}: {k} has shape {arrays[k].shape}, expected {(frames, *want)}")
        if arrays[k].dtype.kind != "f" or not np.all(np.isfinite(arrays[k])):
            raise ValueError(f"{path}: {k} is not a finite floating-point array")
    return frames, nq[0], nv[0], {k: str(v.dtype) for k, v in arrays.items()}


def build_manifest(cache_dir, clips, artifact_id, out, bodies, fps=REFERENCE_FPS, checksums=CLIP_CHECKSUMS):
    """The manifest dict for the clips of cache_dir. clips: ["all"] or explicit names (kept in the given order)."""
    if not artifact_id:
        raise ValueError("an artifact id is required")
    cache_dir, out = Path(cache_dir).resolve(), Path(out).resolve()
    recorded = recorded_clip_names(checksums)
    if list(clips) == ["all"]:
        names, selection, order = recorded, "all", "sorted_clip_names"
    else:
        names, selection, order = list(clips), "listed", "as_listed"
        if "all" in names or len(set(names)) != len(names):
            raise ValueError(f"clips: 'all' alone or distinct names, got {names}")
        unknown = [n for n in names if n not in recorded]
        if unknown:
            raise ValueError(f"clips without a recorded dataset file in {checksums}: {unknown}")
    entries, offset, dims, dtypes = [], 0, None, None
    for name in names:
        file = cache_file_name(name, fps, bodies)
        path = cache_dir / file
        if not path.is_file():
            raise FileNotFoundError(f"missing cache file of clip {name}: {path}")
        frames, nq, nv, dt = read_clip(path, bodies)
        if dims is None:
            dims, dtypes = (nq, nv), dt
        elif (nq, nv) != dims or dt != dtypes:
            raise ValueError(f"{path}: nq, nv {nq, nv} and dtypes {dt} differ from the first clip's {dims} {dtypes}")
        entries.append(dict(name=name, cache_file=file, frames=frames, first_row=offset, last_row=offset + frames - 1,
                            source_file=f"g1/{name}.csv"))
        offset += frames
    try:
        cache_ref = str(cache_dir.relative_to(out.parent))
    except ValueError:
        cache_ref = str(cache_dir)
    arrays = {}
    for k, ((*trailing,), meaning) in ARRAYS.items():
        shape = [{None: dims[0] if k == "qpos" else dims[1], "bodies": len(bodies)}.get(d, d) for d in trailing]
        arrays[k] = dict(dtype=dtypes[k], shape_after_frames=shape, meaning=meaning)
    return dict(
        format=FORMAT, artifact_id=artifact_id, cache_directory=cache_ref, frame_rate_hz=float(fps),
        bodies=list(bodies), arrays=arrays, clip_selection=selection, clip_order=order,
        source_dataset=dict(repository=LAFAN1_DATASET, repository_type="dataset", revision=LAFAN1_REVISION,
                            files="g1/<clip>.csv"),
        clips=entries, total_frames=offset)


def write_json_once(obj, out):
    """Write obj as JSON to out, refusing an existing path."""
    out = Path(out)
    with open(out, "x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def read_manifest(path):
    """A manifest written by this module (format checked)."""
    m = json.loads(Path(path).read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{path}: format {m.get('format')!r}, expected {FORMAT!r}")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nablamimic.export_reference_manifest",
                                 description="Write the JSON manifest of a reference cache directory: clip order, "
                                             "cache file names, frame counts and the dataset's download identity.")
    ap.add_argument("--cache-dir", required=True, help="the directory of the per-clip npz files (MOTION_CACHE_DIR)")
    ap.add_argument("--clips", nargs="+", required=True,
                    help="all (every recorded LAFAN1 clip, sorted, as training with --clips all) or clip names in "
                         "library order")
    ap.add_argument("--artifact-id", required=True, help="the identity this reference cache is published under")
    ap.add_argument("--out", required=True, help="the manifest path (.json); refused if it exists")
    args = ap.parse_args(argv)
    from .env import BM_BODIES  # the bodies training loads (train.py, evaluate.py: bodies=BM_BODIES)
    if not args.out.endswith(".json"):
        ap.error("--out must name a .json file")
    if Path(args.out).exists():
        ap.error(f"{args.out} exists")
    check_download_identity()
    manifest = build_manifest(args.cache_dir, args.clips, args.artifact_id, args.out, BM_BODIES)
    write_json_once(manifest, args.out)
    print(f"{args.out}: {len(manifest['clips'])} clips, {manifest['total_frames']} frames at "
          f"{manifest['frame_rate_hz']:g} Hz, cache {manifest['cache_directory']}")


if __name__ == "__main__":
    main()
