"""The scene files of the recovery pair (studies/recovery/README.md) for nablamimic.engine_rollouts: the placed lying
starts, their twins and the first row of every clip, on the 40-clip library the trainings use.

  python studies/recovery/scene.py --pair studies/recovery/pair.json --out SCENE

The library is the one torch_train builds for --clips all: the clips of LAFAN1_G1_DIR in sorted order, cached in
MOTION_CACHE_DIR, on the model of MENAGERIE_DIR. The recovery setting is that of the recovery condition's arguments
file. SCENE must not exist; the program writes into it placed.json, twins.json and first_rows.json (scene files of
engine_rollouts) and scene.json (every trial's rows, clips and draws, the controls of the first-row rollouts, the row
counts and the library inputs).

The placed starts: body rows evenly spaced over the start rows, command rows evenly spaced over the rows with enough
upright rows ahead, both by rounding a float64 linspace over the row indices; unit uniforms from one torch CPU
generator in the order position, rotation, velocity, joints; then a permutation from the same generator pairs the
command rows with the body rows.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch

from declaration import read_declaration
from nablamimic import recovery
from nablamimic.engine_rollouts import LIBRARY_INPUTS, UNIFORMS
from nablamimic.params import read_params
from nablamimic.torch_env import quat_mul

REPOSITORY = Path(__file__).resolve().parents[2]
# rad, TorchTrackingEnv's reset_rotation_limits, which the environment sets when it is built.
RESET_ROTATION_HALF_WIDTHS = (0.1, 0.1, 0.2)


def recovery_settings(arguments_file):
    """The RecoverySettings of a torch_train arguments file."""
    from nablamimic.torch_train import build_parser
    settings = recovery.settings_from_args(build_parser().parse_args(read_params(arguments_file) + ["--out", "unused"]))
    if settings is None:
        raise ValueError(f"{arguments_file} sets no recovery options")
    return settings


def evenly(values, count):
    """``count`` entries of ``values`` at the rounded points of a float64 linspace over its indices."""
    return values[torch.linspace(0, values.shape[0] - 1, count, dtype=torch.float64).round().long()]


def draws(seed, count):
    """The unit uniforms {name: [count, n] float32}, drawn in the order of engine_rollouts.UNIFORMS, and then the
    pairing permutation [count], from one torch CPU generator seeded with ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    uniforms = {name: 2.0*torch.rand((count, width), generator=generator, dtype=torch.float32) - 1.0
                for name, width in UNIFORMS.items()}
    return uniforms, torch.randperm(count, generator=generator)


def placed_yaw(qpos, body, command, rotation):
    """The pelvis yaw of the placed body minus the command's at placement (rad, wrapped): the reset's rotation of the
    unit uniforms ``rotation`` premultiplied on the body row's root orientation, as TorchTrackingEnv.reset applies
    it."""
    rpy = rotation*torch.tensor(RESET_ROTATION_HALF_WIDTHS)
    angle = torch.linalg.vector_norm(rpy, dim=-1, keepdim=True)
    delta = torch.cat((torch.cos(angle/2), rpy/(angle + 1e-9)*torch.sin(angle/2)), -1)
    placed = quat_mul(delta, qpos[body, 3:7])
    return recovery.wrap(recovery.heading(placed) - recovery.heading(qpos[command, 3:7]))


def scene(library, settings, placed):
    """The scene files and their record on ``library`` (a nablamimic.motion.MotionLibrary) under the recovery
    ``settings`` and the declaration's ``placed_starts``."""
    qpos = torch.as_tensor(library.qpos)
    last = torch.as_tensor(library.last)
    starts = [int(row) for row in library.starts]
    tables = recovery.recovery_rows(qpos, library.starts, last, library.names, settings)
    ahead = dataclasses.replace(settings, command_upright_rows=placed["command_upright_rows"])
    commandable = recovery.recovery_rows(qpos, library.starts, last, library.names, ahead)["command_frames"]
    count = placed["trials"]
    body = evenly(tables["start_frames"].long(), count)
    command = evenly(commandable.long(), count)
    uniforms, permutation = draws(placed["uniforms_seed"], count)
    command = command[permutation]
    yaw = placed_yaw(qpos, body, command, uniforms["rotation"])

    def where(row):
        clip = max(index for index, first in enumerate(starts) if first <= row)
        return library.names[clip], row - starts[clip]

    trials = []
    for index in range(count):
        body_clip, body_in_clip = where(int(body[index]))
        command_clip, command_in_clip = where(int(command[index]))
        trials.append(dict(trial=index, body_row=int(body[index]), body_clip=body_clip, body_row_in_clip=body_in_clip,
                           command_row=int(command[index]), command_clip=command_clip,
                           command_row_in_clip=command_in_clip, permutation=int(permutation[index]),
                           pelvis_yaw_body_minus_command_rad=float(yaw[index])))
    placed_trials = [dict(command_row=item["command_row"], body_row=item["body_row"],
                          reset_uniforms={name: uniforms[name][index].tolist() for name in UNIFORMS})
                     for index, item in enumerate(trials)]
    clips = [dict(clip=name, first_row=first, last_row=int(library.last[first]),
                  controls=int(library.last[first]) - first) for name, first in zip(library.names, starts)]
    record = dict(recovery_settings=dataclasses.asdict(settings), rows=tables["counts"],
                  command_rows_with_upright_rows_ahead=int(commandable.numel()), placed_starts=placed,
                  first_row_controls=max(clip["controls"] for clip in clips), clips=clips, trials=trials)
    files = {"placed.json": {"trials": placed_trials},
             "twins.json": {"trials": [{"command_row": item["command_row"]} for item in trials]},
             "first_rows.json": {"trials": [{"command_row": clip["first_row"]} for clip in clips]}}
    return files, record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", type=Path, required=True, help="the declaration, studies/recovery/pair.json")
    parser.add_argument("--out", type=Path, required=True, help="output directory; must not exist")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"{args.out} exists")
    declaration = read_declaration(args.pair)
    from nablamimic.env import BM_BODIES
    from nablamimic.motion import all_clip_names, load_library
    names = all_clip_names()
    if len(names) != 40:
        raise ValueError(f"the library holds {len(names)} clips, not the 40 of --clips all")
    library = load_library(names, bodies=BM_BODIES)
    settings = recovery_settings(REPOSITORY/declaration["trainings"]["conditions"]["recovery"])
    files, record = scene(library, settings, declaration["placed_starts"])
    record["library_inputs"] = {key: os.environ.get(key) for key in LIBRARY_INPUTS}
    args.out.mkdir(parents=True)
    for name, content in {**files, "scene.json": record}.items():
        with (args.out/name).open("x") as handle:
            handle.write(json.dumps(content, indent=1, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=record["rows"], first_row_controls=record["first_row_controls"],
                          command_rows_with_upright_rows_ahead=record["command_rows_with_upright_rows_ahead"])))


if __name__ == "__main__":
    main()
