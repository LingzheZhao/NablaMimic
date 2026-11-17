"""Readings of one save of the recovery pair (studies/recovery/README.md) from its three nablamimic.engine_rollouts
records and the motion library; no model is loaded.

  python studies/recovery/readings.py --pair studies/recovery/pair.json --scene SCENE --placed PLACED \\
      --twins TWINS --first-rows FIRST_ROWS --out OUT

SCENE is the output of scene.py; PLACED, TWINS and FIRST_ROWS are the rollouts of one checkpoint on its placed.json,
twins.json and first_rows.json. The library is located by LAFAN1_G1_DIR, MOTION_CACHE_DIR and MENAGERIE_DIR, which
must equal the scene's and the rollouts'. OUT must not exist; the program writes save_readings.json and trials.jsonl
(one line per placed start), and prints the save with R (placed starts that get up), J (placed starts that rejoin),
J over R, F (twins with a counted fall), C (first-row clips completed) and E_m (the aligned body-position error E, m).
A rollout without rollouts.npz or without its engine counters, a record of another scene, termination, control count,
weights or checkpoint, a record whose command or body rows are not its scene's, and a library other than the scene's
are refused.

The readings are those of readings_core.py. This program gives the core the record's fields and the library's bodies at
the command row after each control (frame_after). It runs in the trainer's environment.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from declaration import read_declaration
from nablamimic.engine_rollouts import LIBRARY_INPUTS
from readings_core import save_readings

# Each rollout's scene file and the section of pair.json that declares its termination.
ROLLOUTS = {"placed": ("placed.json", "placed_starts"), "twins": ("twins.json", "placed_starts"),
            "first_rows": ("first_rows.json", "first_rows")}
# The core's arrays and the engine_rollouts record fields they are.
RECORD_FIELDS = dict(body_pos_w="body_pos_w", body_quat_w="body_quat_w", command_row_before="frame_before",
                     command_row_after="frame_after", clip_last_row="clip_last_row", available="available", done="done",
                     nonfinite="termination_nonfinite", contacts="engine_contacts",
                     constraint_rows="engine_constraint_rows",
                     contacts_beyond_capacity="engine_contact_capacity_exceeded",
                     rows_beyond_capacity="engine_row_capacity_exceeded")


def rollout_arrays(record, library):
    """The core's arrays of one engine_rollouts record: its fields, and the library's bodies at the command row after
    each control (row 0 where the state is not available)."""
    out = {name: record[field] for name, field in RECORD_FIELDS.items()}
    index = np.where(np.asarray(record["available"], bool), np.asarray(record["frame_after"], np.int64), 0)
    out.update(command_body_pos_w=library.body_pos[index], command_body_quat_w=library.body_quat[index])
    return out


def read_save(declaration, scene_directory, rollouts, library, pelvis, anchor, hidden_clips):
    """The readings of one save from ``rollouts`` {name: directory} (the keys of ROLLOUTS); ``pelvis`` and ``anchor``
    index the pelvis and the torso among the record's tracked bodies."""
    scene_record = json.loads((Path(scene_directory)/"scene.json").read_text())
    scenes = {name: json.loads((Path(scene_directory)/file).read_text()) for name, (file, _) in ROLLOUTS.items()}
    if [trial["command_row"] for trial in scenes["placed"]["trials"]] != \
            [trial["command_row"] for trial in scenes["twins"]["trials"]]:
        raise ValueError(f"{scene_directory}: the twins' command rows are not the placed starts'")
    controls = dict(placed=declaration["placed_starts"]["controls"], twins=declaration["placed_starts"]["controls"],
                    first_rows=scene_record["first_row_controls"])
    descriptions, records = {}, {}
    for name, (file, section) in ROLLOUTS.items():
        termination = declaration[section]["termination"]
        directory = Path(rollouts[name])
        description = json.loads((directory/"description.json").read_text())
        if "engine_diagnostics" not in description or not (directory/"rollouts.npz").is_file():
            raise ValueError(f"{directory}: the rollouts did not complete")
        expected = dict(scene_content=scenes[name], termination=termination, controls=controls[name],
                        actor_key=declaration["actor_key"])
        differing = [key for key, value in expected.items() if description[key] != value]
        if description["settings"]["inputs"] != scene_record["library_inputs"]:
            differing.append("settings.inputs")
        if differing:
            raise ValueError(f"{directory}: {differing} are not those of the {file} rollouts of this study")
        descriptions[name] = description
        with np.load(directory/"rollouts.npz") as stored:
            records[name] = {key: stored[key] for key in stored.files}
        rows = [(trial["command_row"], trial.get("body_row", trial["command_row"])) for trial in scenes[name]["trials"]]
        if list(zip(records[name]["command_row"].tolist(), records[name]["body_row"].tolist())) != rows:
            raise ValueError(f"{directory}: the record's command and body rows are not those of {file}")
    identity = {key: descriptions["placed"][key] for key in ("run", "checkpoint", "completed_updates", "actor_key",
                                                             "run_source_revision")}
    for name, description in descriptions.items():
        if {key: description[key] for key in identity} != identity:
            raise ValueError(f"the {name} rollouts are of another checkpoint than the placed starts'")
    trials = scenes["placed"]["trials"]
    if len(scene_record["trials"]) != len(trials):
        raise ValueError(f"{scene_directory}: scene.json and placed.json hold different trials")
    core = save_readings(*(rollout_arrays(records[name], library) for name in ROLLOUTS), declaration["fall_recovery"],
                         declaration["control_dt_s"], pelvis, anchor, list(library.names),
                         [int(row) for row in library.starts], hidden_clips)
    readings, twin_readings = core["readings"], core["twin_readings"]
    lines = [dict(trial=index, body_clip=trial["body_clip"], command_clip=trial["command_clip"],
                  pelvis_yaw_body_minus_command_rad=trial["pelvis_yaw_body_minus_command_rad"],
                  twin_status=twin_readings[index]["status"], reading=readings[index])
             for index, trial in enumerate(scene_record["trials"])]
    result = dict(save=identity, rollouts={name: dict(directory=str(Path(directory).resolve()),
                                                      source_revision=descriptions[name]["source_revision"],
                                                      gpu=descriptions[name]["gpu"])
                                           for name, directory in rollouts.items()},
                  scene=str(Path(scene_directory).resolve()), trials=len(trials),
                  **core["quantities"],
                  pelvis_yaw_at_placement=[dict(trial=line["trial"], status=line["reading"]["status"],
                                                rad=line["pelvis_yaw_body_minus_command_rad"]) for line in lines],
                  engine=core["engine"], first_rows=core["first_rows"])
    return result, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", type=Path, required=True, help="the declaration, studies/recovery/pair.json")
    parser.add_argument("--scene", type=Path, required=True, help="output directory of scene.py")
    for name in ROLLOUTS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True,
                            help=f"engine_rollouts output on {ROLLOUTS[name][0]}")
    parser.add_argument("--out", type=Path, required=True, help="output directory; must not exist")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"{args.out} exists")
    declaration = read_declaration(args.pair)
    scene_record = json.loads((args.scene/"scene.json").read_text())
    inputs = {key: os.environ.get(key) for key in LIBRARY_INPUTS}
    if inputs != scene_record["library_inputs"]:
        raise ValueError(f"the library inputs {inputs} are not the scene's {scene_record['library_inputs']}")
    from nablamimic.env import ANCHOR, BM_BODIES
    from nablamimic.motion import all_clip_names, load_library
    library = load_library(all_clip_names(), bodies=BM_BODIES)
    if list(library.names) != [clip["clip"] for clip in scene_record["clips"]]:
        raise ValueError("the library's clips are not the scene's")
    hidden = scene_record["recovery_settings"]["hidden_clips"]
    result, lines = read_save(declaration, args.scene, {name: getattr(args, name) for name in ROLLOUTS}, library,
                              BM_BODIES.index("pelvis"), BM_BODIES.index(ANCHOR), hidden)
    args.out.mkdir(parents=True)
    with (args.out/"save_readings.json").open("x") as handle:
        handle.write(json.dumps(result, indent=1, allow_nan=False) + "\n")
    with (args.out/"trials.jsonl").open("x") as handle:
        handle.writelines(json.dumps(line, allow_nan=False) + "\n" for line in lines)
    print(json.dumps(dict(save=result["save"], R=result["R"]["numerator"], J=result["J"]["numerator"],
                          J_over_R=result["J_over_R"]["value"], F=result["F"]["numerator"],
                          C=result["first_rows"]["C"]["completed"], E_m=result["first_rows"]["E_m"]["value"])))


if __name__ == "__main__":
    main()
