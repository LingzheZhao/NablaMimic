"""The tracking exposure and the switch yaw of the recovery pair's trainings (studies/recovery/README.md), from their
log.jsonl and config.json.

  python studies/recovery/exposure.py --pair studies/recovery/pair.json \\
      --training control:0=RUN --training recovery:0=RUN --training control:1=RUN --training recovery:1=RUN --out OUT

OUT must not exist; the program writes exposure.json. At each declared save, the tracking window rows a training has
trained on (RECOVERY.md, "Records"): a recovery training's recovery_tracking_window_rows_total of that update's log
row; a control training's updates times num_envs times horizon. Beside them, for each recovery save, the first control
update whose tracking rows reach the recovery save's, and the mean yaw at the switches back to the command up to the
save, weighted by the switches of each log row. The pair's comparisons are those of its outcome order, recovery 3,000
against control 3,000 and control 1,500; the matched update enters none of them. A log row missing at a declared save,
or a log whose recovery columns do not fit its training's condition, is refused.
"""
import argparse
import json
import math
from pathlib import Path

from declaration import read_declaration, training_roots


def training_exposure(config, rows, condition, saves):
    """The tracking window rows at each save and, for a recovery training, the switch yaw up to each save."""
    recovering = config["recovery_share"] is not None
    if recovering != (condition == "recovery") or any(any(key.startswith("recovery_") for key in row) != recovering
                                                      for row in rows):
        raise ValueError(f"the {condition} training's config.json and log do not fit its condition")
    by_update = {row["completed_updates"]: row for row in rows}
    per_update = config["num_envs"]*config["horizon"]
    missing = [save for save in saves if save not in by_update]
    if missing:
        raise ValueError(f"the {condition} log holds no row at updates {missing}")
    out = dict(window_rows_per_update=per_update)
    if not recovering:
        out["tracking_window_rows"] = {str(save): save*per_update for save in saves}
        return out
    out["tracking_window_rows"] = {str(save): by_update[save]["recovery_tracking_window_rows_total"]
                                   for save in saves}
    out["switch_yaw"] = {}
    for save in saves:
        switched = [(row["recovery_switches"], row["recovery_switch_yaw_mean_rad"]) for row in rows
                    if row["completed_updates"] <= save and row["recovery_switches"]]
        total = sum(count for count, _ in switched)
        out["switch_yaw"][str(save)] = dict(
            switches=total, mean_rad=sum(count*value for count, value in switched)/total if total else None)
    return out


def exposure(declaration, logs):
    """The readings of the trainings {(condition, seed): (config, log rows)}."""
    saves = declaration["saves"]
    out = {f"{condition} seed {seed}": training_exposure(config, rows, condition, saves)
           for (condition, seed), (config, rows) in logs.items()}
    for seed in declaration["trainings"]["seeds"]:
        recovery, control = out.get(f"recovery seed {seed}"), out.get(f"control seed {seed}")
        if recovery and control:
            recovery["matched_control_update"] = {
                save: math.ceil(rows/control["window_rows_per_update"])
                for save, rows in recovery["tracking_window_rows"].items()}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", type=Path, required=True, help="the declaration, studies/recovery/pair.json")
    parser.add_argument("--training", action="append", required=True, metavar="CONDITION:SEED=RUN",
                        help="a torch_train output directory of one condition and seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory; must not exist")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"{args.out} exists")
    declaration = read_declaration(args.pair)
    logs = {key: (json.loads((root/"config.json").read_text()),
                  [json.loads(line) for line in (root/"log.jsonl").read_text().splitlines()])
            for key, root in training_roots(declaration, args.training).items()}
    result = exposure(declaration, logs)
    text = json.dumps(result, indent=1, allow_nan=False)
    args.out.mkdir(parents=True)
    with (args.out/"exposure.json").open("x") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
