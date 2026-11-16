"""The outcome of the recovery pair (studies/recovery/README.md): the declared outcome order of pair.json applied to the
readings of the eight saves, printed and written with every reading it used.

  python studies/recovery/outcome.py --pair studies/recovery/pair.json \\
      --training control:0=RUN --training recovery:0=RUN --training control:1=RUN --training recovery:1=RUN \\
      --save READINGS [--save READINGS ...] --out OUT

RUN is a torch_train output directory; READINGS an output directory of readings.py, whose save names its run and
checkpoint. OUT must not exist; the program writes outcome.json. Each seed's class and the outcome rows are those of
pair.json, read from each save's R (placed starts that get up), J over R (rejoined over the trials that got up with a
readable rejoin), F (twins with a counted fall), C (first-row clips completed), C_getup (completion of each hidden
get-up clip) and E (aligned body-position error). Nothing is pooled across saves or seeds.

How a training ends and when no outcome is given are those of README.md, "Classes, margins and outcomes": training_end
reads the end from DONE, terminal.json (completed before the declared updates: the wall time) and FAILED.json.
"""
import argparse
import json
from pathlib import Path

from declaration import read_declaration, training_roots
from nablamimic.recovery import OPTIONS as RECOVERY_OPTIONS

REPOSITORY = Path(__file__).resolve().parents[2]
OVERFLOW_COUNTERS = ("fwd_contact_overflow_worlds", "fwd_row_overflow_worlds")
INVALID_CONDITION = "Invalid training condition: "  # the trainer's error, followed by {"invalid": ..., "metrics": ...}
# The config.json fields that may differ between the two trainings of a seed: the recovery options, the output, the
# arguments file, and where a job put the compiled engine library. The engine's identity fields (engine.key,
# engine.sources_sha256, engine.consts_sha256) and its capacities must be equal.
PAIR_DIFFERENCES = (*RECOVERY_OPTIONS, "out", "config", "engine.path", "engine.artifact")


def invalid_counters(error):
    """The non-zero counters of the trainer's invalid training condition in a FAILED.json error, {} for another
    error."""
    if INVALID_CONDITION not in error:
        return {}
    text = error.split(INVALID_CONDITION, 1)[1]
    return json.loads(text[:text.rindex("}") + 1])["invalid"]


def flat(config, prefix=""):
    """A config.json with its nested objects flattened into dotted keys (engine.key); an empty object stays a value."""
    out = {}
    for key, value in config.items():
        if isinstance(value, dict) and value:
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def training_end(root, updates):
    """How a training ended: completed, or the cause of an early end (capacity_overflow, wall_time or other)."""
    root = Path(root)
    if (root/"FAILED.json").is_file():
        failed = json.loads((root/"FAILED.json").read_text())
        overflow = any(invalid_counters(failed["error"]).get(name) for name in OVERFLOW_COUNTERS)
        return dict(completed=False, cause="capacity_overflow" if overflow else "other",
                    completed_updates=failed["completed_updates"], detail=failed["error"])
    if not (root/"terminal.json").is_file():
        return dict(completed=False, cause="other", completed_updates=None, detail="no terminal.json or FAILED.json")
    terminal = json.loads((root/"terminal.json").read_text())
    done = terminal["completed_updates"]
    if terminal["status"] == "completed" and done == updates and (root/"DONE").is_file():
        return dict(completed=True, cause=None, completed_updates=done, detail=None)
    cause = "wall_time" if terminal["status"] == "completed" and done < updates else "other"
    return dict(completed=False, cause=cause, completed_updates=done,
                detail=f"terminal.json status {terminal['status']}")


def declared_recovery_options(declaration):
    """The recovery options of each condition as config.json records them."""
    from nablamimic.params import read_params
    from nablamimic.torch_train import build_parser
    out = {}
    for condition, arguments in declaration["trainings"]["conditions"].items():
        parsed = vars(build_parser().parse_args(read_params(REPOSITORY/arguments) + ["--out", "unused"]))
        out[condition] = json.loads(json.dumps({name: parsed[name] for name in RECOVERY_OPTIONS}))
    return out


def training_problems(declaration, configs, options):
    """The declaration's unreadable_when on the trainings' config.json files {(condition, seed): config}."""
    trainings = declaration["trainings"]
    declared = dict(iters=trainings["updates"], ckpt_every=trainings["checkpoint_every"], hours=trainings["hours"])
    problems = []
    for (condition, seed), config in configs.items():
        values = dict(seed=seed, **declared)
        problems += [f"{condition} seed {seed}: config.json {name} {config[name]!r}, declared {value!r}"
                     for name, value in values.items() if config[name] != value]
        if {name: config[name] for name in RECOVERY_OPTIONS} != options[condition]:
            problems.append(f"{condition} seed {seed}: its recovery options are not those of "
                            f"{trainings['conditions'][condition]}")
    for seed in trainings["seeds"]:
        if ("control", seed) in configs and ("recovery", seed) in configs:
            a, b = flat(configs["control", seed]), flat(configs["recovery", seed])
            differing = sorted(key for key in set(a) | set(b)
                               if key not in PAIR_DIFFERENCES and (key not in a or key not in b or a[key] != b[key]))
            if differing:
                problems.append(f"seed {seed}: the two config.json files differ in {differing}")
    return problems


def save_values(reading):
    """The numbers the outcome reads from one save's readings; None where unavailable."""
    first_rows = reading["first_rows"]
    return dict(R=reading["R"]["numerator"], J_over_R=reading["J_over_R"]["value"], F=reading["F"]["numerator"],
                C=first_rows["C"]["completed"], E_m=first_rows["E_m"]["value"], C_getup=first_rows["C_getup"])


def margins(declaration, values):
    declared = declaration["margins"]
    seeds, saves = declaration["trainings"]["seeds"], declaration["saves"]
    out = {name: declared[name] for name in ("near_zero", "gets_up_R", "gets_up_lead", "rejoins_J_over_R",
                                             "tracking_F", "control_within")}
    for updates in saves:
        control = [values["control", seed, updates] for seed in seeds]
        out[f"tracking_C_{updates}"] = max(declared["tracking_C"]["least"], abs(control[0]["C"] - control[1]["C"]))
        out[f"tracking_E_m_{updates}"] = max(declared["tracking_E_m"]["least"],
                                             abs(control[0]["E_m"] - control[1]["E_m"]))
    recovery = [values["recovery", seed, saves[-1]]["R"] for seed in seeds]
    out["still_rising"] = max(abs(recovery[0] - recovery[1]), declared["still_rising"]["least"])
    return out


def classes(declaration, values):
    """Each seed's class of its recovery save at the last update, against the larger control R there."""
    declared, seeds, last = declaration["margins"], declaration["trainings"]["seeds"], declaration["saves"][-1]
    larger = max(values["control", seed, last]["R"] for seed in seeds)
    out = {}
    for seed in seeds:
        r = values["recovery", seed, last]["R"]
        if r <= declared["near_zero"]:
            out[seed] = "near_zero"
        elif r >= declared["gets_up_R"] and r >= larger + declared["gets_up_lead"]:
            out[seed] = "gets_up"
        else:
            out[seed] = "some"
    return out


def tracking_within(values, bounds, seed, updates, last):
    recovery, control = values["recovery", seed, last], values["control", seed, updates]
    terms = dict(C=recovery["C"] >= control["C"] - bounds[f"tracking_C_{updates}"],
                 E=recovery["E_m"] <= control["E_m"] + bounds[f"tracking_E_m_{updates}"],
                 F=recovery["F"] <= control["F"] + bounds["tracking_F"])
    return dict(within=all(terms.values()), terms=terms)


def outcome_rules(declaration, values):
    """Whether each outcome row of the declaration holds, with the readings the rows read. run_stopped is decided by
    read_pair before any save is read, since a stopped pair lacks saves; it is False here, so that the rows are the
    declaration's whole order."""
    seeds, (first, last) = declaration["trainings"]["seeds"], declaration["saves"]
    bounds = margins(declaration, values)
    seed_class = classes(declaration, values)
    gets_up = [seed_class[seed] == "gets_up" for seed in seeds]
    near = [seed_class[seed] == "near_zero" for seed in seeds]
    # J over R enters only the rows that need both seeds gets_up. Where no recovered trial's rejoin was readable it is
    # unavailable: neither below the margin nor at it, so the rows that need its value at that seed do not hold.
    j_over_r = [values["recovery", seed, last]["J_over_R"] for seed in seeds]
    below = [value is not None and value < bounds["rejoins_J_over_R"] for value in j_over_r]
    rejoins = [value is not None and value >= bounds["rejoins_J_over_R"] for value in j_over_r]
    tracking = {(seed, updates): tracking_within(values, bounds, seed, updates, last)
                for seed in seeds for updates in (first, last)}
    rise = {seed: values["recovery", seed, last]["R"] - values["recovery", seed, first]["R"] for seed in seeds}
    getup_completions = [done for seed in seeds for updates in (first, last)
                         for done in values["control", seed, updates]["C_getup"].values()]
    up_and_rejoin = all(gets_up) and all(rejoins)
    rules = dict(
        run_stopped=False,
        split_seeds=(gets_up[0] and near[1]) or (gets_up[1] and near[0]),
        control_within=all(values["control", seed, last]["R"] > bounds["near_zero"]
                           and abs(values["control", seed, last]["R"] - values["recovery", seed, last]["R"])
                           <= bounds["control_within"] for seed in seeds),
        precondition_unmet=not any(gets_up) and not any(getup_completions),
        still_rising=any(not gets_up[index] and rise[seed] > bounds["still_rising"]
                         for index, seed in enumerate(seeds)),
        no_get_up=all(near),
        some_get_up=not any(gets_up),
        gets_up_without_rejoining=all(gets_up) and any(below),
        gets_up_and_tracks=up_and_rejoin and all(tracking[seed, last]["within"] for seed in seeds),
        tracking_cost_at_exposure=up_and_rejoin and all(
            not tracking[seed, last]["within"] and tracking[seed, first]["within"] for seed in seeds),
        tracking_worse=up_and_rejoin and all(
            not tracking[seed, last]["within"] and not tracking[seed, first]["within"] for seed in seeds),
        no_call=True)
    return dict(rules=rules, classes=seed_class, margins=bounds, rise=rise,
                J_over_R_at_least_margin=dict(zip(seeds, rejoins)),
                J_over_R_unavailable_at_seeds=[seed for seed, value in zip(seeds, j_over_r) if value is None],
                tracking={f"seed {seed} against control {updates}": item for (seed, updates), item in tracking.items()},
                larger_control_R=max(values["control", seed, last]["R"] for seed in seeds))


def outcome(declaration, values):
    """The first outcome row of the declaration that holds, with every row's value."""
    found = outcome_rules(declaration, values)
    names = [item["name"] for item in declaration["outcomes"]]
    if names != list(found["rules"]):
        raise ValueError(f"the declaration's outcome rows {names} are not the rows of this program")
    return dict(outcome=next(name for name in names if found["rules"][name]),
                rows_in_order=[dict(name=name, holds=found["rules"][name]) for name in names],
                **{key: value for key, value in found.items() if key != "rules"})


def read_pair(declaration, readings, trainings, configs, ends, options):
    """The pair's outcome from the save ``readings``, the training roots {(condition, seed): root} with their
    config.json files and ends, and the declared recovery options of each condition."""
    trainings_declared = declaration["trainings"]
    wanted = [(condition, seed, updates) for condition in trainings_declared["conditions"]
              for seed in trainings_declared["seeds"] for updates in declaration["saves"]]
    runs = {str(Path(root).resolve()): key for key, root in trainings.items()}
    by_save = {}
    for reading in readings:
        save = reading["save"]
        if save["run"] not in runs:
            raise ValueError(f"the save of {save['run']} is not of a training given")
        key = (*runs[save["run"]], save["completed_updates"])
        if key in by_save:
            raise ValueError(f"two readings of the save {key}")
        by_save[key] = reading
    values, unreadable = {}, []
    for key in wanted:
        if key not in by_save:
            unreadable.append(f"{key}: no reading of this save")
            continue
        values[key] = save_values(by_save[key])
        if by_save[key]["trials"] != declaration["placed_starts"]["trials"]:
            unreadable.append(f"{key}: {by_save[key]['trials']} placed starts, declared "
                              f"{declaration['placed_starts']['trials']}")
        unreadable += [f"{key}: {name} unavailable" for name, value in values[key].items()
                       if value is None and name != "J_over_R"]
    unreadable += [f"{condition} seed {seed}: training not given" for condition in trainings_declared["conditions"]
                   for seed in trainings_declared["seeds"] if (condition, seed) not in trainings]
    unreadable += training_problems(declaration, configs, options)
    stopped = [f"{condition} seed {seed}: {end['cause']} at {end['completed_updates']} updates ({end['detail']})"
               for (condition, seed), end in ends.items() if end["cause"] in ("capacity_overflow", "wall_time")]
    unreadable += [f"{condition} seed {seed}: the training ended otherwise at {end['completed_updates']} updates "
                   f"({end['detail']})" for (condition, seed), end in ends.items() if end["cause"] == "other"]
    out = dict(saves={f"{c}/seed {s}/{u} updates": values.get((c, s, u)) for c, s, u in wanted},
               trainings={f"{c} seed {s}": dict(root=str(trainings[c, s]), end=ends[c, s],
                                                **{name: configs[c, s].get(name) for name in (
                                                    "gpu", "git_commit", "declared_environment", "torch_version")})
                          for c, s in trainings},
               stopped=stopped, unreadable=unreadable)
    if stopped:
        out["outcome"] = dict(outcome="run_stopped", note="the first row of the declaration; the saves that exist are "
                                                          "read as stated")
    elif unreadable:
        out["outcome"] = dict(outcome=None, note="unreadable by the declaration's unreadable_when; no row is taken")
    else:
        out["outcome"] = outcome(declaration, values)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", type=Path, required=True, help="the declaration, studies/recovery/pair.json")
    parser.add_argument("--training", action="append", required=True, metavar="CONDITION:SEED=RUN",
                        help="a torch_train output directory of one condition and seed")
    parser.add_argument("--save", action="append", type=Path, required=True, metavar="READINGS",
                        help="an output directory of readings.py")
    parser.add_argument("--out", type=Path, required=True, help="output directory; must not exist")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError(f"{args.out} exists")
    declaration = read_declaration(args.pair)
    trainings = training_roots(declaration, args.training)
    configs = {key: json.loads((root/"config.json").read_text()) for key, root in trainings.items()}
    ends = {key: training_end(root, declaration["trainings"]["updates"]) for key, root in trainings.items()}
    readings = [json.loads((directory/"save_readings.json").read_text()) for directory in args.save]
    result = read_pair(declaration, readings, trainings, configs, ends, declared_recovery_options(declaration))
    result["save_readings"] = [str(directory.resolve()) for directory in args.save]
    text = json.dumps(result, indent=1, allow_nan=False)
    args.out.mkdir(parents=True)
    with (args.out/"outcome.json").open("x") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
