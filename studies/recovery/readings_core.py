"""The readings of one save of the recovery pair from arrays: numpy and the standard library only, with no file
reading and no trainer, simulator or motion library. The rules are those of README.md, "Readings"; rollouts of any
trainer can be read by building these arrays from its records.

A rollout is a dict of these arrays, with N controls, T trials and B tracked bodies (values after a trial's end are
not read):

  body_pos_w [N, T, B, 3], body_quat_w [N, T, B, 4]
      float: world positions and WXYZ orientations of the tracked bodies after each control.
  command_body_pos_w [N, T, B, 3], command_body_quat_w [N, T, B, 4]
      float: the command's bodies at the command row after each control; any value where the state is not available.
  command_row_before, command_row_after, clip_last_row [N, T]
      int: the command row before and after each control, and the last row of the clip of the row after it. Of the
      rows before, the first-row reading reads only control 0's, which must be the clip's first row.
  available, done, nonfinite [N, T]
      bool: the state after the control is recorded and finite; the control ends the trial; the state after the
      control is non-finite (the termination's non-finite term).
  contacts, constraint_rows [N, T]
      int: the contacts and constraint rows the engine used in the control's step.
  contacts_beyond_capacity, rows_beyond_capacity [N, T]
      bool: contacts or constraint rows beyond the engine's capacity were dropped in the step.

Beside the rollouts the functions take the fall_recovery section of pair.json (``settings``), the control period
(``dt``, s), and the indices of the pelvis (``pelvis``) and of the anchor body, the torso (``anchor``), among the B
tracked bodies. The functions on one trial take its columns, one row per control (K rows, K <= N). The rotations are
numpy float64 functions, which give NaN for an invalid quaternion.

The readings of a save: R (placed starts that get up), J (placed starts that rejoin), J over R, F (twins with a counted
fall), C (first-row clips completed, a count of clips), C_getup (completion of each hidden get-up clip) and E (the
aligned body-position error of the first rows, m).
"""
import math

import numpy as np

RECOVERED = ("rejoined", "recovered_rejoin_unavailable", "recovered_fell_again", "recovered_not_rejoined_at_horizon",
             "recovered_ended_early")
STATUSES = ("numerical_failure", "placed_start_outside_fall_rule", "fall_rule_met_not_counted",
            "fall_rule_met_not_counted_ended_early", "no_fall", "no_fall_ended_early", "fell_not_recovered_at_horizon",
            "fell_ended_early", *RECOVERED)
TIME_TOLERANCE_S = 1e-9
# The arrays of a rollout and the number of their trailing dimensions after [N, T].
FIELDS = dict(body_pos_w=2, body_quat_w=2, command_body_pos_w=2, command_body_quat_w=2, command_row_before=0,
              command_row_after=0, clip_last_row=0, available=0, done=0, nonfinite=0, contacts=0, constraint_rows=0,
              contacts_beyond_capacity=0, rows_beyond_capacity=0)


# ------------------------------------------------------------------------------------------------ rotations

def valid_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    return np.isfinite(q).all(-1) & (np.square(q).sum(-1) > 1e-12)


def up_axis_z(q):
    """The world z component of the body z axis of WXYZ quaternions [..., 4]; NaN for an invalid one."""
    q = np.asarray(q, dtype=np.float64)
    valid = valid_quaternion(q)
    w, x, y, z = np.moveaxis(np.where(valid[..., None], q, 1.0), -1, 0)
    return np.where(valid, (w*w - x*x - y*y + z*z)/(w*w + x*x + y*y + z*z), np.nan)


def yaw(q):
    """The world z-yaw of WXYZ quaternions [..., 4], atan2(R10, R00) of the normalised quaternion's rotation."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q/np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def rotate_z(vectors, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    x, y = vectors[..., 0], vectors[..., 1]
    return np.stack((cosine*x - sine*y, sine*x + cosine*y, vectors[..., 2]), -1)


def quaternion_product(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw), -1)


def local_positions(bodies, pelvis_quaternion, pelvis):
    """[K, B, 3] bodies in their own pelvis heading frame: the position of body ``pelvis`` and its world yaw removed, z
    only translated."""
    return rotate_z(bodies - bodies[:, pelvis:pelvis + 1], -yaw(pelvis_quaternion)[:, None])


def aligned_body_distances(robot, robot_anchor_quaternion, reference, reference_anchor_quaternion, anchor):
    """[K, B] distances (m) between the body's bodies and the reference bodies moved under the body's anchor by E's
    alignment: turned about z by the yaw of R_anchor R_reference_anchor^T and translated so that the reference anchor
    lies at the body anchor's planar position with the reference anchor's height. A rigid planar shift and turn of the
    whole body reads 0; a height difference does not."""
    robot, reference = np.asarray(robot, np.float64), np.asarray(reference, np.float64)
    unit = lambda q: np.asarray(q, np.float64)/np.linalg.norm(np.asarray(q, np.float64), axis=-1, keepdims=True)  # noqa: E731
    conjugate = unit(reference_anchor_quaternion)*np.array([1.0, -1.0, -1.0, -1.0])
    turn = yaw(quaternion_product(unit(robot_anchor_quaternion), conjugate))
    reference_anchor = reference[:, anchor]
    origin = np.concatenate((robot[:, anchor, :2], reference_anchor[:, 2:]), -1)
    aligned = origin[:, None] + rotate_z(reference - reference_anchor[:, None], turn[:, None])
    return np.linalg.norm(robot - aligned, axis=-1)


# ------------------------------------------------------------------------------------------------ fall_recovery

def controls_of(seconds, dt, name):
    count = int(round(seconds/dt))
    if count < 1 or abs(count*dt - seconds) > TIME_TOLERANCE_S:
        raise ValueError(f"{name} = {seconds} s does not lie on the {dt} s control clock")
    return count


def masked(values, available):
    values = np.asarray(values, dtype=np.float64).copy()
    values[~available] = np.nan
    return values


def check_rollout(rollout, name):
    """A rollout holds every array of FIELDS, each [N, T, ...] with one N and T, and the bodies with one B."""
    if set(rollout) != set(FIELDS):
        raise ValueError(f"the {name} rollout holds {sorted(rollout)}, expected {sorted(FIELDS)}")
    shape = np.shape(rollout["available"])
    bodies = np.shape(rollout["body_pos_w"])[2:3]
    for key, trailing in FIELDS.items():
        found = np.shape(rollout[key])
        expected = shape + ((*bodies, 4 if key.endswith("quat_w") else 3) if trailing else ())
        if len(shape) != 2 or found != expected:
            raise ValueError(f"the {name} rollout's {key} is {found}, expected {expected}")


def trial_columns(rollout, trial, floor_height, pelvis):
    """One trial of a rollout against the command rows it reached."""
    available = np.asarray(rollout["available"][:, trial], bool)
    row = np.asarray(rollout["command_row_after"][:, trial], np.int64)
    robot = masked(rollout["body_pos_w"][:, trial], available)
    robot_quaternion = masked(rollout["body_quat_w"][:, trial, pelvis], available)
    command = masked(rollout["command_body_pos_w"][:, trial], available)
    command_quaternion = masked(rollout["command_body_quat_w"][:, trial, pelvis], available)
    clip_end = row == np.asarray(rollout["clip_last_row"][:, trial], np.int64)
    return dict(available=available, row=row, row_hold_allowed=clip_end, reference_end=clip_end & available,
                done=np.asarray(rollout["done"][:, trial], bool),
                nonfinite=np.asarray(rollout["nonfinite"][:, trial], bool), pelvis=pelvis,
                robot_bodies=robot, robot_quaternion=robot_quaternion, robot_height=robot[:, pelvis, 2] - floor_height,
                robot_up=up_axis_z(robot_quaternion), command_bodies=command, command_quaternion=command_quaternion,
                command_height=command[:, pelvis, 2] - floor_height, command_up=up_axis_z(command_quaternion))


def support(columns, horizon):
    """The trial's support [0, end) and the reason it ends."""
    end_limit = min(horizon, len(columns["available"]))
    for i in range(end_limit):
        numerical = ["termination_nonfinite"] if columns["nonfinite"][i] else []
        if columns["available"][i] and not valid_quaternion(columns["robot_quaternion"][i]):
            numerical.append("invalid pelvis quaternion")
        if numerical:
            return dict(start=0, end=i, end_reason="numerical_failure", end_control=i, end_terms=numerical)
        if not columns["available"][i]:
            return dict(start=0, end=i, end_reason="post_step_state_unavailable", end_control=i, end_terms=[])
        row, previous = columns["row"][i], columns["row"][i - 1] if i else None
        if i and row != previous + 1 and not (columns["row_hold_allowed"][i] and row == previous):
            return dict(start=0, end=i, end_reason="reference_row_discontinuity", end_control=i,
                        end_terms=[int(previous), int(row)])
        for reason in ("reference_end", "done"):
            if columns[reason][i]:
                return dict(start=0, end=i + 1, end_reason="environment_done" if reason == "done" else reason,
                            end_control=i, end_terms=[])
    return dict(start=0, end=end_limit, end_reason="horizon" if end_limit == horizon else "record_end",
                end_control=None, end_terms=[])


def rules(columns, settings):
    """Per control: the body in the fall rule, a paired command, an upright command, and the get-up condition."""
    fall, recovered, upright = settings["fall"], settings["recovered"], settings["fall"]["command_upright"]
    cos = lambda degrees: math.cos(math.radians(degrees))  # noqa: E731
    with np.errstate(invalid="ignore"):
        height, up = columns["robot_height"], columns["robot_up"]
        in_rule = (height < fall["pelvis_height_below_m"]) | (up < cos(fall["pelvis_tilt_above_deg"]))
        known = columns["available"] & np.isfinite(columns["command_height"])
        command_upright = (known & (columns["command_height"] > upright["pelvis_height_above_m"])
                           & (columns["command_up"] >= cos(upright["pelvis_tilt_within_deg"])))
        up_again = (known & columns["available"] & ~in_rule
                    & (height >= recovered["command_pelvis_height_fraction"]*columns["command_height"])
                    & (up >= cos(recovered["pelvis_tilt_within_deg"])))
    return dict(fall=in_rule & columns["available"], known=known, low_command=known & ~command_upright,
                recovered=up_again)


def hold(start, end, inside, dt, duration):
    """The first control of the first run of inside controls in [start, end) whose elapsed time (last - first) dt
    reaches ``duration``, and the control that confirms it; None without one."""
    beginning = None
    for i in range(start, end):
        if not inside[i]:
            beginning = None
            continue
        if beginning is None:
            beginning = i
        if (i - beginning)*dt + 1e-12 >= duration:
            return beginning, i
    return None


def falls(columns, settings, span, placed, dt):
    """The onsets and get-ups over the support, the listed controls in the fall rule that count as no fall, and
    whether a placed start's control 0 is in the fall rule (None for another scene)."""
    rule = rules(columns, settings)
    start, end = span["start"], span["end"]
    found, listed, eligible, i, listing = [], [], False, start, None
    in_rule = bool(rule["fall"][start]) if placed and start < end else None
    # Every control of the support has an available state and its command, so each one is paired.
    while i < end:
        by_declaration = in_rule is True and i == start
        if not by_declaration:
            if rule["low_command"][i]:
                eligible, listing = False, None
                i += 1
                continue
            if not rule["fall"][i]:
                eligible, listing = True, None
                i += 1
                continue
            if not eligible:
                if listing is None:
                    listing = dict(kind="at_origin" if i == start else "at_command_rise", control=i, controls=0)
                    listed.append(listing)
                listing["controls"] += 1
                i += 1
                continue
        onset = dict(onset_control=i, onset_by_declaration=by_declaration)
        found.append(onset)
        eligible = False
        held = hold(i + 1, end, rule["recovered"], dt, settings["recovered"]["hold_s"])
        if held is None:
            onset["recovered"] = False
            break
        up, confirmed = held
        onset.update(recovered=True, recovered_control=up, recovery_confirmed_control=confirmed,
                     time_to_recover_s=(up - i)*dt)
        eligible, i = not rule["low_command"][confirmed], confirmed + 1
    return found, listed, in_rule


def body_errors(columns):
    """[K] the rejoin's error per control: the mean distance (m) over the tracked bodies between body and command, each
    in its own pelvis heading frame; NaN where a side is not available."""
    rows = (columns["available"] & valid_quaternion(np.nan_to_num(columns["robot_quaternion"]))
            & valid_quaternion(np.nan_to_num(columns["command_quaternion"]))
            & np.isfinite(columns["robot_bodies"]).all((1, 2)) & np.isfinite(columns["command_bodies"]).all((1, 2)))
    out = np.full(len(rows), np.nan)
    if rows.any():
        robot = local_positions(columns["robot_bodies"][rows], columns["robot_quaternion"][rows], columns["pelvis"])
        command = local_positions(columns["command_bodies"][rows], columns["command_quaternion"][rows],
                                  columns["pelvis"])
        out[rows] = np.linalg.norm(robot - command, axis=-1).mean(1)
    return out


def twin_span(columns, span, twin, settings, horizon, dt):
    """The twin's usable controls (both origins at control 0) and its error at each of them."""
    twin_support = support(twin, horizon)
    twin_falls, twin_listed, _ = falls(twin, settings, twin_support, False, dt)
    length = twin_support["end"]
    reason = f"twin support ended: {twin_support['end_reason']} at twin control {twin_support['end_control']}"
    down = [(entry["control"], f"twin in the fall rule under an upright command at twin control {entry['control']} "
                               f"({entry['kind']})") for entry in twin_listed]
    down += [(onset["onset_control"], f"twin fell: onset at twin control {onset['onset_control']}")
             for onset in twin_falls[:1]]
    if down and min(down)[0] < length:
        length, reason = min(down)
    length = min(length, span["end"] - span["start"])
    paired, mismatch = np.zeros(length, bool), None
    for k in range(length):
        if not (columns["available"][k] and twin["available"][k]):
            continue
        if columns["row"][k] != twin["row"][k]:
            mismatch = dict(control=k, trial_row=int(columns["row"][k]), twin_row=int(twin["row"][k]))
            paired = paired[:k]
            reason = f"twin command row differs from trial control {k}"
            break
        paired[k] = True
    error = np.full(len(columns["available"]), np.nan)
    error[np.flatnonzero(paired)] = body_errors(twin)[np.flatnonzero(paired)]
    return dict(usable_end=len(paired), reason=reason, paired_controls=int(paired.sum()), first_row_mismatch=mismatch,
                twin_error=error, twin_support={key: twin_support[key] for key in ("start", "end", "end_reason",
                                                                                     "end_control")})


def rejoin(columns, settings, onset, limit, twin, dt):
    if twin is None:
        return dict(status="unavailable", reason="no twin")
    rejoined = settings["rejoined"]
    n = controls_of(rejoined["rolling_mean_s"], dt, "rejoined.rolling_mean_s")
    first = onset["recovered_control"] + n - 1
    excess = body_errors(columns) - twin["twin_error"]
    usable_end = min(limit, twin["usable_end"])
    rolling = np.full(len(excess), np.nan)
    for i in range(first, usable_end):
        window = excess[i - n + 1:i + 1]
        if np.isfinite(window).all():
            rolling[i] = window.mean()
    with np.errstate(invalid="ignore"):
        inside = rolling <= rejoined["above_twin_m"]
    held = hold(first, usable_end, inside, dt, rejoined["hold_s"])
    if held is not None:
        return dict(status="rejoined", rejoined_control=held[0], rejoin_confirmed_control=held[1],
                    time_to_rejoin_s=(held[0] - onset["onset_control"])*dt, window_lag_controls=n - 1)
    if twin["usable_end"] < limit:
        return dict(status="unavailable", reason=twin["reason"])
    # Not rejoined only where the excess is defined wherever a hold could lie.
    undefined = ~np.isfinite(rolling[first:usable_end])
    if undefined.any() and hold(first, usable_end, inside | ~np.isfinite(rolling), dt, rejoined["hold_s"]) is not None:
        return dict(status="unavailable", reason=f"excess over the twin undefined at {int(undefined.sum())} of "
                                                 f"{undefined.size} controls; a hold could lie there")
    return dict(status="not_rejoined")


def status(span, found, listed, in_rule):
    if span["end_reason"] == "numerical_failure":
        return "numerical_failure"
    if in_rule is False:
        return "placed_start_outside_fall_rule"
    at_horizon = span["end_reason"] == "horizon"
    if not found:
        if listed:
            return "fall_rule_met_not_counted" if at_horizon else "fall_rule_met_not_counted_ended_early"
        return "no_fall" if at_horizon else "no_fall_ended_early"
    if not found[0]["recovered"]:
        return "fell_not_recovered_at_horizon" if at_horizon else "fell_ended_early"
    outcome = found[0]["rejoin"]["status"]
    if outcome == "rejoined":
        return "rejoined"
    if outcome == "unavailable":
        return "recovered_rejoin_unavailable"
    if len(found) > 1:
        return "recovered_fell_again"
    return "recovered_not_rejoined_at_horizon" if at_horizon else "recovered_ended_early"


def read_trial(columns, settings, dt, *, placed, twin=None):
    """The reading of one trial from control 0: a placed start (``placed``) or an undisturbed start, with the columns
    of its twin when given."""
    horizon = controls_of(settings["horizon_s"], dt, "horizon_s")
    span = support(columns, horizon)
    found, listed, in_rule = falls(columns, settings, span, placed, dt)
    usable = None if twin is None else twin_span(columns, span, twin, settings, horizon, dt)
    for index, onset in enumerate(found):
        if onset["recovered"]:
            limit = found[index + 1]["onset_control"] if index + 1 < len(found) else span["end"]
            onset["rejoin"] = rejoin(columns, settings, onset, limit, usable, dt)
    reading = dict(status=status(span, found, listed, in_rule), support=span, falls=found,
                   not_counted_as_falls=listed, placed_start_in_fall_rule=in_rule)
    if usable is not None:
        reading["twin"] = {key: value for key, value in usable.items() if key != "twin_error"}
    return reading


# ------------------------------------------------------------------------------------------------ counts

def count(numerator, denominator):
    """A count over every trial of its denominator."""
    return dict(numerator=numerator, denominator=denominator, share=numerator/denominator)


def rejoin_unavailable_cause(reading, twin_reading):
    """Why a recovered trial's rejoin could not be read, from its twin's reading at the end of its usable span."""
    twin = reading["twin"]
    end = twin["usable_end"]
    if twin_reading["falls"] and twin_reading["falls"][0]["onset_control"] == end:
        return "twin_fell"
    if any(entry["control"] == end for entry in twin_reading["not_counted_as_falls"]):
        return "twin_in_the_fall_rule_where_no_onset_counts"
    if twin["first_row_mismatch"]:
        return "twin_command_row_differs"
    ended = twin["twin_support"]
    if ended["end"] == end and ended["end_reason"] != "horizon":
        return f"twin_support_ended: {ended['end_reason']}"
    return f"other: {reading['falls'][0]['rejoin'].get('reason')}"


def placed_quantities(readings, twin_readings):
    """R (placed starts that get up), J (placed starts that rejoin), J over R and F (twins with a counted fall);
    ``readings`` and ``twin_readings`` hold every placed start and its twin by trial."""
    if set(readings) != set(twin_readings):
        raise ValueError("the placed starts and the twins are not the same trials")
    total = len(readings)
    states = {trial: reading["status"] for trial, reading in readings.items()}
    recovered = [trial for trial, value in states.items() if value in RECOVERED]
    rejoined = [trial for trial, value in states.items() if value == "rejoined"]
    unavailable = [trial for trial, value in states.items() if value == "recovered_rejoin_unavailable"]
    causes = {}
    for trial in unavailable:
        causes.setdefault(rejoin_unavailable_cause(readings[trial], twin_readings[trial]), []).append(trial)
    readable_rejoins = len(recovered) - len(unavailable)
    numerical = [trial for trial, value in states.items() if value == "numerical_failure"]
    twin_falls = sorted(trial for trial, reading in twin_readings.items() if reading["falls"])
    return dict(
        R=dict(**count(len(recovered), total),
               rule="trials whose status is one of the recovered statuses, over every placed start",
               numerical_failure=numerical,
               got_up_before_numerical_failure=[t for t in numerical if readings[t]["falls"]
                                                and readings[t]["falls"][0]["recovered"]],
               placed_start_outside_fall_rule=[t for t, v in states.items() if v == "placed_start_outside_fall_rule"]),
        J=dict(**count(len(rejoined), total), rule="rejoined trials over every placed start",
               recovered_rejoin_unavailable=len(unavailable), recovered_rejoin_unavailable_trials=unavailable,
               recovered_rejoin_unavailable_by_cause=causes),
        J_over_R=dict(rejoined=len(rejoined), recovered=len(recovered), denominator=readable_rejoins,
                      value=len(rejoined)/readable_rejoins if readable_rejoins > 0 else None,
                      recovered_rejoin_unavailable=len(unavailable),
                      rule="rejoined over the recovered trials whose rejoin was readable; a recovered trial whose "
                           "rejoin is unavailable counts neither as rejoined nor as not rejoined; unavailable when no "
                           "recovered trial's rejoin was readable",
                      over_every_recovered_trial=dict(
                          value=len(rejoined)/len(recovered) if recovered else None,
                          denominator=len(recovered), declared=False)),
        F=dict(**count(len(twin_falls), total),
               rule="twins with a counted fall onset, over every twin", twins_with_a_fall=twin_falls,
               twins_in_the_fall_rule_not_counted=sorted(t for t, r in twin_readings.items()
                                                         if not r["falls"] and r["not_counted_as_falls"]),
               twins_with_numerical_failure=sorted(t for t, r in twin_readings.items()
                                                   if r["status"] == "numerical_failure"),
               rejoin_readings_removed_by_a_twin_fall=len(causes.get("twin_fell", []))),
        statuses={name: sum(value == name for value in states.values()) for name in STATUSES},
        twin_statuses={name: sum(r["status"] == name for r in twin_readings.values()) for name in STATUSES},
        time_to_recover_s=[[t, readings[t]["falls"][0]["time_to_recover_s"]] for t in recovered],
        time_to_rejoin_s=[[t, readings[t]["falls"][0]["rejoin"]["time_to_rejoin_s"]] for t in rejoined])


def first_row_reading(rollout, clip_names, clip_first_rows, hidden_clips, anchor):
    """C, C_getup and E of the rollout from each clip's first row, trial i on the clip ``clip_names[i]`` from its first
    row ``clip_first_rows[i]``. A clip is read over its rows minus one; a trial that does not start at its clip's first
    row, and a clip whose available controls are not a prefix of its rollout, are refused."""
    if not len(clip_names) == len(clip_first_rows) == rollout["available"].shape[1]:
        raise ValueError(f"the first-row rollout holds {rollout['available'].shape[1]} trials, not one per clip of "
                         f"{len(clip_names)} clips with {len(clip_first_rows)} first rows")
    starts = np.asarray(rollout["command_row_before"][0], np.int64).tolist()
    if starts != [int(row) for row in clip_first_rows]:
        raise ValueError("the first-row rollout does not start each trial at its clip's first row")
    clips = []
    for trial, name in enumerate(clip_names):
        # Trial i starts at its clip's first row, the row before control 0.
        controls = int(rollout["clip_last_row"][0, trial]) - int(rollout["command_row_before"][0, trial])
        if rollout["available"].shape[0] < controls:
            raise ValueError(f"the first-row record holds {rollout['available'].shape[0]} controls, clip {name} "
                             f"needs {controls}")
        available = np.asarray(rollout["available"][:controls, trial], bool)
        if np.logical_and(~available[:-1], available[1:]).any():
            raise ValueError(f"the available controls of clip {name} are not a prefix of its record")
        valid = int(available.sum())
        hits = np.flatnonzero(rollout["done"][:valid, trial])
        k = int(hits[0]) if hits.size else valid
        rows = np.asarray(rollout["command_row_after"][:valid, trial], np.int64)
        advancing = bool((np.diff(rows) == 1).all())
        item = dict(clip=name, controls=controls, support_controls=k, first_termination=int(hits[0]) if hits.size
                    else None, available_controls=valid, rows_advance_by_one=advancing,
                    completed=k == controls and advancing, aligned_body_error_m=None, global_body_error_m=None,
                    capacity_exceeded_controls_in_support=int((
                        np.asarray(rollout["contacts_beyond_capacity"][:k, trial], bool)
                        | np.asarray(rollout["rows_beyond_capacity"][:k, trial], bool)).sum()))
        if k:
            robot = np.asarray(rollout["body_pos_w"][:k, trial], np.float64)
            robot_quaternion = np.asarray(rollout["body_quat_w"][:k, trial], np.float64)
            reference = np.asarray(rollout["command_body_pos_w"][:k, trial], np.float64)
            reference_quaternion = np.asarray(rollout["command_body_quat_w"][:k, trial], np.float64)
            item["aligned_body_error_m"] = float(aligned_body_distances(
                robot, robot_quaternion[:, anchor], reference, reference_quaternion[:, anchor], anchor).mean(1).mean())
            item["global_body_error_m"] = float(np.linalg.norm(robot - reference, axis=-1).mean(1).mean())
        clips.append(item)
    errors = [clip["aligned_body_error_m"] for clip in clips]
    by_name = {clip["clip"]: clip for clip in clips}
    return dict(C=dict(completed=sum(clip["completed"] for clip in clips), of=len(clips)),
                C_getup={name: by_name[name]["completed"] for name in hidden_clips},
                E_m=dict(value=float(np.mean(errors)) if None not in errors else None,
                         clips_without_error=[clip["clip"] for clip in clips if clip["aligned_body_error_m"] is None],
                         quantity="mean over the clips of each clip's mean aligned body-position error over its "
                                  "support, in metres"),
                global_body_error_m=(float(np.mean([clip["global_body_error_m"] for clip in clips]))
                                     if None not in errors else None),
                clips=clips)


def engine_report(rollout):
    available = np.asarray(rollout["available"], bool)
    exceeded = (np.asarray(rollout["contacts_beyond_capacity"], bool)
                | np.asarray(rollout["rows_beyond_capacity"], bool)) & available
    return dict(largest_contacts=int(np.max(rollout["contacts"][available], initial=0)),
                largest_constraint_rows=int(np.max(rollout["constraint_rows"][available], initial=0)),
                trials_with_capacity_exceeded=int(exceeded.any(0).sum()))


def save_readings(placed, twins, first_rows, settings, dt, pelvis, anchor, clip_names, clip_first_rows, hidden_clips):
    """The readings of one save from its three rollouts: ``placed`` (the placed starts), ``twins`` (trial i the twin of
    placed start i) and ``first_rows`` (trial i at the first row ``clip_first_rows[i]`` of the clip ``clip_names[i]``).
    ``hidden_clips`` names the clips whose completion C_getup gives. Returns the save's quantities (those of
    placed_quantities), the reading of every placed start and twin by trial, each rollout's engine report and the
    first-row reading."""
    for name, rollout in (("placed", placed), ("twins", twins), ("first_rows", first_rows)):
        check_rollout(rollout, name)
    trials = placed["available"].shape[1]
    if twins["available"].shape[1] != trials:
        raise ValueError(f"{trials} placed starts and {twins['available'].shape[1]} twins")
    floor = settings["floor_height_m"]
    readings, twin_readings = {}, {}
    for trial in range(trials):
        twin = trial_columns(twins, trial, floor, pelvis)
        twin_readings[trial] = read_trial(twin, settings, dt, placed=False)
        readings[trial] = read_trial(trial_columns(placed, trial, floor, pelvis), settings, dt, placed=True,
                                     twin=twin)
    return dict(quantities=placed_quantities(readings, twin_readings), readings=readings, twin_readings=twin_readings,
                engine=dict(placed=engine_report(placed), twins=engine_report(twins),
                            first_rows=engine_report(first_rows)),
                first_rows=first_row_reading(first_rows, clip_names, clip_first_rows, hidden_clips, anchor))
