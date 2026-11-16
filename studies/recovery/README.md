# Recovery from placed lying starts: a pair of trainings

**Question.** Does recovery training from placed lying starts against a hidden get-up reference
([RECOVERY.md](../../RECOVERY.md)) give a PyTorch-trainer policy that gets up from placed lying starts and rejoins its
command? Does that policy keep tracking the 40 LAFAN1 clips, against the same recipe without recovery? The recipe is
the tracking recipe of `configs/base/ours.args` ([RECOVERY.md](../../RECOVERY.md)); `configs/absolute.args` runs it on
NablaEngine with absolute actions.

The study trains a control and a recovery policy at two seeds with `nablamimic.torch_train`, rolls out each save with
`nablamimic.engine_rollouts` and reads the rollouts. The declaration is [pair.json](pair.json), fixed before the
trainings. Every program reads it through `declaration.py`, which refuses a declaration with other top-level keys. It
is JSON because the trainer's environment has no YAML reader.

Values marked *StableMimic* are those of StableMimic (arXiv 2608.02385v1, Section IV-A). Every rollout also follows
that section's evaluation in deterministic actions at 50 Hz with randomisation and observation noise off, and the placed
starts and twins follow it in 100 trials that are the same for every policy. The recovery options are those of
`configs/recovery.args`, which gives the source of each. The alignment in the body-position error E is BeyondMimic's
(Liao et al. 2025, arXiv:2508.08241v4, Supplementary Text S1, "Anchor"). The rest is this study's choice: the scene's
rows and draws, the fall rule's use for the command, the get-up and rejoin rules, the twin, C and E, the margins and the
outcome order.

## Declaration

### Trainings and saves

| | |
| --- | --- |
| Conditions | control (`configs/absolute.args`) and recovery (`configs/recovery.args`: the control plus the seven recovery options) |
| Seeds | 0 and 1. Each training initialises its networks from its seed, so the two conditions of a seed start from equal weights |
| Length | 3,000 updates with a numbered checkpoint every 250 updates, and no wall-time limit (`--hours` unset) |
| Execution | the trainer's defaults: a compiled window with CUDA graphs (`--execution compile-graph`) and TF32 network matrix products (`--matmul-precision nets_tf32`) |
| Card | chosen at launch; the two trainings of a seed share the card model, source revision, image and engine, as `config.json` records them |
| Saves | `actor_ema` of the numbered checkpoint `ckpt/it_NNNNNN.pkl` at 1,500 and 3,000 updates, rolled out with deterministic actions. A training gives two saves and the pair eight |

Each save is read on its own; nothing is pooled across saves or seeds. The unit of repetition is the seed, two per
condition. The two saves of one training are not repetitions.

### Scene

`scene.py` writes the scene files of `engine_rollouts` on the 40-clip library of the trainings (`--clips all`).

- **Placed lying starts.** 100 trials, the same at every save.
  - *Body rows* are evenly spaced over the start rows of `configs/recovery.args`.
  - *Command rows* are evenly spaced over the rows outside its command-excluded clips whose next 1,000 rows are upright
    and lie before their clip's last row. A 20 s rollout therefore meets neither a clip end nor a low command.
  - *Evenly spaced* means the rounded points of a float64 linspace over the row indices.
  - *Placement.* The body is placed as a recovery start of training is placed, from unit uniforms in [-1, 1). One torch
    CPU generator with seed 7 draws them in this order: position [100, 3], rotation [100, 3], velocity [100, 6] and
    joints [100, 29]. The same generator then draws a permutation that pairs the command rows with the body rows, so a
    get-up recording and a command clip are not paired by library order.
- **Twins.** 100 trials at the same command rows, each started at its command row alone.
- **First rows.** One trial at the first row of each of the 40 clips.

Every trial starts through the training reset, which holds the joints within the soft joint range. Every rollout runs on
NablaEngine with domain randomisation and observation noise off ([ROLLOUTS.md](../../ROLLOUTS.md), "Environment and
policy"). Placed starts and twins run 1,000 controls (20 s) under `--termination nonfinite`, so only a non-finite state
ends a trial. First rows run under `--termination recipe` for the longest clip's rows minus one, which `scene.py`
prints. Each clip is read over its own rows minus one.

### Readings

`readings.py` reads one save from its three rollouts and the library. It builds plain arrays from the records and
calls `readings_core.py`, which holds every rule below on those arrays with numpy alone; rollouts of any trainer can be
read by building these arrays from its records. The state after control i is paired with the command's bodies at the
row the command reached at that control, the row the recipe's reward and termination compare it with. The tracked
bodies are the 14 of the record; the pelvis carries the fall rule and the heading frame, and the torso is the anchor of
the body-position error E.

- **Support.** A trial is read from control 0 for at most 20 s (*StableMimic*'s horizon). It ends earlier at a
  numerical failure (a non-finite state or an invalid pelvis orientation), at a control without an available state, or
  at a command row that does not advance by one (a row held at its clip's last row advances); each of these controls
  lies outside the support. It also ends at the command's arrival at its clip's last row and at the trial's end; these
  controls lie inside.
- **Fall rule** (*StableMimic*): the pelvis is below 0.50 m or tilted beyond 60 degrees.
  - A command is upright when its pelvis is above 0.50 m and within 60 degrees of upright.
  - A placed start falls at control 0 when the state after control 0 meets the fall rule. Otherwise the trial is
    `placed_start_outside_fall_rule` and counts as not got up.
  - A later fall counts at a control with an upright command where the body meets the fall rule, and only after a
    control with an upright command outside the fall rule or after a confirmed get-up. A body in the fall rule
    elsewhere is listed, not counted.
- **Get-up.** The body gets up at the first control after a fall that starts a hold of 0.5 s (26 consecutive controls)
  outside the fall rule, with its pelvis at least 0.8 of the command's pelvis height and within 30 degrees of upright.
- **Rejoin.**
  - The error of a control is the per-control form of *StableMimic*'s body-position error: the mean distance over the
    tracked bodies between body and command, each side relative to its own pelvis and turned into its own pelvis yaw
    frame. The per-control mean, the unit (metres) and the 14 tracked bodies are this study's choice.
  - The excess of this error over the twin's at the same control is averaged over a trailing 1.0 s (50 controls), and
    only over controls at or after the get-up: the first average is at the get-up control plus 49, so the earliest
    rejoin is 0.98 s after the get-up. An average is read only where the excess is defined at all 50 controls.
  - A trial rejoins at the first control of the first run of averages within 0.05 m that lasts 2.0 s (101 consecutive
    controls), before any later fall and while the twin is usable. Its time to rejoin, like its time to get up, counts
    from the fall's onset.
  - The twin is usable up to the first of these controls: its first control in the fall rule under an upright command,
    the end of its support, and the first control whose command row differs from the trial's.
  - The rejoin is unavailable when the twin stops being usable before a rejoin could be read, or when the excess is
    undefined where a hold could lie.

| Quantity of a save | Definition |
| --- | --- |
| R | placed starts that get up, over all 100 (*StableMimic* keeps every trial in the denominator). Trials outside the fall rule at placement and trials with a numerical failure count as not got up |
| J | placed starts that rejoin, over all 100 |
| J over R | rejoined trials over the trials that got up and whose rejoin was readable; unavailable when no such trial exists. The count of unavailable rejoins and rejoined over every trial that got up stand beside it |
| F | twins with a counted fall, over all 100 |
| C | clips of the 40 completed: no termination ends the clip within its rows minus one, and its command advances by one row per control |
| C_getup | whether each of the five hidden get-up clips is completed |
| E | body-position error in metres. The command's bodies are moved under the body's torso by BeyondMimic's alignment: turned about z by the yaw between the two torsos and shifted so that the command's torso lies at the body torso's planar position, at the command torso's height. The alignment is applied to all 14 bodies, the torso included. E is the distance per body averaged over the bodies, over each clip's controls before its first termination, then over the 40 clips |

Beside these, each save gives every trial's time to get up and to rejoin; every trial's pelvis yaw between body and
command at placement, with its status (the yaw follows the trainer's reset, is wrapped into [-pi, pi) and enters no
reading); the largest contact and constraint-row counts and the trials whose engine capacity was exceeded; and the
world-frame body error of the first rows.

**Exposure.** `exposure.py` gives the tracking window rows of each training at 1,500 and 3,000 updates from
`log.jsonl` ([RECOVERY.md](../../RECOVERY.md), "Records"): `recovery_tracking_window_rows_total` for a recovery
training, the updates times `num_envs` times `horizon` for a control training. Beside them it gives the yaw at the
switches back to the command up to each save (weighted by the switches) and, for each recovery save, the first control
update that reaches its tracking rows. No outcome row reads the matched update.

**First log rows.** The first 100 rows of each recovery training's `log.jsonl` are read by hand, with no automatic stop,
for `recovery_window_row_share_recovery` (far above 0.5 means recovery episodes outlast tracking episodes),
`recovery_stops`, `recovery_switches`, `engine_diagnostics`, and `recovery_tracking_reward_mean` against the control's
`reward_mean` at the same update.

### Classes, margins and outcomes

Each seed's recovery save at 3,000 updates is classed as:

- *near_zero* when R ≤ 5;
- *gets_up* when R ≥ 50 and R ≥ the larger control R at 3,000 plus 20;
- *some* otherwise.

Tracking is *within* against a control save of the same seed when the recovery save at 3,000 meets all three of:

- C ≥ the control's C − max(2, |C0 − C1|);
- E ≤ the control's E + max(0.005 m, |E0 − E1|);
- F ≤ the control's F + 2.

Here C0, C1, E0 and E1 are C and E of the control seeds 0 and 1 at that control save. The other margins are
`rejoins_J_over_R` 0.5, `still_rising` max(|R0 − R1| of the recovery seeds at 3,000, 5) and `control_within` 10.

The outcome is the first of these rows that holds; `pair.json` gives each row's meaning and what it sends forward.

1. **run_stopped**: a training ends before 3,000 updates by an engine capacity overflow. This row
   is decided from the trainings' ends before any save is read.
2. **split_seeds**: one seed is gets_up and the other near_zero.
3. **control_within**: at each seed the control's R at 3,000 is above 5 and within 10 of the recovery R.
4. **precondition_unmet**: no seed is gets_up, and the control completes none of the five hidden get-up clips at 1,500
   or at 3,000 at either seed.
5. **still_rising**: a seed that is not gets_up has R at 3,000 minus R at 1,500 above still_rising.
6. **no_get_up**: both seeds are near_zero.
7. **some_get_up**: no seed is gets_up.
8. **gets_up_without_rejoining**: both seeds are gets_up, and J over R is below 0.5 at either seed.
9. **gets_up_and_tracks**: both seeds are gets_up, J over R is at least 0.5 at both, and tracking is within against the
   control at 3,000 at both seeds.
10. **tracking_cost_at_exposure**: as row 9, but at both seeds tracking is outside against the control at 3,000 and
    within against the control at 1,500.
11. **tracking_worse**: as row 9, but at both seeds tracking is outside against both control saves.
12. **no_call**: any other case.

An unavailable J over R is neither below nor at its margin, so a row that needs its value at that seed does not hold.

A training ends by its 3,000 updates, by a capacity overflow (a `FAILED.json` whose invalid counters include a contact
or constraint-row overflow, whatever other counters it names), or otherwise. The trainer ends a run
on any non-zero non-finite counter; such a run ends otherwise.

**Unreadable.** No outcome is given, and the reasons are written instead, when any of these holds:

- a declared save has no reading, or its E is unavailable because a clip has no body-position error;
- a training's `config.json` names another seed, other updates or another checkpoint interval than declared, a
  wall-time limit, or recovery options that do not fit its condition;
- the two trainings of a seed have `config.json` files that differ outside the recovery options, `out`, `config` and
  the location of the compiled engine library (`engine.path`, `engine.artifact`). This covers the card, the source
  revision, the image, the engine's identity (`engine.key`, `engine.sources_sha256`, `engine.consts_sha256`) and
  capacities, the torch version and the library inputs;
- a training ended otherwise.

`readings.py` refuses a rollout record that is incomplete; that belongs to another scene, termination, control count,
weights or checkpoint; whose command or body rows are not its scene file's; or that uses another library. That save
then has no reading.

## Commands, in order

Run from the repository root of one committed revision, in the provisioned trainer environment. `RUNS` and `STUDY` are
output roots that do not exist yet, and every program writes to an output directory that must not exist. The library
variables `LAFAN1_G1_DIR`, `MOTION_CACHE_DIR` and `MENAGERIE_DIR` stay the same for every step; `scene.py`,
`engine_rollouts` and `readings.py` compare them with each other.

1. **The four trainings**, one card each. The rollouts read the numbered checkpoints, so no export is needed.

   ```bash
   for seed in 0 1; do
     python -m nablamimic.torch_train --config configs/absolute.args --seed $seed --iters 3000 --ckpt-every 250 \
         --out $RUNS/control_s$seed
     python -m nablamimic.torch_train --config configs/recovery.args --seed $seed --iters 3000 --ckpt-every 250 \
         --out $RUNS/recovery_s$seed
   done
   ```

2. **The scene.** The program prints `first_row_controls`, the control count of the first-row rollouts.

   ```bash
   python studies/recovery/scene.py --pair studies/recovery/pair.json --out $STUDY/scene
   ```

3. **The rollouts**, three per save, one card each.

   ```bash
   FIRST_ROW_CONTROLS=$(python -c "import json; print(json.load(open('$STUDY/scene/scene.json'))['first_row_controls'])")
   for run in control_s0 recovery_s0 control_s1 recovery_s1; do
     for iter in 1500 3000; do
       python -m nablamimic.engine_rollouts --run $RUNS/$run --iter $iter --scene $STUDY/scene/placed.json \
           --termination nonfinite --controls 1000 --out $STUDY/rollouts/$run-$iter-placed
       python -m nablamimic.engine_rollouts --run $RUNS/$run --iter $iter --scene $STUDY/scene/twins.json \
           --termination nonfinite --controls 1000 --out $STUDY/rollouts/$run-$iter-twins
       python -m nablamimic.engine_rollouts --run $RUNS/$run --iter $iter --scene $STUDY/scene/first_rows.json \
           --termination recipe --controls $FIRST_ROW_CONTROLS --out $STUDY/rollouts/$run-$iter-first-rows
     done
   done
   ```

4. **The readings**, on the CPU.

   ```bash
   for save in {control,recovery}_s{0,1}-{1500,3000}; do
     python studies/recovery/readings.py --pair studies/recovery/pair.json --scene $STUDY/scene \
         --placed $STUDY/rollouts/$save-placed --twins $STUDY/rollouts/$save-twins \
         --first-rows $STUDY/rollouts/$save-first-rows --out $STUDY/readings/$save
   done
   ```

5. **The exposure.**

   ```bash
   TRAININGS="--training control:0=$RUNS/control_s0 --training recovery:0=$RUNS/recovery_s0
              --training control:1=$RUNS/control_s1 --training recovery:1=$RUNS/recovery_s1"
   python studies/recovery/exposure.py --pair studies/recovery/pair.json $TRAININGS --out $STUDY/exposure
   ```

6. **The outcome.**

   ```bash
   python studies/recovery/outcome.py --pair studies/recovery/pair.json $TRAININGS \
       $(for save in {control,recovery}_s{0,1}-{1500,3000}; do echo --save $STUDY/readings/$save; done) \
       --out $STUDY/outcome
   ```

The tests, `tests/test_recovery_scene.py`, `tests/test_recovery_readings.py` and `tests/test_recovery_outcome.py`, run
on the CPU on generated libraries, records and logs.

## What the study does not show

- **Small differences.** It cannot carry small differences, or any statement beyond its two seeds.
- **A validation.** It is not a validation of getting up. The placed starts are drawn from the recovery training's own
  start rows, and the five hidden get-up clips are tracking commands in both conditions.
- **Other interference.** It cannot separate interference beyond the lower tracking exposure of recovery training and
  the step budget that tracking and recovery rows share.
- **Pushes, contact and speed.** It covers no pushes or falls during execution, and no contact, since the readings are
  kinematic. It makes no speed comparison.
- **Heading and position.** Rejoining is a posture reading against the twin, not a reading of heading or world
  position.

## Results

**Conditions.** The trainings ran at revision 9245b4e0a477, whose code is that of the head of the branch
`train-from-placed-lying-starts`; the scene, rollouts, readings, exposure and outcome ran at revision 7604830cb368.
Every training and rollout ran on one NVIDIA GeForce RTX 4090 with 48 GB under a 350 W power limit, on NablaEngine with
96 contacts and 416 constraint rows per environment. The largest counts in a rollout of the eight saves are 35 contacts
and 140 constraint rows, and no trial exceeded the capacity. Every training completed its 3,000 updates; no training,
rollout or reading failed or was skipped.

| Save | Get up of 100 (R) | Rejoin of 100 (J) | Rejoined over got up (readable) | Twins that fall (F) | Clips completed of 40 (C) | Body-position error E (m) |
| --- | --- | --- | --- | --- | --- | --- |
| control, seed 0, 1,500 | 23 | 22 | 22 / 23 | 0 | 39 | 0.0344 |
| control, seed 0, 3,000 | 31 | 30 | 30 / 31 | 0 | 40 | 0.0286 |
| control, seed 1, 1,500 | 23 | 23 | 23 / 23 | 0 | 37 | 0.0363 |
| control, seed 1, 3,000 | 32 | 31 | 31 / 32 | 0 | 40 | 0.0291 |
| recovery, seed 0, 1,500 | 99 | 98 | 98 / 98 | 1 | 38 | 0.0541 |
| recovery, seed 0, 3,000 | 99 | 98 | 98 / 99 | 0 | 39 | 0.0387 |
| recovery, seed 1, 1,500 | 99 | 98 | 98 / 98 | 2 | 38 | 0.0522 |
| recovery, seed 1, 3,000 | 99 | 99 | 99 / 99 | 0 | 40 | 0.0382 |

Placed start 99 is outside the fall rule at placement at every save and counts as not got up. At each recovery save of
1,500 updates one rejoin is unavailable because its twin fell. Differences below are taken before rounding.

**Outcome: `tracking_cost_at_exposure`** (row 10; rows 1 to 9 do not hold). Both seeds are gets_up (R 99 against 50
and 52), and J over R is 98 / 99 and 99 / 99. Against both control saves of its seed, the recovery save at 3,000 is
within on C and F, so E decides:

- seed 0: E is 0.0101 m above the control at 3,000, outside the 0.005 m margin by 0.0051 m, and 0.0043 m above the
  control at 1,500, within by 0.0007 m;
- seed 1: 0.0091 m above the control at 3,000, outside by 0.0041 m, and 0.0020 m above the control at 1,500, within by
  0.0030 m.

The control at 1,500 updates, the save the recovery policy is within against, has fewer tracking window rows than the
recovery save: 393,216,000, against 525,203,062 (seed 0) and 522,291,905 (seed 1) at the recovery save of 3,000, which
the control reaches at updates 2,004 and 1,993. The control at 3,000 has 786,432,000. The row therefore does not
show a comparison at equal tracking exposure.

**Saves at every 250 updates (exploratory).** These saves were read after the eight declared saves, by a rule written
down before they were read, with the declared scene and readings. The control at 2,000 updates completes 40 clips at
0.0319 m (seed 0) and 39 at 0.0321 m (seed 1); the recovery save at 3,000 is 0.0068 m and 0.0061 m above it. Two
criteria were fixed from the declared saves' readings: a control tracks at the first save with C ≥ 38 and E ≤ 0.0341 m
(the larger control E at 3,000 plus 0.005 m); a recovery policy gets up at the first save with R ≥ 50, R ≥ 52 (the
larger control R at 3,000 plus 20) and J over R ≥ 0.5. The clock runs from the trainer's launch to the save's
checkpoint file.

| Training | Criterion first met | R / C / E (m) at 250 | at 500 | at 1,000 | at 2,000 | at 3,000 |
| --- | --- | --- | --- | --- | --- | --- |
| control, seed 0 | 1,750 updates, 1,107 s | 9 / 21 / 0.0808 | 17 / 32 / 0.0547 | 25 / 37 / 0.0414 | 21 / 40 / 0.0319 | 31 / 40 / 0.0286 |
| control, seed 1 | 1,750 updates, 1,105 s | 10 / 20 / 0.0720 | 11 / 31 / 0.0561 | 17 / 38 / 0.0401 | 24 / 39 / 0.0321 | 32 / 40 / 0.0291 |
| recovery, seed 0 | 500 updates, 442 s | 7 / 10 / 0.1321 | 97 / 25 / 0.1093 | 99 / 35 / 0.0729 | 99 / 39 / 0.0464 | 99 / 39 / 0.0387 |
| recovery, seed 1 | 500 updates, 443 s | 4 / 1 / 0.1359 | 97 / 25 / 0.1060 | 99 / 29 / 0.0726 | 99 / 40 / 0.0445 | 99 / 40 / 0.0382 |

At 1,750 updates the controls read C 39 and 38 at E 0.0331 m and 0.0332 m; at 500 updates the recovery policies read
J over R 92 / 93 and 94 / 96. The checkpoints at 3,000 updates lie at 1,784, 1,781, 1,804 and 1,806 s on this clock.

**Training time** (`terminal.json`; `total_s` holds the set-up and training seconds):

| Training | Total (s) | Set-up (s) | Training (s) |
| --- | --- | --- | --- |
| control, seed 0 | 1,763.3 | 135.5 | 1,627.2 |
| control, seed 1 | 1,760.5 | 135.4 | 1,624.5 |
| recovery, seed 0 | 1,783.5 | 149.5 | 1,633.5 |
| recovery, seed 1 | 1,785.3 | 150.2 | 1,634.6 |

The four trainings together take 7,092.6 s.

**What the results do not show.**

- Each number is one save of one seed; with two seeds no interval is given and nothing is pooled.
- The placed starts are drawn from the rows recovery training starts from, so R and J are not read on unseen starts.
- Every training and rollout ran on one card model; the times hold for that card.
- No row compares E across conditions at equal tracking exposure. The control at 2,000 updates is the numbered save
  nearest the matched updates, and it was read after the declared saves.
- No rollout applies pushes.
