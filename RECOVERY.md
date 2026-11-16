# Recovery training from placed lying starts

The [PyTorch trainer](TORCH.md) can train a tracking policy that also gets up
from lying on the floor and returns to its command. This is recovery training
in a reduced, single-network form of StableMimic (arXiv 2608.02385v1). A share
of the episodes starts with the body placed at a low row of a get-up recording.
During such an episode the reward follows that recording, called the hidden
reference, while the actor reads only the command, which starts at a separate
upright row and advances as in any episode. Once the body is up, a blend leads
the reward back to the command.

```bash
python -m nablamimic.torch_train --config configs/recovery.args --out runs/recovery_s0 --iters 3000
python -m nablamimic.torch_train --config configs/absolute.args --out runs/absolute_s0 --iters 3000
```

Training time: on one RTX 4090 with 48 GB of memory, with the default
`--execution compile-graph` and `--matmul-precision nets_tf32`, either command
trains its 3,000 updates of 4,096 environments in about 27 minutes after 2 to
2.5 minutes of setup; allocated GPU memory peaks at about 32 GiB.

"The recipe" is the tracking recipe of `configs/base/ours.args` (the
NablaMimic 0.2.0 defaults). `configs/absolute.args` is the recipe on
NablaEngine with absolute actions (joint setpoints are the default pose plus
the scaled action, so no reference row enters the action); it is the control
condition. `configs/recovery.args` adds the seven recovery options and nothing
else. The options go together: a partial set is refused, and so is
`--residual-action` with them. Without them the trainer runs the recipe
unchanged: no extra random draw, the same state keys, and the same reset,
step, reward, termination, critic and log.

## Settings

A control is one 0.02 s policy step (50 Hz); a row is one 0.02 s frame of the
motion library. The terms are defined under [Rows](#rows-of-the-motion-library)
and [Order inside a control](#order-inside-a-control).

| Option | Value | Meaning | Source |
| --- | --- | --- | --- |
| `--recovery-share` | 0.5 | probability that a reset is a recovery start | StableMimic, Sections III-B and III-D |
| `--recovery-hidden-clips` | `fallAndGetUp1_subject1`, `fallAndGetUp1_subject5`, `fallAndGetUp2_subject2`, `fallAndGetUp2_subject3`, `fallAndGetUp3_subject1` | clips that give the start rows and the hidden reference | this repository's choice; StableMimic builds its get-up library from the retargeted LAFAN1 get-up subset (Section IV-A) and names no clips; here `fallAndGetUp1_subject4`, a get-up recording of the library, is not a hidden clip |
| `--recovery-command-excluded-clips` | the five hidden clips and `fallAndGetUp1_subject4` | clips that never give a command row | this repository's choice; in StableMimic commands come only from the LAFAN1 dance subset, disjoint from the get-up library (Section IV-A) |
| `--recovery-low-pelvis HEIGHT UPZ` | 0.50 m, 0.5 | bounds of a low row | this repository's choice; StableMimic starts recovery from perturbed prone, supine and intermediate reference frames (Section III-B) with no height or tilt rule, and uses 0.50 m and 60 degrees as the onset of a fall in its evaluation (Section IV-A) |
| `--recovery-rows UPRIGHT_RUN LONGEST_SPAN COMMAND_UPRIGHT` | 50 (1 s), 375 (7.5 s), 500 | upright run that ends a get-up; longest span from a start row to its get-up end; upright rows ahead of a command row | this repository's choice; StableMimic starts the transition back to the command on similarity to the end of the get-up reference and gives no threshold (Section III-B) |
| `--recovery-reward-scale` | 2.5 | factor on the six tracking terms during recovery | StableMimic, Section III-C |
| `--recovery-controls STOP BLEND` | 100 (2.0 s), 75 (1.5 s) | consecutive controls of target error that stop a recovery episode; controls of the blend | StableMimic, Section III-B, for the durations; counting consecutive controls is this repository's choice, as the paper does not say how persistence is counted |

LONGEST_SPAN leaves room for the blend and 1 s of command tracking inside the
10 s (500-control) episode. COMMAND_UPRIGHT equals the episode limit, so no
clip end and no low command row meets a recovery episode.

## Rows of the motion library

`src/nablamimic/recovery.py` classifies the rows once when the environment is
built.

- A row is **low** where its reference pelvis is below HEIGHT or the world z
  component of its pelvis up axis is below UPZ (a tilt beyond 60 degrees).
  Every other row is upright.
- The **get-up end** of row `r` is the last row of the first run of
  UPRIGHT_RUN consecutive upright rows that ends at or after `r` inside its
  clip (none where the clip holds no such run).
- A **start row** is a low row of a hidden clip whose get-up ends inside its
  clip at most LONGEST_SPAN rows later.
- A **command row** lies outside the command-excluded clips, and it and the
  next COMMAND_UPRIGHT rows are upright and before its clip's last row.

Unknown clip names, an empty start or command table, and a hidden clip that is
not command-excluded are refused: a reset is a recovery start where its body
row differs from its command row, which needs the two tables apart.

## Starts

The recipe trains the actor on 64-control windows that start from states of
a replay, which a data collector fills by running the policy.
`ResetSampler.sample` makes every reset draw: the collector's initial states
at the start of training, the collector's resets and the resets inside a
window. It draws, in this order and for every environment: the reset rows
from the reset bins (1 s bins of 50 library rows, drawn half uniformly and
half by weights that favour bins of low return); the reset noise; one uniform,
below the share for a recovery start; a start row, uniform over the start
rows; a command row, uniform over the command rows. Without the options only
the first two draws are made. Recovery starts among the collector's initial
states get random episode ages, as every initial state does.

`env.reset(frame, noise, body_frame)` places the body at `body_frame` as an
ordinary reset places it at that row: the row's positions and velocities with
the reset noise, no lift (the paper does not state one), a zero previous
action and the history filled with the first frame. The reset noise limits
equal those of StableMimic's implementation details table (Reset row); the
rotation is applied as one axis-angle vector. The command starts at `frame`.
A recovery start gets `body_frame` = start row and `frame` = command row; any
other reset gets both at its reset row.

## The hidden row and what reads it

With the options the state holds five more keys, which `compact`, `restore`,
`select_reset` and the start replay carry like the others:

| Key | Meaning |
| --- | --- |
| `hidden_frame` | the row the last control targeted: the hidden row during recovery, the command row otherwise (at a reset: the body row) |
| `recovering` | whether the hidden row is followed (the recovery regime) |
| `error_controls` | consecutive controls of target error |
| `blend_left` | blend controls still to run |
| `command_offset` | horizontal offset of the command reference, set at the switch back to the command |

| Reads the hidden row | Reads the command row |
| --- | --- |
| the six tracking terms of the reward | the actor's goal rows and anchor orientation |
| the termination penalty (`--term-penalty 2 0.08 0.10`, computed by `hazard_features`: it grows quadratically once the pelvis height difference exceeds 0.08 m or the gravity-z difference exceeds 0.10) | the proprioception history |
| the recovery stop | the action base and the setpoint |
| the critic during recovery: its goal rows, reference torso pose and reference root height | the motion encoder's outputs over a window, computed from the upcoming command rows |
| | the clip end and the episode limit |
| | the reset bins |
| | the recipe's height and tilt termination, outside recovery |

The actor gets no regime flag and no blend count, so it sees what it sees at
deployment. The critic's input keeps its 825 entries and has no regime entry.

## Order inside a control

A control's regime is that of the state it starts from: **recovery** while
`recovering`, **blend** while `blend_left` > 0, **tracking** otherwise. In
each control:

1. The actor acts on the command row's inputs; the setpoint uses the command
   row's action base.
2. The physics steps; the command row advances to `min(frame + 1, clip last)`;
   the history appends the reached state's frame.
3. The target row is `min(hidden + 1, get-up end of hidden)` for a recovering
   state, so the hidden row advances and holds at its get-up end; otherwise
   it is the command row. It is stored as `hidden_frame`.
4. The pelvis height and gravity-z differences of the body are taken against
   the target row and against the command row.
5. A recovering control is a control of **target error** when the body is
   outside the recipe's termination thresholds (0.25 m pelvis height
   difference, 0.8 gravity-z difference; the paper gives no training
   thresholds) at the target row, or, at the get-up end, at the command row.
   An error adds one to `error_controls`; a control without error sets it
   to 0.
6. The **switch**: a recovering control at the get-up end with the body within
   both rows' thresholds stops recovering. `blend_left` becomes BLEND, and
   later controls lower it by one to 0. `command_offset` becomes the body's
   torso position minus the command's torso position, horizontally and
   without gradient, and the command's yaw is kept (StableMimic, Section
   III-D, re-aligns the command's horizontal translation to the recovered
   pose and keeps its yaw; the torso is this repository's choice).
7. The reward and the termination penalty are computed at the target row, and
   the termination of the reached state as below.
8. `done` = termination, clip end or episode limit. A recovery stop is a
   termination, so the episode ends without a bootstrap value; the clip end
   and the episode limit bootstrap as in the recipe.

Nothing else changes at the switch: the previous action, the history, the
episode step and the command row continue.

**Reward.** A recovery control takes the reward scale times each of the six
tracking terms at the target row, with the squared torso height difference
alone as its anchor-position error (StableMimic, Section III-C, uses root
height; the torso link is this repository's choice). A blend control with
`left` blend controls still to run takes `(BLEND - left + 1) / BLEND` times
the six terms: 1/75 at the first blend control, 1 at the last. The paper gives
neither the shape of this weight nor a value for its separate transition
coefficient; no further coefficient is applied here. A tracking control takes
the recipe's terms. Outside recovery the anchor-position reference and the
critic's reference torso are the command torso moved by `command_offset`. The
action-rate term (from the command row's action base), the joint-limit term
and the termination penalty keep the recipe's weights. Recovery rows can
exceed the recipe's largest step reward (`max_reward`: 0.02 s times the sum of
the six tracking-term weights), so `reward_mean` of the log is not
comparable between the two conditions.

**Termination.** While the reached state is recovering, the recipe's height
and tilt terms against the command row are masked (StableMimic, Section III-B:
termination is measured against the get-up target during recovery); a
non-finite state still ends the episode at once. A reached state with STOP or
more `error_controls` is the recovery stop.

**Reset bins.** Only transitions that start in tracking update the reset-bin
statistics; recovery and blend transitions leave the bins of the command rows
as they are. The critic normaliser is updated from all rows.

## Records

`RUN/recovery.json` holds the settings (with the two clip lists), which parts
read the hidden row and which the command row, and six row counts: the low
rows of the hidden clips, those without a get-up end in their clip, those
beyond the longest span, the start rows, the distinct get-up ends and the
command rows. It is a separate file because the exporter checks the fields of
`config.json` exactly. `config.json` and the `resolved_config` of the full
training state (`training_state_itNNNNNN.pt`) record the seven options as
parsed arguments, null in a run without them. With the options, the collector
and replay states in the full training state also hold the five recovery
keys. `ckpt/it_N.pkl` holds no recovery fields.

`log.jsonl` has these extra columns per update, over the window's rows
(controls times environments) by the regime each control started in:

| Column | Meaning |
| --- | --- |
| `recovery_window_row_share_tracking`, `_recovery`, `_blend` | share of the window's rows in each regime |
| `recovery_tracking_window_rows_total` | running total of the window rows in tracking since the run started |
| `recovery_switches`, `recovery_stops` | switches back to the command, recovery stops |
| `recovery_switch_yaw_mean_rad`, `recovery_switch_yaw_p90_rad` | mean and 90th percentile of the absolute yaw between the body's torso and the command's torso at the switches; null without a switch |
| `recovery_clip_ends`, `recovery_episode_limits` | clip ends and episode limits of recovery controls |
| `recovery_terminations_in_blend` | terminations of blend controls |
| `recovery_tracking_reward_mean` | mean reward over the tracking rows divided by the recipe's largest step reward; null without a tracking row |

**Tracking exposure.** A control run trains on `num_envs * horizon` tracking
rows per update (262,144 at 4,096 environments and horizon 64). The control
update matched in tracking exposure to a recovery run's update is the first
whose running total reaches that update's
`recovery_tracking_window_rows_total`.

## Differences from StableMimic

| StableMimic | This repository |
| --- | --- |
| two experts whose actions a mixing network on proprioception blends, trained with extra mixing losses (Sections III-A, III-C) | one actor network |
| start frames half uniform, half weighted towards failed phases (Section III-B) | start rows uniform |
| a success term in the reward, not defined further (Section III-C) | no success term |
| action-rate, joint-limit, self-collision and power penalties (Section III-C) | the recipe's action-rate and joint-limit penalties |
| the critic reads the command and the active get-up reference (Section III-A) | during recovery, the critic reads the hidden row in place of the command row |
| 20 s training episodes (implementation details table) | at most 500 controls (10 s) |

For the start, command and get-up end rows, the settings table gives what
the paper states.

## Not implemented

- pushes during training (the paper's pushes belong to its evaluation);
- falls during tracking turned into recovery during training;
- the residual action base with recovery (refused);
- the JAX trainer (`train.py`).

## Export

The policy of a recovery run exports like any other run; `--critic-out` is
refused for it ([EXPORT.md](EXPORT.md)).

## Rollouts

`python -m nablamimic.engine_rollouts` runs a trained policy closed-loop from
placed lying starts, their upright twins, clips' first rows and pushes, and
records each rollout ([ROLLOUTS.md](ROLLOUTS.md)).

## Tests

`tests/test_recovery.py` runs on CPU with a synthetic get-up library and a
stand-in step that holds or scripts the root. It covers the row tables, the
draws and placement of the starts, the unchanged path without the options,
the readers of the hidden row on both action bases, the advance of the command
and hidden rows, the stop, the switch, the blend and the switch yaw, the
reward's form and gradient, the termination penalty at the hidden row, the
replay keys, a stop inside a window, the reset-bin mask, the log columns and
the option refusals.

`TorchTrainer` needs CUDA, so these tests leave out the trainer's own wiring
(the recovery starts among the collector's initial states and their random
episode ages, the replay ring
with the recovery keys, the reset-bin update fed with the window's regimes,
the merge of the recovery columns into the log row, and that whole training
updates without the options compute what the recipe computes), compiled
execution, CUDA-graph capture with the recovery keys and the memory of a
full-size run.
