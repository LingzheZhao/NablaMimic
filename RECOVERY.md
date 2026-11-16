# Recovery training from placed lying starts

The [PyTorch trainer](TORCH.md) can train a tracking policy that also gets up
from lying on the floor and returns to its command. A share of the episodes
starts with the body placed at a low row of a get-up recording. During such an
episode the reward follows that recording, called the hidden reference. The
actor still reads the command, which starts at a separate upright row and runs
on as in any episode. Once the body is up, a blend leads the reward back to the
command.

The method follows StableMimic (arXiv 2608.02385v1) as the BeyondMimic
reproduction (BM) implemented it in its own NablaMimic trainer. Each value
below is marked as **BM's reading of StableMimic** (a value that BM's
recipes, code or results attribute to the paper) or **BM's choice**. These
attributions are BM's; they are not checked against the paper in this
repository.

```bash
python -m nablamimic.torch_train --config configs/recovery.args --out runs/recovery_s0 --iters 3000
python -m nablamimic.torch_train --config configs/absolute.args --out runs/absolute_s0 --iters 3000
```

`configs/absolute.args` is the 0.2.0 recipe (`configs/base/ours.args`) on
NablaEngine with absolute actions: the joint setpoints are the default pose
plus the scaled action, so no reference row enters the action. It is the
control condition. `configs/recovery.args` adds the recovery options and
nothing else. The options go together: a partial set is refused, and so is
`--residual-action` with them. Without them the trainer runs the recipe
unchanged: no extra random draw, the same state keys, and the same reset,
step, reward, termination, critic and log.

## The setting

| Option | Value | Meaning | Source |
| --- | --- | --- | --- |
| `--recovery-share` | 0.5 | probability that a reset is a recovery start | BM's reading of StableMimic (equal tracking and recovery sampling) |
| `--recovery-hidden-clips` | `fallAndGetUp1_subject1`, `fallAndGetUp1_subject5`, `fallAndGetUp2_subject2`, `fallAndGetUp2_subject3`, `fallAndGetUp3_subject1` | clips whose low rows are recovery starts and whose rows are the hidden reference | BM's choice |
| `--recovery-command-excluded-clips` | the five hidden clips and `fallAndGetUp1_subject4` | clips that never give the command of a recovery start | BM's choice |
| `--recovery-low-pelvis HEIGHT UPZ` | 0.50 m, 0.5 | a reference row is low when its pelvis is below HEIGHT or the world z of its pelvis up axis is below UPZ (a tilt beyond 60 degrees) | BM's choice: StableMimic's fall rule as BM applied it to the reference pelvis |
| `--recovery-rows UPRIGHT_RUN LONGEST_SPAN COMMAND_UPRIGHT` | 50, 375, 500 | a get-up ends at the last row of its first 50 upright rows (1 s); at most 375 rows (7.5 s) from a start row to its get-up end; a command row has 500 upright rows ahead inside its clip | BM's choice |
| `--recovery-reward-scale` | 2.5 | factor on the six tracking terms while the hidden reference is followed | BM's reading of StableMimic |
| `--recovery-controls STOP BLEND` | 100, 75 | consecutive controls of target error that stop a recovery episode (2.0 s); controls of the blend back to the command (1.5 s) | BM's reading of StableMimic; BM's choice to count consecutive controls and to blend linearly |

The anchor term during recovery reads the torso height alone: BM's reading of
StableMimic (root height), with the torso link as BM's choice. The termination
thresholds are the recipe's (0.25 m pelvis height, 0.8 gravity z). BM's
recipe has a further factor on the blend, set to 1.0 and marked as not given
by the paper; it is not an option here.

## Rows of the motion library

`src/nablamimic/recovery.py` computes the tables once when the environment is
built; the run writes their counts to `RUN/recovery.json`.

- A row is **low** where its reference pelvis is below HEIGHT or the world z
  component of its pelvis up axis is below UPZ. Every other row is upright.
- The **get-up end** of row `r` is the last row of the first run of
  UPRIGHT_RUN consecutive upright rows that ends at or after `r` inside its
  clip (none where the clip holds no such run). All rows from a low row to
  that end share one end.
- A **start row** is a low row of a hidden clip whose get-up ends inside its
  clip at most LONGEST_SPAN rows later.
- A **command row** lies outside the command-excluded clips, and it and the
  next COMMAND_UPRIGHT rows are upright and before its clip's last row. With
  COMMAND_UPRIGHT equal to the episode limit, no clip end and no low command
  meets a recovery episode.

The counts are the low rows of the hidden clips, those without a get-up end in
their clip, those beyond the longest span, the start rows, the distinct get-up
ends and the command rows. Unknown clip names, an empty start or command
table, and a hidden clip that is not command-excluded are refused. (A reset
is a recovery start where its body row differs from its command row, which
needs the two tables apart.)

## Starts

Every reset draw goes through `ResetSampler.sample`: the collector's initial
states (which then get random episode ages, recovery starts included, as any
initial state), the collector's resets and the window's resets. It draws, in
this order:

1. the reset rows from the reset bins;
2. the reset noise;
3. one uniform per environment, which makes the environment a recovery start
   below the share;
4. a start row, uniform over the start rows;
5. a command row, uniform over the command rows.

Each draw is made for every environment, and the choice selects. Without the
options only the first two draws are made.

`env.reset(frame, noise, body_frame)` places the body at `body_frame` exactly
as an ordinary reset places it at that row: the row's positions and velocities
with the same reset noise, no lift, a zero previous action and the history
filled with the first frame. The command starts at `frame`. A recovery start
gets `body_frame` = start row and `frame` = command row; any other reset gets
both at its reset row.

## The hidden row and who reads it

With the options the state holds five more keys, which `compact`, `restore`,
`select_reset` and the start replay carry like the others:

| Key | Meaning |
| --- | --- |
| `hidden_frame` | the row the last control targeted: the hidden row while recovering, the command row otherwise (at a reset: the body row) |
| `recovering` | whether the hidden row is followed |
| `error_controls` | consecutive controls of target error |
| `blend_left` | blend controls still to run |
| `command_offset` | horizontal offset of the command reference, set at the switch |

| Reads the hidden row | Reads the command row |
| --- | --- |
| the six tracking terms of the reward | the actor's goal rows and anchor orientation |
| the termination penalty (`hazard_features`) | the proprioception history |
| the recovery stop | the action base and the setpoint |
| the critic, while recovering: its goal rows, reference torso pose and reference root height | the encoder latents of the window |
| | the clip end and the episode limit |
| | the reset bins |
| | the recipe's height and tilt termination, outside recovery |

The actor gets no regime flag and no blend count; it sees what it would see
at deployment. The critic's input keeps its 825 entries and has no regime
entry; after the switch it reads the command row with the reference torso
moved by the command offset.

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
5. A recovering control is a control of target error when the body is outside
   the thresholds of the target row, or, at the get-up end, outside those of
   the command row. An error adds one to `error_controls`; a control without
   error sets it to 0.
6. The **switch**: a recovering control at the get-up end with the body within
   both rows' thresholds stops recovering. `blend_left` becomes BLEND, and
   later controls lower it by one to 0. `command_offset` becomes the body's
   torso position minus the command's torso position, horizontally and
   without gradient; the command's yaw is kept.
7. The reward is computed at the target row (below), the termination penalty
   at the target row, and the termination of the reached state (below).
8. `done` = termination, clip end or episode limit. A recovery stop is a
   termination, so the episode ends without a bootstrap value; the clip end
   and the episode limit bootstrap as in the recipe.

Nothing else changes at the switch: the previous action, the history, the
episode step and the command row continue.

**Reward.** A recovery control takes the reward scale times each of the six
tracking terms at the target row. Its anchor-position error is the squared
torso height difference alone. A blend control with `left` blend controls
still to run takes `(BLEND - left + 1) / BLEND` times the six terms: 1/75 at
the first blend control, 1 at the last. A tracking control takes the recipe's
terms. Outside recovery the anchor-position reference is the command torso
moved by `command_offset`. The action-rate term (from the command row's
action base), the joint-limit term and the termination penalty keep the
recipe's weights. Recovery rows can exceed the recipe's largest step reward,
so `reward_mean` of the log is not comparable between the two conditions.

**Termination.** While the reached state is recovering, the recipe's height
and tilt terms against the command row are masked; the non-finite term still
ends the episode at once. A reached state with STOP or more
`error_controls` is the recovery stop. BM's code masks by the state the
control starts from; the reached state gives the same result, because a
switching control is within the command row's thresholds and a stopping
control is still recovering.

**Reset bins.** Only transitions that start in tracking update the reset-bin
statistics. Recovery and blend transitions leave the bins of the command rows
as they are. The critic normaliser is updated from all rows.

## Records

`RUN/recovery.json` holds the setting (with the two clip lists), the six row
counts and which parts read the hidden row and which the command row. It is a
separate file because the exporter checks the fields of `config.json`
exactly. `config.json`, the `resolved_config` of the full training state
(`training_state_itNNNNNN.pt`) and the exported policy's
`source_run.parsed_arguments` record the seven options as parsed arguments,
with null values in a run without them. With the options, the collector and
replay states in the full training state also hold the five recovery keys.
`ckpt/it_N.pkl` holds no recovery fields.

`log.jsonl` gains these columns per update, over the window's rows (controls
times environments) by the regime each control started in:

| Column | Meaning |
| --- | --- |
| `recovery_window_row_share_tracking`, `_recovery`, `_blend` | share of the window's rows in each regime |
| `recovery_tracking_window_rows_total` | running total of the window rows in tracking since the run started |
| `recovery_switches`, `recovery_stops` | switches back to the command, recovery stops |
| `recovery_switch_yaw_mean_rad`, `recovery_switch_yaw_p90_rad` | absolute yaw between the body's torso and the command's torso at the switches; null without a switch |
| `recovery_clip_ends`, `recovery_episode_limits` | clip ends and episode limits of recovery controls |
| `recovery_terminations_in_blend` | terminations of blend controls |
| `recovery_tracking_reward_mean` | mean reward over the tracking rows over the recipe's largest step reward; null without a tracking row |

**Exposure.** The tracking rows a recovery run has trained on after update N
are `recovery_tracking_window_rows_total` of that update's row. A control run
trains on `num_envs * horizon` tracking rows per update (262,144 at 4,096
environments and horizon 64), so after N updates on N times that. The control
update matched in tracking exposure to a recovery run's update is the first
whose total reaches the recovery run's.

## Choices kept from BM

- The critic reads the hidden row alone while recovering. BM's reading of
  StableMimic gives the critic both rows.
- Episodes last at most 500 controls (10 s); BM's reading of StableMimic gives
  20 s.
- The regularisers are the recipe's two; BM's reading of StableMimic also
  names self-collision and power terms.
- "Persisting" target error is counted as consecutive controls of either
  termination difference.
- A recovery start has no lift, keeps the start row's reference velocity and
  the recipe's reset noise.
- Recovery starts among the collector's initial states get random episode
  ages, as every initial state does.
- COMMAND_UPRIGHT equals the episode limit.

## Not implemented

- two experts with a mixing network;
- failure-adaptive weights of the start rows;
- a success reward;
- pushes, or falls during tracking turned into recovery, during training;
- a regime entry in the critic's input, or the hidden row in the actor's;
- the residual action base with recovery (refused);
- the JAX trainer (`train.py`).

## Export

The policy of a recovery run exports like any other run
([EXPORT.md](EXPORT.md)); its options are recorded in the declaration's
`source_run.parsed_arguments`. `--critic-out` is refused for such a run: its
critic reads the hidden row in place of the command's goal while recovering,
which the critic file's input declaration does not describe.

## Checks

`tests/test_recovery.py` runs on CPU with a synthetic get-up library and a
stand-in step that holds or scripts the root. It covers the row tables, the
draws and placement of the starts, the unchanged path without the options,
the readers of the hidden row on both action bases, the clocks, the stop, the
switch, the blend and the switch yaw, the reward's form and gradient, the
termination penalty at the hidden row, the replay keys, a stop inside a
window, the reset-bin mask, the log columns and the option refusals.

`TorchTrainer` needs CUDA, so these tests do not cover the trainer's own
wiring: the recovery starts among the collector's initial states with their
random episode ages, the replay ring with the recovery keys, the reset-bin
update fed with the regimes of the training window, the merge of the
recovery columns into the log row, and that whole training updates without
the options compute what the recipe computes. Compiled execution, CUDA-graph
capture with the recovery keys and the memory of a full-size run also need a
GPU and are not covered by these tests.
