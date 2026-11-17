# Rollouts of a trained policy from placed starts and after pushes

`python -m nablamimic.engine_rollouts` runs the policy of a numbered
checkpoint of the [PyTorch trainer](TORCH.md) on NablaEngine, the simulator
the trainer uses, closed-loop: each action is computed from the state the
previous action produced. It runs the trials of a scene file and writes one
record per trial. A trial can start from a placed lying body (as a
[recovery start](RECOVERY.md) does), from the command row alone, from a clip's
first row, and can carry a push.

A **control** is one 0.02 s policy step (50 Hz); a **row** is one 0.02 s frame
of the motion library. The **command** is the reference row the policy tracks;
it advances one row per control and holds at its clip's last row.

```bash
python -m nablamimic.engine_rollouts --run runs/recovery_s0 --iter 3000 \
    --scene scene.json --termination nonfinite --controls 1000 --out rollouts/recovery_s0_3000
```

| Option | Meaning |
| --- | --- |
| `--run` | run directory of `nablamimic.torch_train` (`config.json`, `ckpt/`) |
| `--iter` | update of the checkpoint `ckpt/it_NNNNNN.pkl` (six digits) |
| `--actor-key` | `actor_ema` (the running average of the weights, the default) or `actor` (the weights as trained) |
| `--scene` | scene file (JSON), below |
| `--termination` | `recipe` or `nonfinite`, below |
| `--controls` | controls per trial |
| `--out` | output directory; it must not exist |

## Requirements and refusals

- Exactly one CUDA device.
- A run of the PyTorch trainer (`training_framework` `torch` in its
  `config.json`).
- `LAFAN1_G1_DIR`, `MOTION_CACHE_DIR` and `MENAGERIE_DIR` equal to the values
  in the run's `config.json` (an unset variable matches only an unset value).
  The scene's rows index the motion library these paths locate, and the
  program compares the values, not the files. A run trained with a temporary
  cache path can therefore be rolled out only where that path holds the same
  cache.
- A source revision: `NABLA_SOURCE_REVISION`, or else the commit (`HEAD`)
  of the source checkout.
- An engine whose capacity bits (below) are 4 and 16.

The scene is checked against the motion library before the output directory
is made.

## Environment and policy

The environment is the run's (`torch_train.build_environment` with the run's
`config.json`: its clips, solver iterations, action base and action-rate
space) without the seven `--recovery-*` options of
[recovery training](RECOVERY.md), so no hidden reference, recovery stop or
termination masking enters a rollout. Domain randomisation is off (foot
friction 1, zero joint-encoder bias, zero torso centre-of-mass offset),
observation noise is off, and the actions are the actor's output without
noise. The policy acts as in training: `env.policy_obs` into `actor_apply`.
The network matrix-product precision is the run's (`matmul_precision`). The
engine runs forward only, without a tape (the record of the step that
gradients need).

## Scene

`{"trials": [TRIAL, ...]}`, one environment of one batch per trial; unknown
keys are refused at every level. A trial holds:

| Key | Meaning |
| --- | --- |
| `command_row` | library row the command starts from (required) |
| `body_row` | library row whose reference posture the body is placed at; absent: the command row |
| `reset_uniforms` | uniforms in [-1, 1] that scale the reset noise; absent: zeros. `position` [3] times 0.05, 0.05, 0.01 m on the root position; `rotation` [3] times 0.1, 0.1, 0.2 rad, a rotation vector premultiplied on the root orientation (world frame); `velocity` [6] times 0.5, 0.5, 0.2 m/s and 0.52, 0.52, 0.78 rad/s on the root's six velocity coordinates; `joints` [29] times 0.1 rad on the joint positions in actuator order. These are the half-widths of the training reset noise (`TorchTrackingEnv.sample_reset_noise`) |
| `push` | `{"control", "velocity_xy"}`: a world planar velocity change (m/s); or `{"control", "force", "duration", "direction"}`: force (N) times duration (s) divided by the whole-body mass (`body_subtreemass[0]` of the model), along `forward`, `backward`, `left` or `right` of the pelvis yaw at that control |

Rows must lie in the library, uniforms in [-1, 1], push controls inside the
rollout, and forces and durations must be positive and finite.

A placed lying start gives `body_row` (a low row of a get-up clip) and its
uniforms; its twin, an upright start at the same command, gives only the same
`command_row`; a clip's first row gives that row as `command_row`.

Every trial starts through the training reset,
`env.reset(command_rows, noise, body_frame=body_rows)`, as a recovery start
does. The reset places the body row's positions and velocities with the noise,
limits the joints to the soft joint range (the central 90% of each joint's
range) for every trial, with zero uniforms too, sets a zero previous action,
fills the observation history with the first frame and stores the positions
as the engine normalises them (unit root quaternion).

A push at control `c` is added to the root's planar world velocity
(`qvel[:, 0:2]`) of the state control `c` starts from, before that control's
action. The action reads the history of earlier controls, so the policy sees
the push from control `c + 1` on.

## Terminations

- `--termination recipe`: the termination of the tracking recipe
  (`configs/base/ours.args`) ends a trial: the pelvis height differs from the
  command row's by more than 0.25 m, or gravity z (the z component of the unit
  gravity direction in the pelvis frame) differs from the command row's by
  more than 0.8, or the state is not finite.
- `--termination nonfinite`: only a non-finite state ends a trial.

Under `recipe` a placed lying start ends at its first control, because a
lying body is more than 0.25 m below the command's pelvis height; placed
starts are therefore run with `nonfinite`.

Neither the clip end nor the episode limit of training ends a trial. An ended trial keeps
the state it started its last control from: the batched step still computes
it, its result is not used, no non-finite state reaches the engine again and
no other trial depends on it. The control that ends a trial is recorded
(`available` unless its state is not finite); later controls are not, and a
later push is not applied.

## Record

`description.json` is written before the rollouts and again after them. It
holds the run directory, the checkpoint path, its completed updates and the
weights (`actor_key`); the run's source revision (`git_commit`), engine and
GPU from its `config.json`, and this program's source revision; the scene path and its
content; the termination and the controls; the engine record of
`torch_ffi.init`; the GPU name and the PyTorch version; under `settings` the
run's clips, solver iterations, action base, action-rate space and
matrix-product precision (requested and observed), the environment and policy
settings above, the termination thresholds, the whole-body mass and the three
library paths; and, after the rollouts, `engine_diagnostics`, the engine
counters of the rollouts. The counters include the steps of ended trials,
which the batch still computes; per-trial counts come from the fields below.

`rollouts.npz` holds, per control and trial (`[C, T, ...]` with
`C` controls and `T` trials; after a trial's end NaN, -1 or false):

| Field | Shape, type | Meaning |
| --- | --- | --- |
| `body_pos_w`, `body_quat_w` | [C, T, 14, 3], [C, T, 14, 4] float32 | world positions and orientations (WXYZ quaternions) of the 14 tracked bodies after the control |
| `qpos` | [C, T, 36] float32 | root and joint positions after the control, as the engine normalises them |
| `qvel` | [C, T, 35] float32 | velocities after the control: `[0:3]` the pelvis origin's world linear velocity, `[3:6]` the pelvis angular velocity in its body frame, `[6:]` the joints' |
| `frame_before`, `frame_after`, `clip_last_row` | [C, T] int32 | command row before and after the control; last row of its clip (the clip end is `frame_after == clip_last_row`) |
| `available` | [C, T] bool | the state after the control is recorded and finite |
| `done` | [C, T] bool | the control ends the trial |
| `termination_height`, `termination_tilt`, `termination_nonfinite` | [C, T] bool | the recipe's three termination terms, recorded under either termination |
| `push_applied` | [C, T] bool | the trial's push is applied at the control |
| `engine_newton_iterations`, `engine_contacts`, `engine_constraint_rows`, `engine_info_bits` | [C, T] int32 | the engine's report of the control's step per environment (counts of the last substep, bits over the substeps) |
| `engine_contact_capacity_exceeded`, `engine_row_capacity_exceeded` | [C, T] bool | contacts or constraint rows beyond the engine's capacity are dropped in the step (bits 4 and 16) |

and per trial:

| Field | Shape, type | Meaning |
| --- | --- | --- |
| `command_row`, `body_row` | [T] int32 | the trial's rows |
| `push_control` | [T] int32 | the push's control, -1 without a push |
| `push_velocity_xy` | [T, 2] float32 | the world velocity change applied, zero without one |
| `start_qpos`, `start_qvel`, `start_body_pos_w`, `start_body_quat_w` | [T, 36], [T, 35], [T, 14, 3], [T, 14, 4] float32 | the placed state before the first control |

For a push at control `c`, the pelvis yaw that turns a force push and the
root velocity before the push are those of control `c - 1`'s `qpos` and
`qvel` (`start_qpos` and `start_qvel` for `c = 0`); the push adds
`push_velocity_xy` to that velocity.

## Rollouts on MuJoCo C

`python -m nablamimic.mujoco_rollouts` runs the same rollouts with the
physics stepped by MuJoCo's C engine (the `mujoco` package) instead of
NablaEngine:

```bash
python -m nablamimic.mujoco_rollouts --run runs/recovery_s0 --iter 3000 \
    --scene scene.json --termination nonfinite --controls 1000 --threads 16 \
    --out rollouts/recovery_s0_3000_mujoco_c
```

The options are those of `engine_rollouts` with two more: `--threads`, the
threads that step the trials (MuJoCo releases the interpreter lock while it
steps), and `--checks RECORD` (below). The requirements are the same, except
that NablaEngine is needed only by `--checks`.

Only the step and kinematics functions of `TorchTrackingEnv` differ. The
scene, the run's environment, the observations and history, the action base,
scale and setpoint range, the placement, the policy on the CUDA device at the
run's matrix-product precision, the terminations, the loop (`rollout`), the
record fields and the description's fields are those of `engine_rollouts`. The
environment is built as `torch_train.build_environment` builds it; the program
makes the same construction itself because that function takes no kinematics
function.

**Model.** MuJoCo C steps `env.m`, the model object that `engine_rollouts`
builds NablaEngine from (`env.build_model`: the Menagerie scene's options with
the run's solver iterations and a 5 ms timestep, the mjlab contact pairs and
PD gains, the widened setpoint range, sensors disabled), unchanged. The
program refuses a model with actuator activations, with an actuator that does
not drive the hinge joint at `qpos[7 + i]` (the order of actions, reference
and record), with more than one kinematic tree, with a solver other than
Newton, an integrator other than implicitfast or a friction cone other than
pyramidal (the only ones NablaEngine steps), or with a geom pair that the
contype / conaffinity filter lets collide besides the explicit pairs. The
floor keeps its contype and conaffinity; every robot geom has zero, so the
filter adds no pair.

**State.** Each trial keeps one float64 `MjData` for the whole trial. A
control sets `ctrl`, calls `mj_step` four times (4 x 5 ms), and then
`mj_kinematics`, `mj_comPos` and `mj_comVel`, because after `mj_step` the body
poses are those before its last integration. The environment receives float32
copies (in `qpos` with the root quaternion normalised, as NablaEngine hands it
out). A control continues the trial's `MjData` when the state handed in is the
copy handed out. A push, where only `qvel[0:2]` differs, adds the float64
difference of the two float32 velocities, which differs from
`push_velocity_xy` by float32 rounding. Any other state seeds the `MjData`
from the state handed in (`mj_resetData`, then `qpos`, `qvel` and the warm
start): the placement, with a zero warm start as at NablaEngine's reset, and
an ended trial, which the loop holds at its earlier state and whose result is
not used. The warm start between controls is MuJoCo's own `qacc_warmstart`.

**Set-up differences.** These are the differences other than the engine
itself; `engine.differences` in `description.json` lists them too.

| Item | NablaEngine | MuJoCo C |
| --- | --- | --- |
| state between controls | float32, re-entered at every control | float64 `MjData`, float32 copies handed out |
| solver | Newton in float32 with the model's iterations, tolerance (scaled by the mean inertia) and impratio, an exact line search of at most 40 steps that uses neither line-search option (NablaEngine's model classification, `nablaengine.model.fields`; check (b) measures the step) | Newton in float64 with the model's iterations, tolerance, line-search iterations and tolerance, and impratio |
| constraint islands | one problem per world | each island solved on its own, with its own tolerance exit, unless `mjDSBL_ISLAND` is set (`engine.constraint_islands` in the description); the model is one kinematic tree, so a substep has at most one island |
| integrator, cone, timestep, substeps | implicitfast, pyramidal, 5 ms, 4 | the model's, refused unless implicitfast and pyramidal; 5 ms, 4 |
| actuators and gains | the model's gain and bias and, where a joint is `actfrclimited`, the force limit `jnt_actfrcrange` (NablaEngine's model classification); `ctrl` is not clamped to `ctrlrange` (its documented semantics); check (b) on lifted trials measures the actuators | the model's gain and bias, the force limit `jnt_actfrcrange` where a joint is `actfrclimited`, and, where an actuator is `ctrllimited`, `ctrl` clamped to the float64 `ctrlrange` after the environment has clipped it to the float32 copy of that range; check (c) gives the flags |
| contacts | the explicit pairs; a fixed number of contacts and constraint rows per world (the build's record in `engine` of an `engine_rollouts` description), beyond which they are dropped (bits 4 and 16) | the explicit pairs; contacts or rows dropped only when MuJoCo's arena is full, reported by its warnings as bits 4 and 16 |
| foot friction, torso centre of mass | per trial (1 and zero without randomisation) | the model's (1) and zero; any other value is refused |
| divergence | a non-finite state | MuJoCo resets a trial's data when `qpos`, `qvel` or `qacc` holds a value that is non-finite or above 1e10 in magnitude; that control's state is handed out as NaN |

The divergence rule differs from the record's non-finite term in two ways: a
finite value above 1e10 counts, and MuJoCo checks positions and velocities at
the start of a step, so a state that becomes too large in a control's last
substep is reported at the next control.

**Record.** The fields and their meaning are those above.
`engine_newton_iterations`, `engine_contacts` and `engine_constraint_rows` are
MuJoCo's `solver_niter` (the largest over its constraint islands, of which the
model has at most one), `ncon` and `nefc` of the last substep.
`engine_info_bits` holds bits 4 and 16 from MuJoCo's contact-full and
constraint-full warnings in any substep, and no other bit: NablaEngine's other
bits mark model features it does not step (its documentation), and MuJoCo C
steps every feature of the model. The capacity fields follow from those two
bits. `description.json` has the keys of `engine_rollouts`; its `engine`
describes MuJoCo C (version, threads, options and flags, the number of pairs
and kinematic trees, whether constraint islands are on, the set-up
differences), and `engine_diagnostics` holds the program's
counters: calls, trial controls, substeps, trial controls that seed, push,
continue or diverge, trial controls with either capacity bit, and MuJoCo's
warnings by text. As with NablaEngine, the counters and the warnings include
ended trials, which the loop still steps, so an ended trial that diverges
counts at every later control. After the rollouts the program raises unless
the seeded trial controls are the placements and the controls after each
trial's end: an unended trial is seeded once.

A reader of `engine_rollouts` records that requires `engine_diagnostics` and
compares only `scene_content`, `termination`, `controls`, `actor_key` and
`settings.inputs` with its study, and `run`, `checkpoint`, `completed_updates`,
`actor_key` and `run_source_revision` across the rollouts of a save, reads a
MuJoCo C record as it reads a NablaEngine one. The recovery pair study's reader
(`studies/recovery/readings.py`, not part of this branch) compares these keys.
It does not compare `engine`, so it does not refuse rollouts of a save that mix
the two engines, and its readings name the engine only through the rollout
directories they list. Whoever reads a save keeps its rollouts on one engine
and names that engine in the rollout directories or beside the readings.

### Check mode

`--checks RECORD`, with RECORD an `engine_rollouts` output of the same scene,
termination, library inputs and clips, writes `checks.json` in `--out` instead
of rolling out. It decides nothing: differences are reported, not held to a
tolerance. It raises when RECORD's `engine` is not NablaEngine's build record
(a MuJoCo C rollout is refused), when the record's rows are not the scene's,
when a placed tracked body lies nearest another recorded body (the body order
differs), and on the model refusals above (the joint order among them).

- (a) Placement: every trial of the scene is placed through the environment's
  reset with MuJoCo C kinematics; for each start field (`start_qpos`,
  `start_qvel`, `start_body_pos_w`, `start_body_quat_w`) the largest
  difference from the record, its trial, and the entries finite in only one.
- (b) One control: from the placed state and the policy's action at it, one
  control of NablaEngine (twice) and of MuJoCo C (twice, each from a seeded
  `MjData`), on the same float32 inputs. First with the pelvis raised 2 m, so
  that no floor contact acts; self-contacts and joint-limit rows may remain,
  so each trial's contacts, constraint rows and iterations in each engine (of
  the last substep, as both engines report them) are given, for MuJoCo C also
  the largest contact count over the four substeps, and the differences again
  over the trials without contacts at the last substep in either
  (`trials_without_contacts_at_last_substep`). This tests the adapter:
  actuators, gains, the actuator force limit and `ctrl` clamping, integrator
  and gravity. Then as
  placed, with contact. MuJoCo C minus NablaEngine stands beside NablaEngine
  minus NablaEngine (the floor) and MuJoCo C minus MuJoCo C.
- (c) Model: every option of `m.opt`, the disable and enable flags by name,
  each geom's body, type, contype, conaffinity and condim, the pairs, the
  actuators (gain, bias, ranges and their limit flags, armature, damping,
  frictionloss), and the geom
  pairs outside the explicit pairs (none).
- (d) Repeat: two MuJoCo C rollouts of the first 50 controls of the trials
  (fewer with a smaller `--controls`; a later push does not act); whether
  their records are identical byte for byte (so +0.0 and -0.0 differ), else
  the differing fields and the size of each difference (the largest absolute
  difference of a floating-point field, 0 when only signs of zero or NaN
  payloads differ; the count of unequal entries otherwise).

MuJoCo's warnings during the checks are in `mujoco_warnings`.

### What a comparison of the two engines' records shows

Both records are read by the same reader, so a save's readings (get-ups,
rejoins, falls, completed clips, body-position error) can be set side by side
for the two engines, and the trials that change outcome can be listed beside
the trials that change outcome between repeated rollouts on one engine. The
engines differ in implementation, solver details and precision together, so a
difference is not attributed to one of them. A closed-loop trial grows small
differences, and whether a NablaEngine rollout repeats trial by trial is not
established here, so single trials are not compared one to one.

## Tests

`tests/test_engine_rollouts.py` runs the rollouts on CPU with a synthetic
library and a stand-in engine step. It covers: a placed start equals a
recovery training start at the same rows and noise; a twin equals an ordinary
reset at the command row; the uniforms reach the reset by value; a push
changes only its trial's planar root velocity, from its control on (a world
change unturned, a force push turned by the pelvis yaw at its control); each
record field's shape and type; the record holds the stepped state of the
tracked bodies; the clip end holds the command row without ending a trial;
the height and tilt terms each end a trial; the two terminations agree up to
the first recipe termination; an ended trial leaves the other trials' records
bit-equal to a batch without it; invalid scenes and environments are refused;
the command line parses.

They leave out what only the engine establishes (its per-environment report,
the normalised positions, the dynamics of a push) and `main`: the run,
library-path, checkpoint and capacity-bit checks and the files written. These
need a GPU run.

`tests/test_mujoco_rollouts.py` runs MuJoCo C on CPU with the G1 model the
trainer builds (skipped without the Menagerie checkout) and a synthetic
library. It covers: the model passes the refusals and collides only through
its pairs, and a swapped actuator order, another solver, integrator or cone, a
colliding geom and foot pairs outside the model are refused; the island
setting described is the model's flag; controls continue one float64
`MjData` bit-equal to an uninterrupted one, which seeding from the float32
copies does not give; a push adds its change in float64 and leaves the other
trials alone; another state seeds its trial; the bodies are those of the
stepped positions and the counts are MuJoCo's; thread counts give equal
results; a diverging trial is handed out as NaN and seeded by its next state;
warnings are collected from every thread and concurrent counts are not lost;
the kinematics; the refused settings; the record has the fields, shapes and
types of `engine_rollouts.rollout` with a stand-in forward of the engine's
output layout; an ended trial is seeded at every later control, leaves the
other trials' records bit-equal and is counted by the seed check, which
refuses another count; two rollouts repeat byte for byte, and a sign of zero
counts as a difference; the check mode with MuJoCo C standing in for
NablaEngine; a MuJoCo C record is refused as the check mode's record; the
placement check's refusals; the description has the keys of the description
literal in `engine_rollouts.main`; the command line. `main`, the
policy on a CUDA device and NablaEngine itself need a GPU run.
