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

## Videos of a record

`python -m nablamimic.render_rollouts` makes videos of a record by replaying
its recorded states: no physics and no policy run, so a video is a replay,
not a second rollout.

```bash
MUJOCO_GL=egl python -m nablamimic.render_rollouts --record rollouts/recovery_s0_3000 \
    --trials 0 4 --stride 2 --out videos/recovery_s0_3000
```

A frame shows the robot after a control and, translucent, the reference pose
of the command row after it (`frame_after`, see [Record](#record)): the
command, not a prediction. A trial that ended stays at its last recorded
state, labelled with the control that ended it. A second `--record` of the
same trials and starts is shown beside the first; `--help` lists the options.
For each trial the output folder (which must not exist) holds
`trial_NNNN.mp4` (H.264), `trial_NNNN.png` (8 frames) and `trial_NNNN.json`.

`--ghost robot` draws the command's pose with its root moved horizontally to
the robot's root at every frame (height, orientation and joints unchanged),
which keeps it beside a body placed at another clip's pose; the default
`world` draws it where the command places it. A video ends `--tail` seconds
(default 2) after the earlier of the control that ends the trial and the
clip end, the first control whose `frame_after` equals `clip_last_row`; side
by side, after the later of the two panels' points. It never runs past the
record, and `all` or a negative tail shows every control. The JSON gives the
options, the stopping control and its reason.

The model is built by `robot.build` from `MENAGERIE_DIR`, and the reference
poses are read from `MOTION_CACHE_DIR`. `LAFAN1_G1_DIR`, `MOTION_CACHE_DIR`
and `MENAGERIE_DIR` must equal `settings.inputs` in the record's
`description.json`. The tool needs an offscreen OpenGL backend selected by
`MUJOCO_GL` (`egl` or `osmesa`) and `ffmpeg` with libx264 on `PATH`.

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

`tests/test_render_rollouts.py` checks the replay and its refusals with a
synthetic record and clip cache, and renders a short video where
`MUJOCO_GL=egl` opens (otherwise it checks the refusal).
