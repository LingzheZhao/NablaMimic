# Export a policy, critic and reference cache

Two commands write a trained run in files that a program outside this package
reads with NumPy and JSON alone:

- `nablamimic.export_reference_manifest` writes a JSON manifest of the
  processed reference cache the run trained on.
- `nablamimic.export_bm_policy` writes the actor as a tracking policy file
  (format `bm-nabla-tracking-policy-v1`) and, optionally, the critic and its
  input normalizer as a second file (format `bm-nabla-critic-v1`).

Each output is written once: an existing destination is refused.

## Runs the exporter reads

The exporter reads runs of both trainers in this package, the JAX `train.py`
and the PyTorch `torch_train.py`. The run's `config.json` names the trainer: a
`torch_train.py` run records `training_framework` as `"torch"`, a `train.py`
run records no `training_framework`, and any other value is refused.

A `train.py` checkpoint pickles optax optimizer states, so loading it needs
this package's pinned dependency environment. A `torch_train.py` checkpoint
holds NumPy arrays and Python values only, but exporting its run still needs
the optional `torch` dependency, because the configuration check imports the
argument parser of `torch_train.py`. Compiling the model needs the JAX and
MuJoCo packages for either trainer.

Either run must be on one device and one process, must not be resumed, and must
not use `--shac-strict`, which applies the update rule and gradient guard of
SHAC and halts training at an exploding actor gradient.

Only numbered checkpoints `RUN/ckpt/it_N.pkl` are read; `params.pkl` is saved
at a different point of the update loop and is not an input. A run directory
holding `DIVERGED` or `HALTED` is refused.

### train.py runs

- An outer update is one update of the training loop: `num_envs` windows of
  `horizon` simulation steps, then the actor and critic steps. The trainer saves
  checkpoint N before starting update N, after N completed outer updates. The
  checkpoint's `iter` must equal N and its `env_steps` must equal
  `N * num_envs * devices * horizon`.
- The checkpoint stores no completed-update or averaging counter. With
  `actor_ema > 0` (an exponential moving average, EMA, of the actor weights),
  the checkpoint must hold `actor_ema`, which in a run that is not resumed has
  N averaging updates at checkpoint N; with `actor_ema = 0` it must not hold
  one. The declaration records these derived counts and the configured decay
  and warmup. `configs/ours.args` uses decay 0.999 with warmup (from
  `configs/base/ours.args`): the decay after n averaging updates is
  min(0.999, (1 + n) / (10 + n)).
- The checkpoint fields must be exactly those `train.py` writes for the
  configuration: `actor`, `critic`, `cnorm`, `st_a`, `st_c`, `bin_q`, `iter`,
  `wall_s` and `env_steps`, with `st_r` for the action-target optimizer and
  `actor_ema` for the averaged actor. A missing or unsupported field is
  refused; no field is added or converted.
- `config.json` must hold the `train.py` core fields, and no field outside the
  parser's arguments and the fields `train.py` adds. Optional command-line
  records the parser knows may be present without becoming required, and absent
  ones are not filled with defaults. `resume` must be null. A run started
  with `--config` records `argv`, `config_args`, `config_file` and a list of
  `segments` (one per start of the trainer, each with its own engine record);
  it must hold exactly one segment, not resumed.
- `DONE`, when present, must hold the final update, and a checkpoint label
  above it is refused.

### torch_train.py runs

- `config.json` holds exactly the trainer's parsed arguments (its parser adds
  `--matmul-precision`, `--execution`, `--regression-batch`,
  `--initial-checkpoint`, `--save-wall-minutes` and `--full-state-at-end` to
  that of `train.py`), the fields of `train.FIXED` and the fields the trainer
  adds: `training_framework`, `git_commit`, `devices`, `processes`,
  `observed_precision`, `gpu`, `torch_version`, `torch_cuda`, `random_stream`,
  `clock`, `inputs`, `declared_environment` and `engine`. A missing field or a
  field outside that set is refused. The trainer records no `argv`,
  `config_args`, `config_file`, `segments` or `git_dirty`.
- The run must have completed its requested updates: `DONE` holds
  `completed`, `terminal.json` has status `completed` and a
  `completed_updates` equal to `iters`, and there is no `FAILED.json`. The
  trainer writes `DONE` and status `completed` also for a run that its
  `--hours` budget stopped before `iters`; such a run is refused, with both
  numbers named. A checkpoint label above the terminal `completed_updates` is
  refused.
- Checkpoint fields are exactly `actor`, `critic`, `cnorm`, `bin_q`, `iter`,
  `completed_updates`, `ema_updates`, `wall_s`, `env_steps`,
  `training_framework` and `execution`, with `actor_ema` added when
  `actor_ema > 0`. They hold no optimizer state. At checkpoint N, `iter` and
  `completed_updates` must equal N, `env_steps` must equal
  `N * num_envs * horizon`, `ema_updates` must equal N with averaging and 0
  without, and `training_framework` and `execution` must be `"torch"` and the
  configured execution. The declaration takes its counts from
  `completed_updates` and `ema_updates`.
- A run started with `--initial-checkpoint` counts its updates from zero like
  any other run.
- The engine record in `config.json` states the build's contact capacity as
  `maxcon`, and `--engine-contact-capacity` must equal it. The record states
  no constraint-row capacity, so `--engine-constraint-row-capacity` is taken
  as given.
- The networks are saved as the same trees of `[in, out]` kernels as in a
  `train.py` checkpoint, so the policy and critic files have the same entries,
  shapes and orientation for both trainers.

The policy inputs, the action map, the critic inputs and the critic normalizer
of `torch_env.py` are those of `env.py`, and `torch_env.py` compiles its model
with the same `env.build_model`, so one declaration format describes runs of
both trainers.

## Reference manifest

```bash
python -m nablamimic.export_reference_manifest --cache-dir CACHE --clips all \
    --artifact-id REFERENCE_ID --out MANIFEST.json
```

Give the processed cache the run read. `--clips all` takes every clip listed in
`scripts/lafan1_g1.sha256`, in sorted name order, as training with
`--clips all` does; listed names keep their given order. The command checks
that `scripts/download_lafan1.sh` fetches the declared LAFAN1 dataset revision,
and checks each clip's six cache arrays (`qpos`, `qvel`, `body_pos`,
`body_quat`, `body_lin_vel`, `body_ang_vel`) for their names, shapes, body
count and finite values. The cache is not modified.

The manifest (format `nablamimic-reference-manifest-v1`) states the dataset
identity, frame rate, tracked body names in body-axis order, the meaning and
layout of each array, the clip selection and order, each clip's cache file name
and frame count, and each clip's first and last row in the concatenated
library. A program that reads the manifest takes the cache file names from it.
The cache directory is stated relative to the manifest's directory when the
cache lies inside it and as an absolute path otherwise; each clip is at
`<manifest directory>/<cache_directory>/<cache_file>`.

## Policy file

```bash
python -m nablamimic.export_bm_policy --checkpoint RUN/ckpt/it_N.pkl --run-config RUN \
    --policy ema --artifact-id POLICY_ID --source-artifact-id CHECKPOINT_ID \
    --reference-manifest MANIFEST.json --engine-revision ENGINE_REVISION \
    --engine-contact-capacity CONTACTS --engine-constraint-row-capacity ROWS \
    --out POLICY.npz
```

`--run-config` takes the run directory or its `config.json`; the checkpoint
must be in that run's `ckpt/` directory. `--policy live` selects `actor`, the
weights as trained; `--policy ema` selects `actor_ema` and is refused when the
checkpoint holds no averaged weights. `--artifact-id` is the identity the
policy file is published under, `--source-artifact-id` the published identity
of the checkpoint, and `--reference-manifest` the manifest of the reference
cache the run trained on. The NablaEngine revision and the contact and
constraint-row capacities per simulated world are declared by the caller for
the build the run trained with; they are not inferred from the weights.

The file is read with `np.load(path, allow_pickle=False)`. For the
four-hidden-layer actor:

| Entry | Representation |
| --- | --- |
| `encoder.0.weight` through `encoder.8.weight` | Linear kernels in [out, in] layout |
| Matching encoder biases | [out] vectors |
| `decoder.0.weight` through `decoder.8.weight` | Linear kernels in [out, in] layout |
| Matching decoder biases | [out] vectors |
| `declaration` | Scalar Unicode array containing one JSON document |

Linear indices are 0, 2, 4, 6 and 8, with ELU between Linear layers. Each
exported weight is the transpose of the checkpoint's [in, out] kernel. The
encoder takes the goal rows and its output passes through tanh. The decoder
input is that latent, the 930-value proprioception history, then the six
anchor-orientation values. The checkpoint's shape marker `anchor_dec` and the
optimizer states are not exported.

The declaration holds `format` and `artifact_id`, then:

- `policy_source`: the source artifact id, the checkpoint label, the paths of
  the checkpoint and `config.json`, the completed outer updates and averaging updates with the source of each count,
  the selected weights, the configured averaging decay and warmup, and
  `run_end_markers`, the records of `DONE` and `terminal.json`. For a
  `torch_train.py` run, `initial_weights` gives the `--initial-checkpoint` path
  as `config.json` records it or, without one, the seed the networks were
  initialised with;
- `architecture`: layer widths, activation, tanh on the encoder output and
  float32 weights;
- `inputs`: five proprioception terms in order with their scale factors, a
  history of ten frames oldest first with the newest appended last and the
  first frame tiled at a reset with a zero previous action, the previous action
  as executed with its exploration noise, ten goal reference rows held at the
  clip end, and the first two columns of the relative rotation between the
  robot's and the reference's torso (the anchor body);
- `action`: joint order, default pose (`nominal_joint_position`, the
  knees-bent pose), scale `0.25 * effort / kp` (`kp` the joint position gain,
  `effort` the joint torque limit), setpoint limits, gains and torque limits,
  and the setpoint `clip(baseline + scale * action, setpoint_lower,
  setpoint_upper)`; the baseline is the default pose or, with
  `residual_action`, the next reference row's joint positions, so a policy
  trained with `residual_action` still needs the reference to execute;
- `timing`: control step, physics step, substeps and solver iterations;
- `training_domain`: the declared engine revision and capacities with the
  engine records of `config.json`, the model, the reference manifest's
  identity, clips and frame counts, the matrix-multiplication precision of
  training (`matmul_precision`), and `source_run`. `source_run` holds the run's
  arguments as `config.json` records them, with `git_commit` and `gpu`; a
  `train.py` run adds `git_dirty`, and a `torch_train.py` run adds its torch
  build and declared environment. No git revision is added that the run did
  not record.

The action values come from `env.build_model` compiled for the run's
configuration, the routine the training environment uses.
`checkpoint_evaluation_policy` is null because neither trainer's checkpoint
stores such a field.

Only an actor trained with `--anchor-to-decoder`, observation mode
`anchor_ori` (`hist_len` 10) and no `--actor-obs-norm` is exported. The command
checks finite float32 weights, the configured layer widths, the fixed scene
(`robot` `mjlab`, `contacts` `all`), the reference clips and their order, and
the 50 Hz reference rate. Runs trained on MuJoCo Warp, runs on a sharded motion
library (`--motion-lib`) and PPO policies are refused.

## Critic file

```bash
python -m nablamimic.export_bm_policy ... --out POLICY.npz \
    --critic-out CRITIC.npz --source-revision SOURCE_REVISION
```

The critic file holds `critic.{0,2,4}.weight/bias` in the same transposed
Linear layout, `normalizer.mean`, `normalizer.variance`, a scalar `format`
entry and a JSON declaration. The source revision is required and must equal
`git_commit` where `config.json` records one.

The critic's 825 inputs are proprioception (93), reference goal (680), anchor
orientation (6), root error (4), anchor-relative pose (9) and domain
randomisation draws (33). The normalizer applies
`clip((x - mean) / sqrt(variance + 1e-5), -10, 10)`. The checkpoint stores no
normalizer update count, and the declaration states that it is absent. Shapes,
finite float32 values and strictly positive variance are required.

## CPU tests

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_export*.py'
```

The tests build synthetic runs and cover the count rules, the selection of the
weights as trained or averaged, configuration and end-record refusals, the
action and critic layout, NumPy execution against the JAX and PyTorch networks,
output-path protection and the manifest's clip order. The comparisons with a
compiled model need the MuJoCo Menagerie checkout (the robot model files); the
`torch_train.py` tests need `torch` and are skipped without it. The tests cover
file layout and inference, not physical behaviour.
