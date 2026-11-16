# Export a current policy, critic and reference cache

The exporters write files that an external executor can read using NumPy and
JSON. Loading the source checkpoint still requires this package's pinned
dependency environment because it contains optimizer state. Output files are
written once; existing destinations are refused.

The command reads runs of both trainers in this package: the JAX `train.py`
and the PyTorch `torch_train.py`. The run's `config.json` names the trainer: a
`torch_train.py` run records `training_framework` as `"torch"`, a `train.py`
run records no `training_framework`, and any other value is refused. Either
run must be on one device, without resume or the SHAC guard. Only numbered
`RUN/ckpt/it_N.pkl` checkpoints are read. `params.pkl` is saved at a
different point in the update loop and is not an export input. A run marked
`DIVERGED` or `HALTED` is refused. The sections below describe a `train.py`
run first; [PyTorch-trainer runs](#pytorch-trainer-runs) gives what differs
for a `torch_train.py` run.

## train.py checkpoint counts and EMA

The trainer saves numbered checkpoint N before starting update N, after N
completed outer updates. Its native `env_steps` must equal
`N * num_envs * devices * horizon`. The exporter requires exact agreement with
the filename and native `iter`; it does not add fields to the checkpoint.

The current schema stores no completed-update or averaging counter. When
`actor_ema > 0`, `actor_ema` must be present and a fresh run has performed N
averaging updates at checkpoint N. When averaging is disabled, that entry must
be absent. The declaration records these derived counts and the configured
decay and warmup. The upstream `ours.args` recipe uses decay 0.999 with warmup;
both its live and averaged weights can be exported explicitly.

Freshness is checked through the complete current configuration schema:
`resume` is null, there is one process and one device, and a configured run
has exactly one segment with no resume. Required and allowed checkpoint fields
follow the current producer, including the optional action-target optimizer and
averaged actor entries. Missing or unsupported fields are rejected, without a
format converter or alternate reader.

The required configuration fields are the explicit upstream 0.2 core schema.
Known optional CLI records may be present without becoming mandatory core
fields, and absent optional records are not filled with defaults. This keeps
the baseline schema usable when an optional feature is added to the parser.
Policy export still requires the explicit `obs_mode=anchor_ori` contract.

## PyTorch-trainer runs

A `torch_train.py` run is read by the records that trainer writes:

- `config.json` holds exactly the trainer's parsed arguments (its parser adds
  `--matmul-precision`, `--execution`, `--regression-batch`,
  `--initial-checkpoint`, `--save-wall-minutes` and `--full-state-at-end` to
  that of `train.py`), the fields of `train.FIXED` and the fields the trainer
  adds: `training_framework`, `git_commit`, `devices`, `processes`,
  `observed_precision`, `gpu`, `torch_version`, `torch_cuda`, `random_stream`,
  `clock`, `inputs`, `declared_environment` and `engine`. A missing field or a
  field outside that set is refused. The trainer records no `argv`,
  `config_args`, `config_file`, `segments` or `git_dirty`. The declaration's
  `source_run` holds the parsed arguments, the source revision, the GPU, the
  torch build and the declared environment, and no `git_dirty`.
- The run must be completed: `DONE` holds `completed`, `terminal.json` has
  status `completed`, and there is no `FAILED.json`. The trainer writes these
  at the end of a run that was not interrupted, which includes a run stopped
  by its `--hours` budget, so the terminal `completed_updates` can be below
  `iters`. The declaration records both files in `run_end_markers`, and a
  checkpoint label above the terminal `completed_updates` is refused.
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
  any other run. The declaration's `policy_source.initial_weights` gives the
  path of that file as `config.json` records it; a run started without one
  records the seed its networks were initialised with.
- The networks are saved as the same trees of `[in, out]` kernels as in a
  `train.py` checkpoint, so the policy and critic files have the same entries,
  shapes and orientation.
- The configuration check imports the parser of `torch_train.py`, so exporting
  such a run needs the optional `torch` dependency in addition to the JAX and
  MuJoCo packages that compiling the model needs.

The policy inputs and the action map of `torch_env.py` are those of `env.py`,
so a declaration states them in the same terms for both trainers: the five
proprioception terms in the same order and scaling, a history of ten frames
oldest first with the newest appended last and the first frame tiled at a
reset with a zero previous action, the previous action as executed with its
exploration noise, ten goal rows held at the clip end, the first two columns of
the anchor's relative rotation, and setpoints
`clip(baseline + 0.25 * effort / kp * action, setpoint_lower, setpoint_upper)`
around the knees-bent default pose (`nominal_joint_position`) or, with
`residual_action`, the next reference row's joint positions.
The critic inputs, the normalizer and its update are also the same.

## Reference manifest

```bash
python -m nablamimic.export_reference_manifest --cache-dir CACHE --clips all \
    --artifact-id REFERENCE_ID --out MANIFEST.json
```

Use the processed cache the run actually read. `--clips all` uses the recorded
LAFAN1 dataset's sorted clip order; explicitly listed names retain their given
order. The command checks the six current cache arrays, their shapes, body
ordering and finite values. The manifest states the dataset identity, frame
rate, body names, array conventions, clip order, exact cache filenames and row
intervals in the concatenated library. Readers use these filenames directly.

The manifest's cache directory is relative when it lies within the manifest's
directory and absolute otherwise. A reader opens
`<manifest directory>/<cache_directory>/<cache_file>`. The source cache remains
unchanged.

## Policy file

```bash
python -m nablamimic.export_bm_policy --checkpoint RUN/ckpt/it_N.pkl --run-config RUN \
    --policy ema --artifact-id POLICY_ID --source-artifact-id CHECKPOINT_ID \
    --reference-manifest MANIFEST.json --engine-revision ENGINE_REVISION \
    --engine-contact-capacity CONTACTS --engine-constraint-row-capacity ROWS \
    --out POLICY.npz
```

`--run-config` also accepts the run's `config.json`. The checkpoint must be in
that run's `ckpt/` directory. The Engine revision and capacities are explicit
declarations about the build used for training, not values inferred from weights.
`--policy live` selects `actor`; `--policy ema` selects `actor_ema` and
fails if averaged weights are unavailable.

The output format is `bm-nabla-tracking-policy-v1`, readable with
`np.load(path, allow_pickle=False)`. For the standard four-hidden-layer actor:

| Entry | Representation |
| --- | --- |
| `encoder.0.weight` through `encoder.8.weight` | Linear kernels in [out, in] layout |
| Matching encoder biases | [out] vectors |
| `decoder.0.weight` through `decoder.8.weight` | Linear kernels in [out, in] layout |
| Matching decoder biases | [out] vectors |
| `declaration` | Scalar Unicode array containing one JSON document |

Linear indices are 0, 2, 4, 6 and 8, with ELU between Linear layers. Each exported
weight is the transpose of the native [in, out] kernel. The encoder output uses
tanh. Decoder input order is latent, 930-value proprioceptive history, then six
anchor-orientation values. The shape marker and optimizer state are not exported.

The declaration records checkpoint identity and derived counts, explicit weight
selection, architecture, observation layout, noise-related input meaning, action
joint order, nominal pose, scale, setpoint limits, timing, training configuration
and reference identity. Action parameters come from the same model-building
routine the environment uses. Live residual targets remain the next reference
row's joint positions, not the nominal pose; exporting a policy does not remove
that external reference dependency.

Only the current anchor-orientation actor with anchor-to-decoder structure and
no actor observation normalization is supported. Finite float32 weights,
configured layer widths, the fixed mjlab/all-contact model, reference clip order
and 50 Hz reference rate are checked. MuJoCo Warp training, sharded motion
libraries and decoder position-feedback variants are outside this export
contract.

The exported declaration's `checkpoint_evaluation_policy` is null because
the native current checkpoint stores no such field. The selected weights and
configured averaging rule are stated separately. Run arguments and native source
metadata are retained as recorded; the exporter does not invent a git revision.

## Optional critic file

```bash
python -m nablamimic.export_bm_policy ... --out POLICY.npz \
    --critic-out CRITIC.npz --source-revision SOURCE_REVISION
```

The second file uses `bm-nabla-critic-v1`. It holds
`critic.{0,2,4}.weight/bias` in the same transposed Linear layout,
`normalizer.mean`, `normalizer.variance`, a scalar `format` entry and a JSON
declaration. The source revision is required and must agree with a recorded
non-null `git_commit`.

The critic's 825 inputs are proprioception (93), reference goal (680), anchor
orientation (6), root error (4), anchor-relative pose (9), and domain draws (33).
The normalizer applies
`clip((x - mean) / sqrt(variance + 1e-5), -10, 10)`. Its stored mean and variance
have no stored update count; the declaration states that absence. Shapes, finite
float32 values and strictly positive variance are required.

## CPU validation

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_export*.py'
```

Fresh current-schema inputs test native count semantics, live/EMA selection,
configuration and marker rejection, action/critic layout, NumPy execution against
the actual JAX networks, output-path protection and reference manifest order.
The compiled-model comparison runs when the current Menagerie checkout is
available. The `torch_train.py` run tests need the optional `torch`
dependency and are skipped without it. These tests do not establish a
downstream executor's physical behavior, nor do earlier exports validate the
updated source automatically.

The pinned provisioned CPU suite passed all 78 tests with no skips. An integration
run then used the actual official seed-zero checkpoint after 1,500 updates and
its processed reference cache. All four CLI/verification stages and all 142
checks passed: forty clips, 441,120 frames, fourteen bodies, twenty-nine actions,
current checkpoint counts, both live/EMA policy exports and the same checkpoint
critic. Exported kernels, normalizer arrays and model-derived action parameters
matched exactly. NumPy actor/critic inference on fresh deterministic inputs
satisfied rtol 1e-5 and atol 1e-6 against JAX for both policies. These measurements
validate export layout and inference only; no downstream physical execution was
performed by this integration check.
