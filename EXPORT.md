# Export a current policy, critic and reference cache

The exporters write files that an external executor can read using NumPy and
JSON. Loading the source checkpoint still requires this package's pinned
dependency environment because it contains optimizer state. Output files are
written once; existing destinations are refused.

The supported producer is the current upstream JAX `train.py`, on one device,
started fresh without resume or the SHAC guard. Only numbered
`RUN/ckpt/it_N.pkl` checkpoints are supported. `params.pkl` is saved at a
different point in the update loop and is not an export input. A run marked
`DIVERGED` or `HALTED` is refused. Torch checkpoints and configurations are
outside this command's schema.

## Current checkpoint counts and EMA

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
available. These tests do not establish a downstream executor's physical
behavior, nor do earlier exports validate the updated source automatically.
