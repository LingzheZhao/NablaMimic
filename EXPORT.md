# Exporting a policy and its reference for other executors

Two commands write files that a program without this package can read with NumPy
and JSON alone: the actor of a training checkpoint as a policy file, and a
manifest of the reference cache the run trained on. A checkpoint
(`ckpt/it_N.pkl`) pickles optimizer states and therefore loads only in this
package's environment; the exported files do not. Both commands run in that
environment, write their output once, and refuse an existing output path.

## Reference manifest

```bash
python -m nablamimic.export_reference_manifest --cache-dir CACHE --clips all \
    --artifact-id REFERENCE_ID --out MANIFEST.json
```

`CACHE` is the directory of processed clips that `nablamimic.motion` writes
(`MOTION_CACHE_DIR`). `--clips all` lists every LAFAN1 clip recorded in
`scripts/lafan1_g1.sha256`, in sorted name order, which is the order
`--clips all` gives the training library; explicit names keep the order given,
as `train.py` does. The command refuses a clip without a recorded dataset file,
a missing cache file, arrays other than the six of `motion.CACHE_KEYS`, shapes
that disagree with the 14 tracked bodies or across clips, and non-finite values.

The manifest is JSON. It states the format (`nablamimic-reference-manifest-v1`),
the artifact identity given, the cache directory, the frame rate, the tracked
bodies in their array order, each array's dtype, trailing shape and meaning,
the clip selection and order rule, the dataset repository and revision that
`scripts/download_lafan1.sh` downloads, and per clip its name, cache file name,
frame count, first and last row in the concatenated library, and source file.
The cache directory is relative to the manifest when the cache lies inside the
manifest's directory and absolute otherwise; a reader opens
`<manifest directory>/<cache_directory>/<cache_file>`. The cache file names
carry a tag that `motion.cache_file_name` computes from the body list; a reader
takes the names from the manifest and does not compute the tag.

Build the manifest from the cache the run itself read (its `MOTION_CACHE_DIR`).
When that directory is a copy of a published cache, the manifest describes the
published cache only if the copy is complete; compare the two directories' file
names and sizes before publishing the manifest under the published cache's
identity.

## Policy file

```bash
python -m nablamimic.export_bm_policy --checkpoint RUN/ckpt/it_N.pkl --run-config RUN \
    --policy live --artifact-id POLICY_ID --source-artifact-id CHECKPOINT_ID \
    --reference-manifest MANIFEST.json --engine-revision ENGINE_REVISION \
    --engine-contact-capacity 64 --engine-constraint-row-capacity 288 --out POLICY.npz
```

`--run-config` takes the run directory or its `config.json`; the run directory
is the directory of `config.json`, and the checkpoint must lie in its `ckpt/`.

The output is an `.npz` in the format `bm-nabla-tracking-policy-v1`, readable
with `np.load(path, allow_pickle=False)`. For the trained actor, with four
hidden layers per network, it holds 21 entries:

| Key | Shape | Content |
| --- | --- | --- |
| `encoder.0.weight` ... `encoder.8.weight` | [out, in] | goal encoder Linear weights, 680 -> 2048, 1024, 512, 256 -> 32 |
| `encoder.0.bias` ... `encoder.8.bias` | [out] | its biases |
| `decoder.0.weight` ... `decoder.8.weight` | [out, in] | decoder Linear weights, 32 + 930 + 6 -> 2048, 1024, 512, 256 -> 29 |
| `decoder.0.bias` ... `decoder.8.bias` | [out] | its biases |
| `declaration` | 0-d unicode | one JSON document |

All weights are float32. Layer `i` of the checkpoint's `enc` list, a pair
`(w, b)` with `w` of shape [in, out] applied as `x @ w + b`, becomes
`encoder.{2i}.weight = w.T` and `encoder.{2i}.bias = b`; `dec` maps to
`decoder.*` the same way. The indices 0, 2, 4, 6 and 8 are the Linear layers of a
flat sequence with ELU between them and none after the last. The encoder output
passes through tanh, and the decoder input is the concatenation of that latent,
the proprioception history and the anchor orientation (`nets.actor_apply` with
`--anchor-to-decoder`). The `anchor_dec` shape marker, the critic, its
normaliser, optimizer states and reset-bin statistics are not exported.

`--policy` selects weights through `params.evaluation_actor`: `live` is the
training actor and `ema` the averaged one, refused when the checkpoint holds no
consistent EMA. Released-recipe checkpoints hold live weights only.

The declaration has eight sections. `policy_source` names the kind
(`imported_nablamimic`), the source artifact identity, the completed outer
updates and the averaging updates, how the update count was obtained (the
checkpoint's `completed_updates`, or for checkpoints without it the trainer's
rule `env_steps / (num_envs x devices x horizon)`, which must also equal the
file label), the selected weights (`live` or `averaged`), and the end markers
found in the run directory (`run_end_markers`: `DONE` with its last update,
`HALTED` with the guard's record, or neither while a run is unfinished). `architecture`
gives the widths, activation, latent squash and decoder input order. `inputs`
names each proprioception and goal term with its width and factor, the
ten-frame history (frame-major, oldest first), the previous action
(the executed action including exploration noise), the ten goal rows with the
last row held past the clip end, the torso anchor orientation as the first two
columns of the relative rotation, and no normalisation. `action` gives the joint
names in actuator order, the nominal pose, the per-joint scale
`0.25 x effort / kp`, the gains and torque limits, the baseline (the next
reference row's joint positions for `--residual-action`, the nominal pose
otherwise), the setpoint range, and that the encoder bias never enters the
setpoint. These values come from the model `env.build_model` compiles for the
run's configuration, cast to float32 as the environment uses them. `timing`
gives the 0.02 s control step, the 0.005 s physics step, four substeps and the
run's solver iterations. `training_domain` records the engine (the declared
revision and capacities given on the command line, which `config.json` does not
contain, and the identities the engine reported at training), the robot model,
the reference (the manifest's identity, selection and clips, which must agree
with the run's `--clips`), the training matmul precision and the run's
arguments. A run started with `--config` records `argv` (the arguments used,
without `--config`, `--out` and `--resume`), `config_args` and `config_file`,
and the declaration copies them. A run started without `--config` records none
of the three; the declaration then says so in `arguments_source` and holds
`parsed_arguments`, every top-level field of `config.json` except the fixed
settings and bookkeeping that `train.py` adds (`NON_ARGUMENT_FIELDS`), instead
of an empty `argv`.

`train.py` ends a run with a marker in its directory. `DIVERGED` (a non-finite
loss, checked on every tenth update) is refused: the checkpoint at that label
holds the weights that produced the non-finite loss, and the nine updates before
it were not checked. `HALTED` (the `--shac-strict` guard) is recorded rather
than refused: the guard masks the rejected actor and optimizer update before the
halt save, so the checkpoint written at the halt holds the weights after its
label's completed updates. Only the actor-input statistics advance at the halt,
and actors with them are refused. A run resumed after a halt can hold both
`HALTED` and `DONE`; both are recorded.

Runs of `torch_train.py` (branch `codex/pytorch-nablaengine`), whose
`config.json` declares `training_framework` `torch`, keep the same checkpoint
keys but other run records: `DONE` holds `completed` (recorded as its content),
and `config.json` holds the parsed arguments, among them `config` (the file the
run started with), without `argv`, `config_args`, `config_file` or `git_dirty`.
The declaration then gives `arguments_source`
`parsed_arguments_in_config_json_torch_train_records_no_argv`, the parsed
arguments without the trainer's bookkeeping (`TORCH_NON_ARGUMENT_FIELDS`) and
`git_dirty` `not_recorded_by_torch_train`; `source_run.training_framework` is
`jax` for `train.py` runs and `torch` for these.

The command refuses rather than fills in: a checkpoint other than a numbered
`it_N.pkl`, a label or update count that disagrees with the checkpoint, an actor
without the anchor-to-decoder structure or with observation normalisation,
kernel shapes that differ from the configured widths, non-finite or non-float32
parameters, runs trained on MuJoCo Warp or a sharded motion library, a robot
other than `mjlab` or contacts other than `all` (the action values come from
`env.build_model`, which compiles that fixed scene only), a missing
configuration field, a checkpoint outside the run directory's `ckpt/`, a run
directory holding `DIVERGED`, and a manifest whose bodies, frame rate or clips
differ from the run's.
