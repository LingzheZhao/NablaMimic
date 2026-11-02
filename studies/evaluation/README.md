# Nominal evaluation studies

These programs evaluate a current NablaMimic checkpoint on all 40 LAFAN1 G1 clips.
They add bounded process scheduling, recorded body/capacity measurements, ordered
persistent evaluation and optional two-thread trajectory compression. They use the
caller-provided execution environment, GPU identifiers and ordinary input/output
paths. The core training and nominal-evaluation instructions remain in the
[repository README](../../README.md).

## Environment and inputs

Use the repository's pinned Python 3.12/Linux/CUDA environment and
[requirements lock](../../requirements-lock.txt). In particular, use the CUDA12
Warp build and the exact patched MuJoCo Warp installed by
[setup_env.sh](../../scripts/setup_env.sh) and
[setup_mujoco_warp.sh](../../scripts/setup_mujoco_warp.sh). Capacity instrumentation
uses that SDK's native counters and rejects an unexpected interface; substituting
another MuJoCo Warp revision is not a validated configuration. NVIDIA drivers and
`nvidia-smi` are required for GPU execution. Unit tests below run on CPU.

Install this checkout's `nablamimic` package in that environment. Provide the
Menagerie asset, 40 licensed LAFAN1 CSV inputs and prepared 50Hz motion cache using
`MENAGERIE_DIR`, `LAFAN1_G1_DIR` and `MOTION_CACHE_DIR`. The repository's
`python -m nablamimic.motion` command builds the matching cache; prepare it before
running concurrent evaluators. Keep that dedicated cache to the 40 current NPZ
files. Datasets and trained weights are not included in these studies.

Use a run whose current `config.json` declares `clips: [all]` and whose numbered
checkpoints contain the current policy metadata. EMA evaluation additionally
requires the saved EMA weights and update count. Do not treat a weight-only file
or an older checkpoint as this format. Output roots must be fresh and outside the
source, model and input directories.

## Independent checkpoint batch

From the repository root, set the declared committed revision and actual paths:

```bash
export NABLA_SOURCE_DIR="$PWD"
export NABLA_SOURCE_REVISION=YOUR_COMMITTED_REVISION
export NABLA_EVALUATION_REVISION="$NABLA_SOURCE_REVISION"
export PYTHONPATH="$NABLA_SOURCE_DIR/src"
export MENAGERIE_DIR=/absolute/path/to/mujoco_menagerie
export LAFAN1_G1_DIR=/absolute/path/to/lafan1_g1
export MOTION_CACHE_DIR=/absolute/path/to/reference-cache

python studies/evaluation/eval_batch.py \
  --source "$NABLA_SOURCE_DIR" --source-revision "$NABLA_SOURCE_REVISION" \
  --run /absolute/path/to/current-run --iters 250 500 --policies ema \
  --reference-cache "$MOTION_CACHE_DIR" \
  --workers 2 --workers-per-gpu 2 --devices GPU-ASSIGNED-UUID \
  --memory-fraction 0.4 --seconds 900 \
  --out /absolute/path/to/new-evaluation-output
```

Replace 250/500 with existing numbered saves. GPU indices or UUIDs are resolved
and checked before launch. Sharing a GPU requires explicit `--workers-per-gpu`;
the default is one worker per GPU. Each process has a private writable cache and
short compiler temporary directory. The stated JAX memory fraction is per worker;
its per-GPU sum cannot exceed one and preallocation is disabled. Warp/driver
allocations are additional. More concurrent workers are not a speedup guarantee.

Each condition runs the original scalar evaluator and then an independent observed
evaluator in fresh processes. A failed original evaluation does not suppress the
observed evaluation. The parent records every condition, assigned worker, command,
return code, time budget and interruption status in `manifest.json`. Declared image
identity may optionally be supplied through `NABLA_VERIFIED_IMAGE_REFERENCE`;
provenance is not an independent environment verification.

The batch runner sorts unique requested checkpoints. It does not represent repeated
A-B-A occurrences or maintain a persistent physics session. Those are separate
entry points below.

## Ordered persistent sessions

The explicit sequence wrappers preserve order and repeated checkpoints, creating
separate occurrence directories. One process owns one serial physical session;
original and observed evaluations use different processes and physics instances.
Set the same source/data environment as above, and configure a writable cache for
the bootstrap wrapper:

```bash
export WARP_CACHE_PATH=/absolute/path/to/evaluation-cache/warp
export JAX_COMPILATION_CACHE_DIR=/absolute/path/to/evaluation-cache/jax
export NABLA_ENGINE_CACHE_AUDIT=/absolute/path/to/evaluation-cache/bootstrap-original.json
mkdir -p "$WARP_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"

python studies/evaluation/bootstrap.py --script \
  studies/evaluation/persistent_original_evaluate.py \
  --run /absolute/path/to/current-run --iters 250 500 250 --policy ema \
  --forward-only --out /absolute/path/to/new-original-sequence

export NABLA_ENGINE_CACHE_AUDIT=/absolute/path/to/evaluation-cache/bootstrap-observed.json

python studies/evaluation/bootstrap.py --script \
  studies/evaluation/instrumentation/persistent_observed_evaluate.py \
  --run /absolute/path/to/current-run --iters 250 500 250 --policy ema \
  --forward-only --writer-workers 2 \
  --out /absolute/path/to/new-observed-sequence
```

The example retains separate bootstrap audit files. Use distinct cache directories
for simultaneous processes. The bootstrap
retains optional Engine-audit plumbing; these nominal MuJoCo Warp evaluations do
not require its training-only branch. The direct wrappers can also run in an
already configured environment.

Persistent evaluation restores the complete native initial snapshot only between
independent conditions. Sticky physics flags are never cleared within a trajectory.
The first rollout can include lazy CUDA graph capture. `--forward-only` skips
unused backward initialization and remains opt-in. These implementation choices
have not established strict floating-point equivalence to every fresh-process
execution; repeated GPU execution can differ. Keep numerical comparisons separate
from timing or successful native-state restoration.

`--writer-workers 2` permits at most two owned per-clip compression payloads in
flight, reserving a slot before allocating the next payload. Writers drain before
the next physical condition. Its default is one writer. The fresh observed wrapper
also accepts this option before its `--` separator. The checkpoint batch uses that
wrapper's default.

## Reading outputs

The observed evaluator preserves 40 clip traces, native aggregates and capacity
arrays, along with scalar returns, body errors, termination/support information and
an explicit engineering gate. The gate is a declared study rule rather than the
paper's return threshold. Its mathematical definitions and margins are encoded in
[observed_helpers.py](instrumentation/observed_helpers.py); native-capacity sampling
is in [evaluation_capacity.py](instrumentation/evaluation_capacity.py).

Original scalar results and observed traces belong to separate executions. Neither
transfers physics validity to the other. Partial, terminated and failed conditions
remain visible, and capacity-invalid descendants remain excluded from valid metric
support. A completed process or reference horizon alone does not establish fidelity,
contact/flight behavior, numerical equivalence or a training-speed reproduction.
The original donor normalized return retains its defined missing-tail zero fill;
physical errors are reported on their actual recorded support.

These scripts are research entry points, not a portable benchmark claim. Comparing
speed requires the same checkpoints, support, initialization mode, cache conditions,
hardware and a protocol fixed before execution. The study data and outputs belong
outside the repository.

## CPU checks

The study tests use fresh synthetic observation arrays and short CPU subprocesses;
they exercise process ownership, cancellation, explicit GPU assignment, memory-limit
configuration, bounded writer payloads and preservation of invalid/partial support.
Run them in the pinned environment from the repository root:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover \
  -s studies/evaluation/tests -v
```

The test directory must remain beneath `studies/evaluation` so its relative imports
resolve the scheduler and instrumentation modules. These 21 CPU checks do not replace
provisioned GPU simulation or numerical validation of the pinned SDK.
