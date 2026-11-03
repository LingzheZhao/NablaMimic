# Frozen Torch compilation and precision diagnosis

This study diagnoses the current Torch implementation at fixed weights and inputs.
It does not attribute the final return difference to a compiler, precision mode,
reset implementation or random stream without a measured intervention. It does
not alter training, tolerances, simulator options or existing results.

Run the program separately for the final JAX EMA checkpoint and final Torch EMA
checkpoint. Each invocation records its checkpoint path, completed/EMA update
counts and saved elapsed time. These are separate readings; do not average them.
The checkpoint's critic and normalizer stay frozen alongside its selected actor.

Each invocation evaluates six conditions: eager, fullgraph Inductor, and explicit
CUDA Graph replay, each with FP32 matmul and with production TF32 permission.
cuDNN TF32 stays disabled. Default FMA contraction stays enabled. The protocol
uses 40 clips, batch 40 and horizon 4. Two fixed cases choose frame offsets 25 and
75 (bounded by clip length) and separate reset candidates at offsets 100 and 150.
All randomized reset/observation/action draws are materialized once with seed 0,
saved, and shared byte-for-byte across the six conditions. No reset sampling or
optimizer update runs inside the comparison. Each graph replays both cases.

Three measurements separate likely mechanisms:

- Full actor output and VJPs for the same fixed output cotangent, with gradients
  for history, goal and every trainable parameter. A separate BF16-to-FP32
  roundtrip of those same observations measures regression-storage sensitivity;
  it is not silently substituted for the full-precision observation condition.
- The actual current four-step differentiable window, including native physics,
  action-cotangent clipping and action targets. Report actual term/done/frame
  support, retained rewards/critic observations, loss, dL/da and clipping readings.
- One native step with an action frozen from eager FP32, followed by negative
  mean `body_ori` reward and its action gradient. Report raw and clipped action
  gradients, clipping support and all 29 action-column comparisons. Indices 21,
  27 and 28 are highlighted without dropping other joints. This fixed-action
  comparison separates derivative execution from upstream policy differences.

A small accompanying orientation-primitive check includes exact positive and
negative zero for the custom real-absolute VJP, and identity, near-identity and
exact-pi quaternions. It records the intended positive absolute derivative at
zero separately from pairwise numerical comparisons. A boundary failure is not
assumed to explain the observed wrist error without support in actual states.

The comparator uses the existing rtol 1e-4 and atol 1e-5, finite values, and exact
integer/Boolean support. It also reports bitwise agreement, maximum absolute
difference, relative L2 error, both vector norms, cosine and sign disagreements.
These additional quantities are descriptive; they do not relax the frozen gate.
Failed parity leaves remain failed. A completed diagnostic may have
`parity_passed: false`; runtime errors, nonfinite results or native invalid
counters cause a failed process. All raw arrays and partial output stay visible.

Use the pinned provisioned Torch overlay and corrected 96-contact/416-row Engine
source. The program checks Engine origin and capacities, uses private empty
native/compiler caches and a short temporary directory, and requires exactly one
CMP 170HX. There is no CPU/eager fallback and no local GPU execution. The parent
job owns a bounded wall-clock timeout and cancellation. No other evaluation or
training shares this GPU allocation.

Example (run twice with explicit distinct checkpoint/output paths):

```sh
python compile_precision_check.py \
  --source-root "$NABLA_SOURCE_DIR" --source-revision "$NABLA_SOURCE_REVISION" \
  --engine-root "$NABLA_ENGINE_ROOT" --engine-revision "$NABLA_ENGINE_REVISION" \
  --reference-cache "$MOTION_CACHE_DIR" --checkpoint "$CHECKPOINT" \
  --policy ema --out "$FRESH_OUTPUT"
```

The source must include the corrected materialized collector-history boundary;
this study performs no collector mutation. Old FP32 eager-versus-graph failures
and the separately passed compiled-versus-graph capture check retain their
original status. This study is a new frozen-weight condition.
