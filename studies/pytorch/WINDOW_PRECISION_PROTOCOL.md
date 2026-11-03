# Frozen full-window precision comparison

One invocation reads one current live checkpoint and its saved critic and critic
normalizer. Run separately for update zero and update 1,500; their readings stay
separate. This is a frozen-weight diagnostic, with no regression, optimizer, EMA
or adaptive-sampler update. It cannot establish the cause of a three-hour return
gap or estimate variation across training seeds.

Both conditions use the complete production actor, 4,096 environments, a
64-control-step window, the same 96-contact/416-row Engine, four native substeps
and ten solver iterations. Both compile the real control body, encoder and
action-target function and capture the complete differentiable window in an
explicit CUDA Graph. The sole condition is permission to use TF32 in CUDA matrix
multiplication, set before compilation. cuDNN TF32 stays disabled, default FMA
contraction stays enabled, and compilation failures never select an eager path.

Generate randomized reset starts, random initial episode ages, reset states,
action noise and both observation-noise streams exactly once with Torch seed
zero. The actual materialized input tensors, including full physical/reset
state and warmstart, are saved and supplied to both conditions. Each condition
owns a private copy through the graph-input contract. Capture setup restores the
saved CPU/CUDA RNG states, and measured execution must not advance them. These reset-start inputs
deliberately do not represent a replay-restored intermediate training state;
intermediate weight checks isolate weight-dependent window sensitivity on the
same controlled distribution. The current checkpoint does not supply the
intermediate replay/collector state needed to claim a training continuation.

Run FP32 then TF32 sequentially on one provisioned GPU. Each uses private compiler
caches. Only one precision graph remains live at a time; export each selected
array to CPU, release the graph/callables, reset Dynamo and clear unused CUDA
allocations before creating the next graph. Report memory before proceeding.
Do not create a second simultaneous trainer. Keep immutable code, environment,
reference files and model identities explicit in the output provenance.

Save per-step noiseless policy action and action used by physics, raw loss-action
cotangent, clipped ascent direction, action target, live-row and clipping masks,
clipping scales, term/done/frame supports, reward terms and bootstrap values.
The raw cotangent is the derivative with respect to the zero delta inserted
after the production in-graph cotangent clip; the target direction is its
sanitized, second-stage clipped negative. Retain the original in-graph aggregate
clipping statistics separately. All twenty-nine action columns receive norm,
relative-L2, cosine, sign-change and threshold-mismatch summaries; no joint is
selected after observing the results. Indices 21, 27 and 28 are highlighted only
because the prior independent final-policy audit identified these wrist errors.

Use the existing numerical margins, absolute tolerance 1e-5 and relative
tolerance 1e-4. Integer/Boolean supports and clipping masks must match exactly.
All numerical outputs must be finite. Keep all nine native invalid-state guards
at zero, including nonfinite qpos/qvel/ctrl rows, and do not count an invalid
window as usable evidence. Preserve failures, first mismatch coordinates,
per-step summaries and per-joint summaries; do not relax the threshold. The
earliest mismatch is a descriptive location, not proof that it caused every
subsequent mismatch. A strict numerical failure still provides retained
diagnostic arrays, but the process exits nonzero and the result remains failed.

An early wrist cotangent/target discrepancy would support a narrowly specified
follow-up training precision comparison. Near-equivalent tested windows would
weaken an immediate large-gradient-distortion account on these inputs, while
leaving cumulative, late-training and different-state effects unresolved.
Torch FP32-versus-TF32 sensitivity alone does not prove that the observed
cross-framework TF32 difference caused the JAX-versus-Torch learning gap.

Example invocation inside the provisioned environment:

```bash
python window_precision_check.py \
  --source-root "$NABLA_SOURCE_DIR" --source-revision "$NABLA_SOURCE_REVISION" \
  --engine-root "$NABLA_ENGINE_SOURCE_DIR" --engine-revision "$NABLA_ENGINE_CAPACITY_REVISION" \
  --reference-cache "$MOTION_CACHE_DIR" \
  --checkpoint /path/to/it_001500.pkl --expected-updates 1500 \
  --out /fresh/output/window-live-1500
```
