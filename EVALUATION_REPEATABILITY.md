# Evaluation execution repeatability

Identical weights produce different full-reference returns even when one JAX
executable and its captured MuJoCo Warp graph are reused. A separate observed
execution first diverges at the native physics output, before observations or
policy actions differ. Fixed-input native and center-of-mass repetitions further
locate variation in an observed component graph at a tree-level atomic
accumulation stage. These results narrow the investigation; they do not uniquely
explain the Torch/JAX live-return gap or establish a defect in Torch training.

## Fixed conditions and state restoration

The sole policy is the original Torch fit's live checkpoint at 1,500 updates,
from [the fixed training revision](https://github.com/LingzheZhao/NablaMimic/commit/b13375e22948d505579a627e8fc1bce99cccb31d).
It is distinct from the later TF32 repeat and network-matmul-highest fit. Evaluation
uses [the official evaluator](https://github.com/OamIcs/NablaMimic/commit/bfbb2a6773b4776c17402a34a93ec7589d9c7c18):
JAX highest network matmul, the pinned MuJoCo Warp build with the existing backward
patches, Warp 1.18, ten solver iterations, deterministic mean actions, no noise
or randomization, all forty LAFAN1-G1 clips and the original 500-control reward
bins. The longest scan has 13,654 controls at 50 Hz. Evaluation uses MuJoCo Warp;
NablaEngine is the separately measured training backend.

One exclusive RTX 6000 Ada 48 GB runs a single process with default JAX BFC
fraction .4 and preallocation disabled. The original constructor and its warmup
are retained. Before compilation, the diagnostic saves 586 reachable Model/Data
arrays, totaling 54,833,971 logical bytes, including existing gradients and
aliases. Every repetition restores them in place and verifies their bytes,
addresses and root identities. All 31 actor/start/key/reference argument leaves
also keep identical bytes and storage. Unused scratch bytes are preserved,
including nonfinite bit patterns; there is no blanket zeroing.

Each stage lowers and compiles exactly once, then performs one complete priming
run to finish lazy Warp capture. All outputs and the Warp device are synchronized
before restoration. Three measured calls reuse the same compiled/runtime objects.
Host inspection additionally verifies unchanged forward-call descriptors,
capture objects, graph execution handles and staging storage. It permits only
the SDK's supported updates to caller memcpy endpoint addresses.

This proves reuse of the observed executables and restoration of public arrays.
It does not restore private solver/collision workspaces or every opaque captured
allocation. No deterministic-kernel flag, reduction replacement or solver setting
is introduced.

## Original and observed evaluator repetitions

Stage A uses the unchanged original rollout. Stage B is a separately compiled
rollout with additional observation outputs. Its reversible AST checks retain
the original scan, carry, rewards, termination, binning and continued stepping
of inactive rows. The Python step wrapper computes the existing control
expression once and exposes the same six FFI inputs and three raw outputs,
alongside the original postprocessing. It does not change the native FFI ABI
or the four physical substeps. Raw physics qpos remains distinct from qpos after
MJX kinematics/normalization.

| Executable | Repetition | Full-reference return | References without recorded termination |
| --- | ---: | ---: | ---: |
| Original | 1 | .7330382703 | 34/40 |
| Original | 2 | .7421763592 | 35/40 |
| Original | 3 | .7331697829 | 35/40 |
| Observed | 1 | .7424479182 | 35/40 |
| Observed | 2 | .7196062739 | 33/40 |
| Observed | 3 | .7379073332 | 34/40 |

The original readings span .00913809 with fixed compilation and capture. Thus
separate compilation cannot account for all evaluation variation. The observed
readings span .02284164. All three pairs within each stage fail the unchanged
strict comparison, with differing execution supports. All nine original/observed
aggregate comparisons also fail. Instrumentation can change fusion, storage or
scheduling, so these groups remain separate; B's first divergence cannot be
assigned to an individual A execution.

All summaries reconstruct exactly from the unrounded saved reductions. No
failed or skipped measured condition is removed. Each trace retains 21 fields
without masking inactive rows, totaling 4,237,109,280 bytes (3.946 GiB), and is
written before the next repetition. The GPU worker's peak host RSS is 11.71 GiB; the maximum
phase-boundary device-memory sample is 5.96 GiB. Those device samples are not
continuous peak telemetry. Three repetitions establish no confidence interval,
quality distribution or uncertainty bound for other checkpoints.

## First divergence in the observed executable

For observed repetitions 1 and 2, the first control has identical observations,
policy actions and all six physical inputs across the entire forty-world batch.
The first raw-output bit difference is in `fallAndGetUp1_subject5`, root x:
.0010265400633215904 versus .0010265391319990158 m, a difference of
9.313225746e-10 m. This is the return after four physical substeps, not a
measurement of the first internal substep.

At that same control, warmstart component 13 differs by 4.076957703e-5 in its
native acceleration units and exceeds the frozen rtol 1e-4 / atol 1e-5 check.
Observations and actions first differ at the second control. The first supported
termination difference appears at control 370 (7.40 s) in
`fallAndGetUp1_subject4`; alive support separates from control 371. These are
ordered observations of this pair, not a numerical amplification factor or proof
that one initial component alone causes a later termination in another row.

All three observed pairs first differ in raw physics outputs at control one,
with identical whole-batch inputs there. Bitwise and tolerance events remain
separate: repetitions 1/3 first exceed tolerance at control two, where their
inputs have already diverged. Their first unmasked termination difference at
control 608 belongs to an already inactive row; the first termination difference
on common alive support is control 4,032. The unmasked and supported records
are both retained. All recorded observed fields are finite, including inactive
rows. Height/tilt labels remain reference-tracking criteria, not force-verified
mechanical falls.

## Conditional native and center-of-mass repetitions

The frozen repetitions-1/2 trigger qualifies because the first split occurs at
control zero in zero-based indexing and every physical input is bitwise equal.
A separate process on another RTX 6000 Ada of the same model reads those actual
forty-world inputs and the public native snapshot. It does not resample a start,
reduce the batch or reconstruct a later state. Its original physics function
is compiled once, primed once and measured fifteen times, with public-state,
argument and captured-graph checks between calls.

Compared with the first measured call, all fourteen subsequent native calls
have bit differences in qpos, qvel and warmstart. qpos/qvel remain within the
fixed tolerance; thirteen warmstart comparisons exceed it. Maximum absolute
component differences across those repetitions are respectively 1.19209e-7,
2.38419e-6 and 1.08337e-3 in their native mixed-coordinate units. The new process's
first qpos/qvel readings are within tolerance of the two original observed
readings, but not bitwise equal; warmstart fails both cross-process comparisons.
This is a new fixed-input repeatability experiment, not an exact replay of B's
hidden state or outputs.

Four restored calls of the original kinematics produce bitwise-identical values
for all seven recorded kinematic fields. One such set of fields, with the same
model constants, is then frozen for sixteen repetitions of the original captured
`com_pos`. Ten of the
fifteen subsequent calls differ from the first in subtree COM, inertia and
motion-axis outputs, although all are within tolerance. The largest final
subtree-COM difference is 4.65661e-10 m. Variation therefore exists in the original
component before adding stage-observation copies.

A separate graph adds copies after initialization and each original tree-depth
kernel. It preserves the original kernels, launch shapes and tree order. Eleven
of fifteen subsequent repetitions first differ after tree depth 2, where child
bodies 2, 8 and 14 all accumulate into parent body 1; all preceding recorded
stages are bitwise identical. This is the
[pinned `_subtree_com_acc` kernel](https://github.com/etaoxing/mujoco_warp/blob/4af093f02a3f48f0009ca7c916674b928136f310/mujoco_warp/_src/smooth.py#L700-L714),
which uses floating-point atomic addition. At this stage the buffer contains
mass-weighted position sums, not yet COM positions in metres. The largest stage
difference is 1.90735e-6 in mass-times-length units. Its ratio to a final COM or
native-output maximum is not an amplification factor. Final outputs of all sixteen
observed component calls are within tolerance of the original component baseline;
only three are bitwise equal.

This localizes the observed component graph's first numerical variation to the
sibling-accumulation stage and provides a measured example consistent with
[Warp's documented atomic-order nondeterminism](https://nvidia.github.io/warp/v1.18/user_guide/faq.html).
It does not prove that this kernel uniquely causes A's return variation, B's
first native-output difference, or the Torch/JAX live gap. Contact ordering,
other reductions and unobserved native workspaces remain outside that causal
claim. No atomic kernel or whole evaluator is rewritten.

## Validity flags and interpretation

The first diagnostic attempt stopped before any full-evaluator priming or measured evaluation:
its checker incorrectly treated every nonzero `overflow` bit as a capacity
failure. The pinned SDK defines 512 as a solver-iteration limit and 1024 as a
line-search-iteration limit. The corrected checker retains those diagnostic
bits and rejects capacity or unknown bits; it changes no physics state, solver
budget or numerical tolerance. The failed attempt and its full snapshot remain
preserved.

The successful A/B constructor snapshot already contains the line-search flag
in six worlds; all forty worlds have it by each repetition's end. Solver-
iteration-limit bits are zero, as are capacity/unknown bits throughout the
checked boundaries. C likewise retains line-search flags with no capacity or
unknown bits. Flags are never cleared, so their presence cannot count events
or prove convergence. The original rollout keeps stepping inactive rows: a
terminal flag may first appear after termination or the reference horizon ends.
Thus 40/40 final flags do not mean that all forty valid trajectories suffered a
solver failure. Finite fields and valid capacities do not certify exact physical
derivatives or a fully converged solve.

The bounded investigation is complete. It establishes runtime evaluation
variation with fixed compilation and localizes variation within an observed
physics component, while preserving uncertainty about its contribution to final
return. This evaluation study changes none of the training defaults for TF32,
compilation, CUDA Graph, EMA or the schedule.
Fine quality comparisons should repeat the corresponding fixed checkpoints in
an interleaved evaluation protocol before adding training seeds; the ranges
above must not be reused as error bars for the later TF32/highest fits.
