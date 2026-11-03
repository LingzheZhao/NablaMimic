The current-paper recipe improves reference tracking in our single-seed reproduction:
at 1,500 outer updates, full-clip normalized return rises from **0.724010 to 0.837293**;
after three hours, it rises from **0.860851 to 0.902134**. These are executed-policy
scores on all 40 LAFAN1 clips, evaluated with MuJoCo Warp.

The comparison starts from the [pinned upstream released recipe](https://github.com/OamIcs/NablaMimic/tree/3494f168d7839f37396fefbda296d5554bdf6617).
The three recipe changes follow the [current project paper, §5 and Appendix D](https://sailab.ai/nablamimic/assets/nablamimic.pdf):
policy EMA, continued training contributions after episode resets within a window,
and learning-rate warmup shortened from 500 to 100 outer updates.
[Implementation details](PAPER_RECIPE.md) describe weight selection and update counts.

Both conditions use one NVIDIA CMP 170HX, seed 0, the same pinned dependency environment,
G1 asset and reference data, and the same NablaEngine build with a 288-row capacity
correction and tape headers 7/9/8/8. That correction is separate from the recipe
comparison and is not installed by this repository's upstream dependency setup.
The numbers characterize that fixed experimental environment; they are not a
benchmark of the unmodified upstream installer.
The workload retains all 40 clips, 4,096 training environments, 64-control windows,
a 4,096-environment collector, 1,000 replay rows, UTD 0.4, eight actor regression
steps per outer update, ten solver iterations, and reference-residual actions.

| Measurement | Released recipe, live weights | Current-paper recipe, EMA weights |
|---|---:|---:|
| Return at 1,500 outer updates | 0.724010 | 0.837293 |
| Return at approximately three native training hours | 0.860851 | 0.902134 |
| Outer updates completed in that budget | 11,667 | 12,221 |
| Earliest evaluated checkpoint with return ≥ 0.82 | 92.79 min | 22.26 min |
| Previous evaluated point below 0.82 | 61.84 min: 0.805015 | 11.23 min: 0.772622 |

The sampled evaluations establish earlier above-threshold performance, but do not locate
the first crossing: intervening checkpoints were not evaluated and monotonic improvement
is not assumed. They do **not** establish an exact 4.2× time-to-threshold improvement.
Update throughput differs by about 4.75%; the main result is improved learning quality,
including the comparison at the same number of updates.

Training always uses live weights. At the final new-recipe checkpoint, evaluating live
weights gives **0.867189**, while evaluating its EMA gives **0.902134**. This isolates
weight selection at that checkpoint; it is not a separate EMA-training ablation.
EMA updates once after the complete eight-step actor regression group.
The new training path requires a fresh output directory and rejects resume: complete
RNG, replay, collector and physical-continuation recovery is not implemented.

Native training time excludes explicit actor/collector/EMA compilation and initial
replay construction, but includes lazy critic/statistics compilation and checkpoint
work. The paper excludes startup compilation and evaluation, reports three seeds on
RTX PRO 6000, and gives 0.910 ± 0.004 at three hours. Our hardware, seed count, sparse
evaluation and clock accounting do **not** verify the paper's eight-minute claim.

The paper's nominal metric is preserved: deterministic playback from each clip's first
frame, no evaluation randomization or noise, ten MuJoCo Warp solver iterations,
full-clip normalization, and zero contribution from the missing tail after termination.
Clip means receive equal weight. Body errors, executed support and capacity evidence
remain separate from the scalar return. In an independent observed execution, the final
EMA policy completed 40/40 clips, while only 12/40 met the separate body-fidelity gate
against its required 36/40. A qualifying clip requires p95 world body error ≤ 0.30 m,
p95 torso planar error ≤ 0.30 m, p95 absolute wrapped torso yaw error ≤ 20°,
and mean donor-aligned body error ≤ 0.10 m. All clips must complete without invalid
physical states or capacity loss. Completion alone does not establish fidelity.

Optional forward-only initialization skips unused adjoint resources and backward warmup.
Persistent evaluation restores native arrays in place between independent conditions,
with stable addresses and sticky diagnostic flags preserved throughout each trajectory.
A reset probe verified all 586 saved arrays across three boundaries, including a
between-condition contamination test; its strict numerical repeat check still failed.

A separate matched single-GPU study used forward-only initialization and two bounded
output-writer threads in every group. Fresh serial, persistent serial and simultaneous
original/observed persistent processes took **641.437, 581.534 and 520.372 seconds**:
a batch-time reduction of **18.87%** from the first to the last condition. This does
not measure the combined benefit against the original default initialization.
All six cross-group comparisons and all three within-group repeats failed the frozen
comparison (`rtol=1e-5`, `atol=1e-6` for floats; exact discrete-support rules), including
the fresh-process repeat. Two original executions also
terminated one clip early. These timings are one fixed-order batch observation;
**numerical equivalence is not established**, and the optimizations remain opt-in.

A fresh PyTorch run with fullgraph Inductor and five explicit CUDA Graphs completed
**12,640 outer updates in 10,801.301 native training seconds**. It uses the same
NablaEngine physical equations, with a separate [96-contact / 416-row capacity
extension](https://github.com/LingzheZhao/NablaEngine/pull/2) based on the published
288-row correction. Setup/compilation/capture took 151.980 seconds; the complete
training Python process, including imports, input loading and final serialization,
took 11,161.825 seconds. All 12,640 logged updates were contiguous, with correct
optimizer/EMA counts and zero capacity, unsupported-state, nonfinite-gradient and
replay-mismatch counters. Finite solver iteration-limit, nonconvergence and singular
diagnostics remain recorded; they were not all zero. Logged Torch allocation and
reservation stayed near 20.506/38.076 GiB, with a 31.815 GiB allocation peak during
setup. These are allocator readings, not every native or driver allocation.

This is one seed on the CMP 170HX, using the same all-forty-clip workload and
zero-update JAX actor/critic weights, with fresh Torch optimizer and EMA state.
The existing original and independently observed MuJoCo Warp evaluators use
separate fresh processes and physics states. Early conditions use at most two
workers on the same GPU; final EMA/live and the separately predeclared update-1500
condition are sequential. Evaluation begins after timed training.

| Checkpoint / policy | Native training min | Original return | Observed return | Observed complete | Body-fidelity qualifiers |
| --- | ---: | ---: | ---: | ---: | ---: |
| 560 EMA | 8.0048 | 0.735056 | 0.765108 | 37/40 | 0/40 |
| 1402 EMA | 20.0081 | 0.832314 | 0.832242 | 40/40 | 0/40 |
| 1500 EMA, matched update count | 21.4118 | 0.833954 | 0.824701 | 39/40 | 0/40 |
| 4211 EMA | 60.0043 | 0.868747 | 0.867958 | 40/40 | 2/40 |
| 6317 EMA | 90.0095 | 0.873693 | 0.874791 | 40/40 | 6/40 |
| 12640 EMA | 180.0093 | 0.876124 | 0.877816 | 40/40 | 11/40 |
| 12640 live | 180.0093 | 0.846315 | 0.846955 | 40/40 | 0/40 |

Original execution completed 35/40 clips at update 560 and 40/40 in the other
conditions. All seven observed conditions had finite scientific scalars and zero
invalid active steps or observed capacity losses. No condition passed the separate
body-fidelity gate requiring 36 qualifiers and all forty completions. The final EMA
world-body and donor-aligned MPJPE means were 0.266111 m and 0.025490 m on its own
recorded executed support. Original and observed readings remain independent;
partial trajectories and their supports are retained.

At 1,500 updates, current-paper JAX took 22.2628 minutes with original return
0.837293; this Torch condition took 21.4118 minutes with return 0.833954, a **3.82%
shorter native clock**. In approximately three native hours, Torch completed
**3.43% more updates**, but its final original EMA return was **0.026010 lower**
than JAX's 0.902134. JAX's final independent observed return was 0.902194, with
40/40 completions and 12/40 qualifiers. This Torch run has not matched that JAX
learning result, and its eight-minute checkpoint remains below 0.82. The sampled
twenty-minute checkpoint is above 0.82; it does not locate the first crossing.

These are bounded observations, not a causal framework benchmark. Torch excludes
all compilation/capture from its native clock, while JAX includes some lazy
compilation. JAX's per-update checkpoints include optimizer state; Torch's policy
checkpoints contain weights and metadata, with full state saved after timed training.
Torch and JAX have distinct random streams, compiler arithmetic and capacity
conditions; equal time also produces different update and schedule positions.
A read-only audit found no new mismatch in the declared EMA, LR, optimizer,
normalization, clipping, replay, horizon or action-control recipe. It did not
establish the cause of the learning gap or numerical equivalence of the complete
initialization/training paths.

For the new capacity, 36 CPU model/tape checks, 122 CPU tests, 32 mathematical update
groups and 122 native-interface checks passed. The production compiled-versus-graph
check passed 1,923 checks, including 612 bitwise-identical arrays and independent
collector-history copy checks. The full-size twelve-update smoke passed. The
[capacity study](studies/pytorch/CAPACITY_PROTOCOL.md) additionally passed 999 old/new
checks on shared inputs, real collision coverage through 96 contacts and beyond
the overflow boundary, 61 high-contact interface checks and 67 independent
Jacobian/Hessian checks. The first 5,120-state stress pool reached only 82 contacts
and remains a failed coverage condition; a separate frozen 43,040-state pool
covered every count 65–96 and correctly flagged 3,600 states above 96. Capacity
validation does not make 96 a universal upper bound.

The earlier 64-contact three-hour attempt stopped after **26.623 native
minutes** when update 1,878 exceeded the 64-contact capacity. The last valid saved
policy is update 1,750; there is no completed three-hour Torch result for this build.
The [published Engine row-capacity fix](https://github.com/LingzheZhao/NablaEngine/pull/1)
retains that 64-contact limit. A contact-capacity extension is a separate Engine condition.

In that earlier failed condition, at 8.0055 and 20.0058 native minutes, EMA's
original nominal returns were
**0.756694 and 0.836650**, with 37/40 and 40/40 completed clips. At update 1,500,
return was **0.839496** after 21.2962 minutes, compared with current-paper JAX's
0.837293 after 22.2628 minutes. The 4.34% shorter native clock is a bounded observation:
Torch excludes all setup/compilation/capture, JAX includes some lazy compilation,
and per-update checkpoint contents differ. The same zero-update actor/critic weights
were used, but the frameworks have different RNG streams. This single-seed partial
run does not establish a causal framework speed advantage or the eight-minute claim.

At that failed condition's latest valid saved checkpoint, EMA/live original returns
were **0.812234 /
0.800892**, completing 38/40 and 35/40 clips. Their independent observed executions
returned 0.811798 / 0.807769 and completed 38/40 and 36/40. Across those five earlier
Torch checkpoint/policy conditions, observed physical-invalid and capacity-loss counts
were zero, but the separate body-fidelity gate remained 0/40. All partial trajectories
remain included. These checkpoint scores do not imply monotonic learning.

For that earlier 64-contact build, the production compiled-versus-compiled-graph
check passed all 1,924 checks, with all 612 compared arrays bitwise identical and independent collector-history copy checks
passing. The full-size twelve-update smoke passed the native validity and graph-count
gates. The separate eager-versus-compiled-graph FP32 check failed 17/1,924 checks,
including a reset-frame support difference; strict eager/compiled numerical equivalence
is not established. [The Torch study](studies/pytorch/README.md) defines the protocol.

## Fixed-input diagnosis of the Torch learning gap

The final original EMA evaluations both complete all forty reference horizons.
An approximate decomposition of the per-clip reward terms attributes 73.65% of
the 0.026010 return gap to body orientation and 18.11% to root position. The terms
were rounded to four decimals, so this attribution is approximate. The largest
joint-error increases include left wrist yaw, right wrist pitch and right wrist
yaw: their mean absolute errors increase from 0.1058/0.1023/0.0546 rad in JAX to
0.3928/0.4001/0.2747 rad in Torch. This locates the quality loss; it does not by
itself identify a training defect.

A frozen [reset and orientation study](studies/pytorch/RESET_ORIENTATION_PROTOCOL.md)
completed 1,361 comparisons on one CMP 170HX: 1,261 passed and 100 failed, with no
threshold changes. All failures involved TF32 network outputs or their own-action
downstream quantities. Independent reset/restore states and features passed at
160 rows over all forty clips under nominal, domain-randomization-only and full
training-reset settings. All FP32 readouts passed. With the same observations,
FP32 action differences for the initial, final JAX EMA and final Torch EMA weights
were at most 2.21e-8, 6.74e-6 and 6.20e-6. Allowing TF32 increased the final-policy
differences to 0.002795 and 0.004387 dimensionless action units. Holding the action
identical passed all 396 control comparisons, including exact control targets.
This isolates a cross-framework network-arithmetic difference under TF32
permission; it does not establish identical kernel selection.

All 31 orientation checks passed. Each final checkpoint supplied 64 separate
states from its observed fight1 trajectory: 32 evenly spaced eligible rows and
32 additional low-orientation-reward rows. Quaternion VJPs agreed with an
independent float64 finite-difference calculation within 3.41e-8; independent
MJX/native kinematics and one-step native control VJPs also passed, including
all three wrist coordinates. The one-step test used zero warmstart because the
trace did not record it, so it is a controlled counterfactual, not exact replay.
This error-enriched selection does not estimate overall policy quality.

The [compiler and precision study](studies/pytorch/PRECISION_PROTOCOL.md) separately
holds each final EMA policy, critic and normalizer fixed and tests two explicit
inputs over forty clips and four controls. All six execution/precision conditions
completed for each checkpoint. An earlier auxiliary nested-VJP compilation
failure is retained as a probe failure, without serving as training evidence. At each precision, compiled execution and explicit
CUDA Graph replay agree bitwise on all 173 retained fields. Fixed-action body
orientation gradients and boundary primitives pass throughout. FP32 eager versus
compiled execution fails two fields per checkpoint: a few BF16 history values
cross rounding boundaries. Those failures remain failures; they are not evidence
of bitwise eager/compiled equivalence.

Changing compiled FP32 to TF32 changes the four-step action-gradient vector by
0.295%/1.233% relative L2 for the two JAX-weight cases and 1.140%/1.329% for the
Torch-weight cases. Termination and frame support remain equal. In one Torch-weight
case, policy-output differences of about 0.003 in the first three controls grow
to 0.118756 in the fourth, at the right ankle pitch in fallAndGetUp2_subject3;
the target difference is 0.118243. The largest observed amplification is not
specific to the wrists. All nine retained invalid-state/gradient counters are
zero, while finite solver iteration-limit and nonconvergence counters are not
necessarily zero. This demonstrates precision-dependent closed-loop divergence,
not a demonstrated cause of the final three-hour learning gap.

The [production-shape window study](studies/pytorch/WINDOW_PRECISION_PROTOCOL.md)
then fixes the initial live weights and the Torch update-1,500 live weights
separately, at 4,096 environments and 64 controls. Each uses the checkpoint's own
critic and normalizer, the same materialized randomized reset states and noise,
and one compiled CUDA Graph at a time. Neither condition advances RNG state or
changes weights. These are controlled reset-start windows, not continuations of
the missing intermediate replay state.

| Live checkpoint | Raw action-gradient relative L2 difference | After target-gradient clipping | First unequal termination (control step) | Unequal termination entries |
| --- | ---: | ---: | ---: | ---: |
| Initial weights | 8.59% | 3.83% | 27th | 24 |
| Torch update 1,500 | 87.03% | 36.71% | 20th | 7 |

Both precision conditions complete with all nine invalid guards at zero. Each
checkpoint fails 26 of 29 strict precision comparisons; these remain failed
conditions, with finite outputs and no runtime exception. The
first initial-policy branch has FP32 terminate while TF32 continues, so these
readings do not establish that FP32 has better policy quality. Backpropagation
at the start of the window includes later branch differences; its early mismatch
must not be described as an early local-physics derivative defect. The common
64-control budget includes different reset paths after each first branch.

At update 1,500, the clipped wrist-gradient relative differences are
21.31%/25.75%/25.55%, with TF32-to-FP32 norm ratios 1.00066/1.00178/1.00049.
The target increment's overall relative difference is approximately 36.74%,
while its norm ratio is 1.00042. This is a change in direction and sample
allocation, without evidence of uniform wrist-gradient or step-size suppression.
Other joints also differ substantially. Torch peak allocated memory remains
below 16 GB, falling below 1 GB after each graph is released; reserved memory
also drops before the next condition. No simultaneous second trainer is used.

The precision intervention therefore changes actual reset branches and learning
targets at the production window size. It still does not establish the direction
or magnitude of its contribution to the final return gap. That attribution
requires a matched training comparison, with common random-number controls and
replicates; the present study makes no optimizer updates or convergence claim.

The actor-regression source audit and earlier shared-sample CPU update check found
no formula mismatch in BF16 storage, targets, clipping, Adam, schedules or EMA.
Distinct framework random streams remain an uncontrolled training difference.
The earlier 64- and 96-contact Torch runs also differ in their first update while
all invalid counters are zero; their different physical GPU assignments and
capacity builds prevent attributing that historical difference to capacity alone.

This work covers simulation-based NablaMimic reference tracking. It does not establish
generated skills or command composition, and does not integrate NablaMimic into BM.
