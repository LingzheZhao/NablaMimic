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

The PyTorch reproduction uses the same native NablaEngine with fullgraph Inductor
and explicit CUDA Graphs. Its first three-hour attempt stopped after **26.623 native
minutes** when update 1,878 exceeded the 64-contact capacity. The last valid saved
policy is update 1,750; there is no completed three-hour Torch result for this build.
The [published Engine row-capacity fix](https://github.com/LingzheZhao/NablaEngine/pull/1)
retains that 64-contact limit. A contact-capacity extension is a separate Engine condition.

At 8.0055 and 20.0058 native minutes, Torch EMA's original nominal returns were
**0.756694 and 0.836650**, with 37/40 and 40/40 completed clips. At update 1,500,
return was **0.839496** after 21.2962 minutes, compared with current-paper JAX's
0.837293 after 22.2628 minutes. The 4.34% shorter native clock is a bounded observation:
Torch excludes all setup/compilation/capture, JAX includes some lazy compilation,
and per-update checkpoint contents differ. The same zero-update actor/critic weights
were used, but the frameworks have different RNG streams. This single-seed partial
run does not establish a causal framework speed advantage or the eight-minute claim.

At the latest valid saved checkpoint, EMA/live original returns were **0.812234 /
0.800892**, completing 38/40 and 35/40 clips. Their independent observed executions
returned 0.811798 / 0.807769 and completed 38/40 and 36/40. Across all five evaluated
Torch checkpoint/policy conditions, observed physical-invalid and capacity-loss counts
were zero, but the separate body-fidelity gate remained 0/40. All partial trajectories
remain included. These checkpoint scores do not imply monotonic learning.

The production compiled-versus-compiled-graph check passed all 1,924 checks, with all
612 compared arrays bitwise identical and independent collector-history copy checks
passing. The full-size twelve-update smoke passed the native validity and graph-count
gates. The separate eager-versus-compiled-graph FP32 check failed 17/1,924 checks,
including a reset-frame support difference; strict eager/compiled numerical equivalence
is not established. [The Torch study](studies/pytorch/README.md) defines the protocol.

This work covers simulation-based NablaMimic reference tracking. It does not establish
generated skills or command composition, and does not integrate NablaMimic into BM.
