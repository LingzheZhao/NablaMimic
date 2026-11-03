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

This work covers simulation-based NablaMimic reference tracking. It does not establish
generated skills or command composition, and does not integrate NablaMimic into BM.

## Decoder position feedback

The optional `--decoder-position-feedback` extends the upstream controller's
relative torso orientation with the current reference torso position expressed in
the measured torso frame. These three values go directly to the decoder. The
encoder still receives only reference features, and the critic's dimensions and
values, reward, termination, and existing orientation-noise draws remain unchanged.
Position observation noise uses a separate random-key derivation. This is a
controller extension, separate from the paper-recipe reproduction above.

The comparison retains all 40 original LAFAN1 clips, seed 0, 4,096 environments,
64-control windows, 1,000 replay rows, eight actor-regression updates and twenty
critic updates per outer update, the EMA recipe and reference-residual actions.
Both training runs use a CMP 170HX. Deterministic evaluations run every clip from
its first frame, retaining native terminations and partial trajectories. The
9,000-update pair was evaluated on the same RTX 4090 with 48 GiB; the final
three-hour policies were evaluated on CMP 170HX. Original scalar evaluations and
observed trajectories are independent executions and are never pooled.

Common body errors use fourteen named LINK origins on identical positive 50 Hz
time supports within each pair. World error is the per-frame mean Euclidean body
error, averaged over the clip. Aligned error translates reference torso XY and
rotates reference body offsets by the difference of projected world headings,
preserving reference torso height. Clip readings receive equal weight. A native
tracking-failure endpoint is retained separately from prefix means; heading
singularities retain world-position support but have no aligned-error value.
This projected-heading body alignment is distinct from the donor's body alignment
using the yaw of the relative quaternion, also used in the observed quality rule.
The rule's separate heading bound uses the wrapped difference of world headings.
No contact or flight is inferred from body positions.

| Comparison | Orientation feedback: world / aligned error | Position and orientation feedback: world / aligned error | Completed clips |
| --- | --- | --- | --- |
| EMA at 9,000 completed updates, 64-contact capacity in both runs | 31.68 / 2.63 cm | 4.95 / 2.78 cm | 40/40 in each condition |
| Final three-hour EMA checkpoints | 24.99 / 2.56 cm | 4.85 / 2.82 cm | 40/40 in each condition |
| Final three-hour live checkpoints | 61.45 / 2.67 cm | 5.30 / 2.91 cm | 40/40 in each condition |

At 9,000 updates, world error decreases on all forty clips, by 84.4% in the equal
clip mean; aligned error increases slightly. The original orientation policy has
5/40 clips meeting the existing observed body-quality rule, versus 36/40 with
position feedback. Training times to those saves are 7,962.995 and 8,555.197
seconds respectively. This is a single-seed, equal-update comparison, not a
training-throughput benchmark.

The initial position-feedback run was rejected when a diagnostic read near update
9,180 reported a real contact-capacity overflow. It did not finish three hours.
Its earlier 9,000-update checkpoint was checked independently: all 153 numeric
arrays were finite, EMA counted 9,000 updates, actor-regression Adam counted 72,000
and critic Adam 180,000, with zero required invalid-physics counters through the
later 9,170 read. These earlier weights support the intermediate comparison; they
do not turn the failed full run into a successful one.

A fresh complete run used 96 contacts and 416 constraint rows, without changing
collision geometry, physical equations or training scale. It finished 11,034
updates in 10,800.549 native training seconds, with 88,272 actor-regression and
220,680 critic updates, finite model/optimizer state and no recorded required
invalid-physics event. EMA and live evaluations separately completed 40/40 and
each had 37/40 clips meeting the observed body-quality rule. Observed normalized
returns were 0.923660 and 0.918485; the independent original evaluations gave
0.923769 and 0.918469. The orientation control above ends at 12,221 updates under
64-contact capacity. Thus the final three-hour comparison changes both the
available contact storage and the number of updates; the equal-update comparison
is reported separately.

The complete run uses a separately tested capacity build. The fork's
[public 96-contact implementation](https://github.com/LingzheZhao/NablaEngine/pull/2)
is related work, not the exact build executed in this JAX experiment; its numerical
checks must not be substituted for this run's checks. The upstream installer still
pins its original Engine. Merely selecting the new recipe does not install the
measured capacity correction. An overflow remains a failed run, and 96 is not a
universal contact bound.

Eighteen focused CPU checks passed, covering decoder feedback and gradients,
reference-only encoding, unchanged critic/reward/termination, weight selection,
EMA and reset-window behavior. The full training and the four final evaluations
exercise the complete GPU path. These results concern reference tracking: the
decoder still needs an external reference pose and the residual action's nominal
joint target. They do not establish reference-free generation, command composition,
other seeds, or the separate PyTorch trainer's position-feedback implementation.
