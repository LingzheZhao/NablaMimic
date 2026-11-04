# Torch EMA and precision diagnostics

The Torch repeat retains competitive EMA return and lower live-policy return
than the two JAX point readings. The highest-precision fit has a small live-return
increase in this comparison, but fewer completed clips, lower EMA return and
1.73 times the complete training-process duration; it does not support changing the default. Precision
sensitivity is measured before physics and through an actual late optimizer
update, but the EMA–live gap still has no unique causal explanation. An apparent
late curve decline is also confounded by evaluation variation with identical weights.

## Conditions and original-condition repetition

The Torch fits use the [fixed Torch source](https://github.com/LingzheZhao/NablaMimic/commit/b13375e22948d505579a627e8fc1bce99cccb31d),
the [official v0.2 recipe](https://github.com/OamIcs/NablaMimic/commit/bfbb2a6773b4776c17402a34a93ec7589d9c7c18),
the qualified 96-contact/416-row NablaEngine and RTX 6000 Ada 48 GB GPUs.
The original model, forty references, initial weights, seed-zero setting,
4,096 environments, horizon 64, replay 1,000 and optimizer/EMA schedules stay
fixed. The repeat uses another physical card of the same model. Same-seed
technical repetitions are not independent training seeds; forty evaluation
clips do not supply independent training repetitions. [TORCH.md](TORCH.md)
defines the original comparison and its timing boundaries.

| Fit | EMA return / completed | Live return / completed | EMA minus live |
| --- | ---: | ---: | ---: |
| Torch first, TF32 permitted | .835171 / 39 | .745769 / 35 | .089401 |
| Torch repeat, TF32 permitted | .835851 / 40 | .761672 / 37 | .074178 |
| JAX first, TF32 permitted | .828801 / 39 | .793606 / 37 | .035195 |
| JAX repeat, TF32 permitted | .839127 / 40 | .785574 / 35 | .053553 |

These are separate final-checkpoint, cold-process readings from the original
MuJoCo Warp evaluator: highest FP32 network matmul, deterministic mean actions,
ten solver iterations, no noise/randomization and forty full reference horizons.
The repeat live return is .015903 above the first Torch reading, but .031934
and .023901 below the original two JAX readings. Its three height terminations
are `run2_subject4` at 8.02 s, `fightAndSports1_subject1` at 10.36 s and
`fallAndGetUp2_subject2` at 15.08 s. Four originally terminated clips complete
and two new clips terminate; completion counts alone hide this change.

The repeat completes all 1,500 updates with all nine native invalidity counters
zero, expected optimizer/graph counts, and recorded allocator peaks. Its final
numbered checkpoint clock is 815.154 s and complete training process 997.608 s.
It additionally saves 18.196 GB of actual optimizer, replay, collector and RNG
state after training. This extra work prevents an equal-save-work speed ratio
against the original benchmark. The fixed producer has no isolated save timer.
The CPU audit verifies nonzero finite Adam moments, actor/critic counts
12,000/30,000, EMA/outer count 1,500, 4,096,000 replay rows, 3,750 collector steps,
and bitwise agreement between full-state weights and numbered checkpoint 1,500.
This supports the bounded learner-update diagnostic below, not production resume.

## Seven-checkpoint evaluation and repeatability

The unchanged evaluator is compiled once in each of six separate processes,
one per fit and policy key, and reuses its environment across seven checkpoints.
At most two processes share one GPU, each with JAX BFC fraction .35 and private
cold caches. Every cell reports EMA / live full-reference return.

| Updates | Torch first | JAX first | JAX repeat |
| ---: | ---: | ---: | ---: |
| 0 | .002604 / .002604 | .002604 / .002604 | .002604 / .002604 |
| 250 | .512410 / .510066 | .529692 / .495357 | .494768 / .515346 |
| 500 | .739076 / .676808 | .752608 / .636561 | .729826 / .652638 |
| 750 | .809756 / .750366 | .789911 / .730889 | .770750 / .665531 |
| 1000 | .822023 / .728507 | .807528 / .762485 | .813381 / .760021 |
| 1250 | .831152 / .741008 | .815597 / .726769 | .832044 / .787572 |
| 1500 | .818863 / .738870 | .809388 / .753234 | .839894 / .778960 |

Torch's gap is already .062267 at 500 updates. Both JAX fits also exhibit large
gaps at some checkpoints. At 1,000 Torch live trails both JAX live readings,
but at 1,250 it exceeds the first JAX reading. From 1,250 to 1,500 Torch EMA
falls .012289 and live .002138 in this evaluation sequence. EMA completion
changes 40 to 38; live completion increases 33 to 35. Neither sequence establishes
sustained late training deterioration.

Re-evaluating identical Torch final weights changes EMA by -.016307 and live
by -.006899 versus their original cold readings. The full-reference return change
of `fightAndSports1_subject1` accounts for about 94.5% of the EMA shift; this clip
now reaches the height criterion at 60.30 s instead of completing. Net
full-reference return changes in `dance2_subject2` and `dance2_subject3` account
for about 94.6% of the live shift, with their supports changing in opposite
directions. These approximate clip contributions use the cold report's
four-decimal per-clip returns; they do not separate prefix reward changes from
lost or gained support. The stored weights are unchanged. Per-clip
returns, support and cause are retained for all checkpoints. Historical
cumulative complete-process times at those checkpoints were not recorded;
they remain unavailable, rather than being reconstructed by adding a setup time.

A separate same-card control evaluates the same final Torch weights in three
ordered batches, with EMA/live concurrent in each: S1 uses fresh processes
for only checkpoint 1,500; M uses fresh processes for all seven checkpoints;
S2 repeats S1. Source, inputs, host summary observer, allocator settings and
private cold caches are held fixed.

| Evaluation context | EMA return / completed | Live return / completed |
| --- | ---: | ---: |
| S1, fresh single checkpoint | .838574 / 40 | .723423 / 34 |
| M, seven checkpoints | .838247 / 40 | .745780 / 35 |
| S2, fresh single checkpoint | .837968 / 40 | .738354 / 35 |

The two cold live readings differ .014931 and change support/cause in six clips.
EMA M lies between S1 and S2, all with complete supports; live M exceeds S2
by .007426. All six pairwise combined strict checks fail (native output,
unrounded per-clip rtol 1e-4 / atol 1e-5, exact support/cause). Fresh processes
therefore already vary, and M does not reproduce the earlier low EMA reading.
Two S readings and one M cannot identify or exclude a physics-history,
compilation-context or other mechanism. They do not form a confidence interval.

## Common-prefix and additional-tail accounting

Separate observed final-checkpoint executions record rewards and physical state
at 50 Hz. All six fail strict numerical comparison with the corresponding
curve execution; five also change support/cause. The following decomposition
therefore explains only these observed executions. Every clip retains its
full-reference denominator. Net tail means left-only tail minus right-only tail.

| Left minus right | Full-reference difference | Common prefix | Net tail |
| --- | ---: | ---: | ---: |
| Torch EMA minus live | +.095368 | +.024601 | +.070767 |
| JAX first EMA minus live | +.053812 | +.022147 | +.031665 |
| JAX repeat EMA minus live | +.040475 | +.020972 | +.019503 |
| Torch live minus JAX first live | -.015070 | -.004001 | -.011069 |
| Torch live minus JAX repeat live | -.056550 | -.006947 | -.049603 |
| Torch EMA minus JAX first EMA | +.026487 | -.000798 | +.027285 |
| Torch EMA minus JAX repeat EMA | -.001657 | -.001657 | .000000 |

Within the observed Torch pair, about 74.2% of the gap comes from additional
executed tails. This percentage is not transferred to the historical .089401
gap or the seven-checkpoint curves. Common-prefix reward remains explanatory;
it does not replace the full-reference primary metric or erase termination loss.

All twelve GPU workers complete. The first CPU analyzer exits with a
`partial_or_invalid` result: its six trace summaries fail because it treats a
per-control cause history as an endpoint scalar. Its curve readings remain
available. A separate corrected analysis
preserves that failed attempt and reads the existing traces without rerunning
GPU evaluations. All six current cause histories, 240 native per-clip reward
reconstructions and nine forty-clip decompositions pass their accounting checks.
The largest per-term decomposition residual is below 5e-16; FP32 native summation
and float64 reward resummation remain distinct numerical quantities.

The earliest observed Torch live height event is `fallAndGetUp1_subject4` at
7.40 s, control 370. Actual root height .118364 m is .253602 m below reference
height .371966 m, crossing the .25 m tracking threshold. Gravity-z error .538568
stays below the .8 tilt threshold. Over the last fifty supported controls actual
height spans .118364–.374736 m and reference .314141–.371966 m. This locates a
height tracking failure; it does not establish monotonic descent, mechanical
fall or force-classified contact. No substep force evidence was recorded.

## Production-window execution and precision controls

The first fit's trained live actor, critic, normalizer and reset-bin distribution
are frozen for a 4,096-by-64 differentiable window. A single preparation draws
reset starts and materializes all physical state, warmstart, action/observation
noise and minibatch indices. These are counterfactual starts from the trained
bin mixture, not recovered training replay. Ordinary weights omit Adam moments,
so this experiment performs no optimizer update.

Three fresh sequential processes on one card compare compiled TF32, graph TF32
and graph highest. The primary target displacement is
`action_scale * (target - noiseless_action)` after gradient clipping and RMS
target construction. It does not apply the actuator-range clamp. All inputs
remain bitwise unchanged, all four RNG states remain unchanged, and each measured
window executes 64 forward/backward calls with all nine invalidity counters zero.
The integer support/branch fields remain exact comparison requirements; floats
retain rtol 1e-4 / atol 1e-5 throughout.

| Fixed-weight comparison | Clipped-gradient relative L2 | Primary target displacement relative L2 | First branch difference |
| --- | ---: | ---: | --- |
| Original compiled TF32 to graph TF32 | .0003154 | .0003090 | None |
| Original graph TF32 to graph highest | .4216201 | .4257414 | Control 17 |
| Repeated compiled TF32 to graph TF32 | .0002947 | .0002904 | None |
| Same-card compiled TF32 repetition | .0002989 | .0002940 | None |
| Repeated graph TF32 to graph highest | .4216121 | .4257397 | Control 17 |

The repetition copies the original 326 MB payload byte for byte and runs all
four declared processes on the same physical card. Old-to-new same-mode target
differences are .0002953 for compiled TF32, .0003067 for graph TF32 and .000005267
for graph highest, with no branch differences. All these strict comparisons
still fail. The compiled/graph discrepancy is comparable to fresh same-mode
variation; this does not identify a capture-specific defect, establish numerical
equivalence or relax the original margins. These few repeats do not establish
an upper bound on execution variability.

The precision effect is much larger and repeats: about 42.57% target relative
L2, clipped-gradient cosine .9114, and the first differing done/termination
at zero-based control 16, environment 2286 (0.34 s into the window). The first
control's policy output already differs by relative L2 .000453, before this
branch split.
Later trajectory and reset differences contribute to the whole-window result;
it is not an isolated local-physics derivative comparison. Finite execution
and zero invalidity counters do not imply a similar learning direction. Other
solver diagnostics remain visible: `fwd_iteration_limit_worlds` is 391 in each
of these repeated weight-only windows; finite nonconvergence and singularity
records are also nonzero. The later actual-replay TF32/highest windows record
316/301 for that iteration-limit counter. These are accumulated native readings,
not a count of distinct failing environments or a local derivative accuracy test.

## Actual late optimizer update

The separate TF32 repeat provides authentic terminal replay, warmstart, Adam
moments, counts and all persistent state. A fixed window is sampled from that
replay and shared across the same three execution conditions. After constructing
and compiling/capturing the source trainer at its official initial state, the
terminal state is copied in place with stable storage identities and bitwise
checks on all 133 persistent tensor leaves before the update. The 36 explicit
input leaves and four RNG states remain unchanged; native counters pass their
declared checks.

Each condition runs the actual source window, eight actor regressions, twenty
critic updates and one statistics/EMA update. Actor/critic counts advance
12,000/30,000 to 12,008/30,020 and EMA/outer count to 1,501. Collector and replay
remain unchanged. This is one controlled learner update, not a resumed training
run or a complete collector-plus-learner iteration.

Actual policy change is read on the same reference condition's 262,144 BF16
history/goal rows converted to FP32, using one common eager highest-precision
noiseless readout before and after each update. The primary value subtracts
raw actions first, then multiplies by action scale; there is no control-range
clamp or new physical rollout. This full-window query has a different support
from official JAX's final-state action-change diagnostic, so their magnitudes
are not treated as matched measurements.

| Execution | Actual setpoint-change RMS (rad) | Critic MSE before | Critic MSE after |
| --- | ---: | ---: | ---: |
| Compiled TF32 | .0123790 | .0565919 | .0350685 |
| Graph TF32 | .0123785 | .0565919 | .0350685 |
| Graph highest | .0123297 | .0561472 | .0350670 |

Critic fit uses each condition's own target window and the unchanged pre-update
normalizer, with the source formula `1 - MSE/(Var(target) + 1e-12)`. Initial
fixed-query policy outputs are bitwise identical across conditions. Actual
policy movement is smaller than the approximately .02955 rad RMS target schedule;
target displacement and realized weight-update effect must remain separate.

Compiled versus graph TF32 has target relative L2 .0001354 and actual policy-change
relative L2 .0016199, with no support split. Graph TF32 versus highest has
.2864832 target relative L2 and .2195937 actual policy-change relative L2; its
first done split is control 53 (zero-based 52, environment 3998). Similar RMS
magnitudes therefore do not mean matching update vectors. Both strict comparisons
fail and remain recorded. One window cannot establish persistent update oscillation
or critic failure; critic MSE decreases in all three conditions.

Single instrumented window calls take about .352/.346/.565 s for compiled TF32,
graph TF32 and graph highest. Stage synchronization, dispatch and diagnostic
readbacks make these diagnostic call timings, not steady-state throughput.
The control body combines network, physics and reset work, so their separate
contributions remain unavailable. The full-window query also includes file and
host/device transfers; its cost does not estimate JAX's resident final-state
diagnostic overhead. These measurements do not attribute the original complete-
process speed gain to individual components.

## Precision intervention decision

Before the repeat and probes, the follow-up rule required live return below
each original JAX reading by more than .005 and graph TF32-to-highest primary
target displacement relative L2 at least .10. A same-precision mismatch had
to be investigated first. Both numerical triggers are met. The identical-input
same-card compiled repetition completes that investigation without turning any
strict failure into a pass. This motivates a separate full highest-precision
training comparison; it does not already establish precision as the cause of
the EMA–live gap. The intervention retains the official schedule and changes
training matmul precision; the code's default remains TF32.

## Complete highest-precision intervention

A new 1,500-update fit changes only network matmul precision to `highest`,
with CUDA matmul TF32 disabled before compilation/capture. The same fixed source,
initial weights, seed-zero RNG settings, recipe, schedule, references, Engine
and terminal full-state saving as the TF32 repeat are used. The actual state
and sampling trajectories can diverge after changing precision. It uses another
physical RTX 6000 Ada 48 GB; there is no simultaneous same-device training pair.
Native physics precision and BF16 regression observation storage are unchanged.
Both fits pass all 1,500-update guards, graph/optimizer counts, seven-checkpoint
audits and complete terminal-state audits.

| Condition | Checkpoint clock (s) | Complete train process (s) | EMA return / completed | Live return / completed |
| --- | ---: | ---: | ---: | ---: |
| TF32 repeat | 815.154 | 997.608 | .835851 / 40 | .761672 / 37 |
| Highest | 1541.432 | 1725.746 | .831668 / 39 | .768678 / 34 |

Highest improves this live point reading by .007006 but reduces EMA by .004183;
EMA/live completion decreases by one/three clips. The gap narrows from .074178
to .062990 without closing the live gap to either JAX reading. Highest EMA has
a height termination in `fallAndGetUp3_subject1` at 70.30 s. Live height
terminations occur in `fallAndGetUp1_subject1` at 78.70 s,
`fallAndGetUp1_subject5` at 78.68 s, `fallAndGetUp2_subject2` at 15.06 s,
`jumps1_subject2` at 205.16 s and `walk3_subject5` at 182.40 s; `walk2_subject3`
reaches the tilt criterion at 199.54 s. Although fewer clips finish, mean
executed reference fraction rises from .929176 to .936115. This is another
reason to retain supports and full-reference return rather than substituting
completion count. There is no common-prefix reward decomposition for this pair.

The saved-clock duration ratio is 1.891 and the complete-process ratio 1.730,
with the same terminal-save policy. Both show the same logged allocator peaks,
31.81 GiB allocated and 38.24 GiB reserved, without growth across their logged
updates. These are allocator measurements, not total-device peaks. Unlike the
original Torch/JAX speed comparison, both new Torch fits perform the same kinds
of diagnostic and save work; different cards, state paths and single-run timing
still limit attribution. The .007006 live gain is not a reproducible-benefit or
statistical-equivalence claim. The separately measured evaluation variability
belongs to the first Torch fit and is not a confidence interval for these fits.

This intervention does not support replacing the current TF32 default: the
quality result is mixed and the measured compute cost is higher. Fixed-input
diagnostics establish sensitivity to precision, while this training pair records
different outcomes; the pair does not uniquely attribute those differences to
precision or explain the EMA–live gap. The sparse checkpoint curves and one actual-moment
update do not establish persistent update oscillation, so no schedule change
is tested or folded into the implementation.

## First-control eager and compiled localization

A bounded forward-only probe reuses the first weight-window payload byte for
byte and keeps the trained actor, original FP32 observations, all reference
constants and RNG fixed. Four fresh sequential processes on the same physical
card run eager/compiled at TF32/highest. The continuous and reset encoders each
retain the original 64-by-4,096 batch; only the first-control latent/action is
read. Native stepping/kinematics callbacks are forbidden; there are no physics,
backward, optimizer or CUDA Graph calls. All five phases complete, with all
input/weight/reference checks and saved RNG checks passing.

Both compiled conditions reproduce their matching original production first
policy actions bitwise; eager highest also passes the unchanged first-action
tolerance. Eager TF32 fails that production closure, and its result is retained.
The reduced observation/decoder compilation boundary and `no_grad` scope are
explicit: passing first-action closure does not validate a complete window.

| Comparison | First encoder latent relative L2 | First action relative L2 | Strict comparison | Both production action closures |
| --- | ---: | ---: | --- | --- |
| Eager to compiled, TF32 | .000000 | .000187578 | Fail | Fail |
| Eager to compiled, highest | .000000 | .000000203 | Pass | Pass |
| TF32 to highest, eager | .000471223 | .000493217 | Fail | Fail |
| TF32 to highest, compiled | .000471223 | .000452969 | Fail | Pass |

The production-closed compiled comparison locates a precision-related change
at the encoder output and action before physics integration. Its maximum
absolute action difference is .00171128 in network action units. Eager versus
compiled TF32 has identical recorded latent outputs but a different final
action; the observation construction and decoder share the changed compilation
boundary, so this does not identify the decoder alone or a particular layer.
The eager-TF32 mismatch cannot be transferred to production gradients. No
first-action result establishes a local physical VJP error or explains the full
training quality difference by itself.
