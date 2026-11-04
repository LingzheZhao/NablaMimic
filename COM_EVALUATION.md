# COM range-sum backport evaluation

The COM-only upstream backport removes the measured component's run-to-run bit
variation. In the paired full-evaluator experiment, the three-return range falls
from .02563164 to .01208337. Native physics and full returns still vary. This is
a bounded improvement under the stated conditions, not whole-simulator
determinism or a unique explanation of differences between specific live checkpoints.

Use the five-patch SDK for subsequent common evaluations. Adoption rests on
removing the measured local atomic-accumulation variation, agreement with the
COM formula and compatibility checks, with no observed time penalty in this
comparison. The 50% range-reduction gate below was a predeclared trigger for
one follow-up experiment, not a permanent adoption requirement or a general
uncertainty bound.

Current installations obtain the same validated `mujoco_warp` package sources from the
[SDK fork](https://github.com/LingzheZhao/mujoco_warp/tree/d39d6ad691e63ce94e66df69d509eff42047c2d9), at the commit selected by
[the dependency installer](scripts/setup_mujoco_warp.sh). Moving delivery from local diffs to the SDK fork
does not change the historical build identities, checkpoints or measurements reported below.

## Implementation and fixed conditions

The [tested patch revision](https://github.com/LingzheZhao/NablaMimic/commit/1cde4106d1a2e932e4adb6d96e0e432398e79bdb)
backports the COM portion of
[MuJoCo Warp #1872](https://github.com/google-deepmind/mujoco_warp/pull/1872), with
the subtree-size metadata from
[#1730](https://github.com/google-deepmind/mujoco_warp/pull/1730). The baseline is
the pinned adjoint branch plus patches 0001–0004; the intervention adds 0005.
Only `types.py`, `io.py` and the COM kernels/calls in `smooth.py` change. The
old nonzero-mass division rule, `_cinert`, `_cdof`, RNE/CRB and analytic-adjoint
interfaces are preserved. The newer zero-mass `xipos` fallback is excluded.
World summation follows upstream's root-COM times root-mass regrouping, including
the world's own mass; it need not round like the old tree sum.

One exclusive RTX 6000 Ada 48 GB runs the original SDK first, then the patched
SDK, in separate processes with fresh compiler caches. Both use the unchanged
official v0.2 evaluator, JAX 0.12.0.1 highest network matmul, Warp 1.18.0,
MuJoCo 3.17.0 and NumPy 2.5.4. The sole policy is the original Torch TF32
update-1500 live checkpoint from [the training comparison](https://github.com/LingzheZhao/NablaMimic/pull/2).
All forty LAFAN1-G1 references, ten solver iterations, deterministic mean actions,
no randomization/noise, key 1, 500-control bins and full-reference normalization
are retained. The longest rollout is 13,654 controls at 50 Hz. No training is
performed.

The component/native inputs are the six actual forty-world physical inputs and
seven saved kinematic fields from the earlier fixed-input probe. Model constants
are checked against that probe. Each SDK keeps the original constructor's
forward/backward warmup, takes a fresh current-schema public Model/Data snapshot,
and verifies in-place restoration and stable compiled/captured-graph identities.
The patched schema adds `body_subtreenum`; ancestor walks independently verify
every body's descendant range. The constructor snapshots differ in 58 of 586
common arrays, all Data fields. They are restored within each condition, not
claimed identical across SDKs. Opaque solver/collision workspaces remain outside
the restoration guarantee.

## Local formula, repetition and compatibility

Every captured G1 component call is checked against two FP64 references from the
same FP32 masses and inertial-frame positions: one uses independently summed
subtree masses, the other the stored FP32 subtree masses promoted to FP64.
Both preserve the old nonzero-mass branch. All checks pass rtol 1e-5 / atol 1e-6 m.
Maximum absolute errors over the references/readings are 1.90413e-7 m for the
original and 2.28078e-7 m for the patch; fixed order does not mean smaller error
at every input.

| Reading | Original | COM backport |
| --- | ---: | ---: |
| Subsequent component calls differing from the first, out of 15 | 5 | 0 |
| Subsequent native qpos/qvel/warm calls with bit differences, out of 14 | 14 / 14 / 14 | 14 / 14 / 14 |
| Native qpos/qvel comparisons outside rtol 1e-4 / atol 1e-5 | 0 / 0 | 0 / 0 |
| Native warmstart comparisons outside that margin, out of 14 | 14 | 13 |
| Largest within-condition warmstart difference, native acceleration units | .00167847 | .00279236 |

Each component field (`subtree_com`, `cinert`, `cdof`) has the stated bit-difference
count. All 225 cross-SDK native pairs pass qpos/qvel tolerances and fail warmstart
tolerance. Those failures are retained. They are not gradient correctness tests,
and mixed physical coordinates must not be compared through an amplification ratio.

The [public component check](studies/mujoco_warp/README.md) also passes five fresh
forward-only cases, sixteen captured repetitions each: branching/multiple roots
with a zero-mass subtree, nonzero world mass, a tiny positive subtree, all-zero
mass and tiny positive total mass. Masses are nonnegative and subtree masses
consistent. This does not extend the old adjoint's support to zero-mass dynamics.
The original SDK does not run these new-kernel fixtures.

Both original constructor warmups complete. A separate root-x cotangent smoke
produces finite, nonzero qpos/qvel/control gradients through the unchanged old
adjoint; all three cross-SDK gradient comparisons pass the original numerical
margin. This checks execution compatibility, not an independent derivative oracle.
Forward and all five backward buffers have zero capacity/unknown bits, including
after priming and before restoration. Sticky solver/line-search flags remain
recorded; inactive rows continue stepping, so final flags are not counts of
failures within valid trajectory supports. The final forward buffer of the third
patched full rollout also contains an `ITERATIONS` bit (512) for
`fallAndGetUp1_subject4`, alongside its line-search bit; the original condition
has no observed 512 bit. This clip records a height-tracking termination at 7.40 s,
but the sticky final flag does not locate the iteration-limit event before or
after that termination. It remains a retained solver diagnostic, not a capacity
failure or evidence that the valid segment failed to solve.

## Full-reference repeatability and time

Each condition compiles the original full rollout once, primes it once, then
measures three calls with restored public state and unchanged arguments. No
observation trace or altered rollout is used.

| SDK | Return 1 / 2 / 3 | References without recorded termination |
| --- | --- | --- |
| Original | .7262557353 / .7518873710 / .7419154229 | 33 / 36 / 35 of 40 |
| COM backport | .7384331374 / .7417246254 / .7296412557 | 34 / 35 / 34 of 40 |

The frozen eligibility rule requires an original return range above 1e-6, a
patched range no larger than half the original (or 1e-6), and no increase in
either the mean per-clip return range or normalized support dispersion.
The observed return range decreases by 52.86%; mean per-clip range changes from
.02706995 to .01747683. Support dispersion, defined as the sum of per-clip
alive-length ranges divided by summed full lengths, changes from .03089916 to
.02148590. All gates pass without using higher mean return as a selection rule.
Mean return changes from .74001951 to .73659967; this repeatability result is
not a quality gain for the fixed policy. Three repeats give neither a confidence
interval nor a general error bound.

| Mean synchronized call time | Original | COM backport |
| --- | ---: | ---: |
| Captured COM, 16 calls | 58.49 microseconds | 45.64 microseconds |
| Four-substep native physics, 15 calls | 2.665 ms | 2.595 ms |
| Full evaluator, 3 calls | 33.816 s | 33.093 s |

These are host wall times around execution plus device synchronization, excluding
state restoration, readback, compilation and priming; they are not GPU-event
kernel timings. This ordered comparison shows no observed time penalty and a
slightly lower full-evaluator mean (about 2.1%); it does not establish a stable
full-evaluator speedup ratio.
Complete worker processes take 221.28/218.93 s but include different validation
work, so their ratio is not a throughput estimate. Peak host RSS is 7.12/7.08 GiB;
maximum phase-boundary device samples are 1.977/1.971 GiB, not continuous peaks.

Both SDK conditions and CPU analysis complete successfully with no failed or
skipped execution phase. Strict native numerical mismatches remain separate
diagnostic results. The gate permits a subsequent common-evaluator comparison
of existing checkpoints; it does not change the measured identity of any older
score or establish which learner caused a remaining quality difference.

## Existing checkpoints under the common patched evaluator

After the frozen gate passed, five existing update-1500 fits were evaluated
with this same patched SDK, each EMA/live, in three interleaved rounds. One
process on another RTX 6000 Ada 48 GB visited the five fits in table order,
EMA then live, in every round. The original evaluator was compiled once and
primed once with the original Torch EMA; all thirty measured calls reused that
executable and native capture. Each call restored the same current public
snapshot and verified fixed actor/shared argument bytes and storage. Model
identities were never pooled. The shared reference, full denominator, solver,
precision and action settings match the preceding validation.

All thirty calls and all ten conditions complete successfully, with no failed
or unexecuted reading. Summaries reconstruct from the unrounded per-clip
reductions, including every partial trajectory. Private native workspace is
still not fully restored; interleaving does not establish its independence from
call history. This comparison shares one physical GPU internally, but does
not share the physical GPU or actor-call sequence of the paired SDK experiment.
Capacity/unknown flags remain zero. One final forward-buffer reading, TF32-repeat
live in round one, contains 512 plus 1024 for `fightAndSports1_subject1`, which
records termination at 9.72 s. As in the paired study, this sticky end reading
cannot place the iteration-limit event within or after valid support.

### EMA policies

| Fit | Three full-reference returns | Mean | Range | References without recorded termination, out of 40 |
| --- | --- | ---: | ---: | --- |
| Torch TF32, original | 0.834704755 / 0.822746756 / 0.838472568 | 0.831974693 | 0.015725812 | 39 / 39 / 40 |
| Torch TF32, repeat | 0.835444105 / 0.836097656 / 0.836012607 | 0.835851456 | 0.000653550 | 40 / 40 / 40 |
| Torch highest | 0.838622175 / 0.838844303 / 0.838398873 | 0.838621784 | 0.000445429 | 40 / 40 / 40 |
| JAX, first | 0.829645525 / 0.828592687 / 0.828477568 | 0.828905260 | 0.001167957 | 39 / 39 / 39 |
| JAX, repeat | 0.839145848 / 0.838496185 / 0.839397533 | 0.839013189 | 0.000901348 | 40 / 40 / 40 |

### Live policies

| Fit | Three full-reference returns | Mean | Range | References without recorded termination, out of 40 |
| --- | --- | ---: | ---: | --- |
| Torch TF32, original | 0.749076463 / 0.726988177 / 0.729363815 | 0.735142818 | 0.022088287 | 35 / 34 / 34 |
| Torch TF32, repeat | 0.761471429 / 0.762023941 / 0.760251321 | 0.761248897 | 0.001772621 | 37 / 37 / 36 |
| Torch highest | 0.767387634 / 0.745821016 / 0.770006977 | 0.761071876 | 0.024185961 | 34 / 33 / 35 |
| JAX, first | 0.741904312 / 0.733742818 / 0.754594372 | 0.743413834 | 0.020851554 | 31 / 33 / 34 |
| JAX, repeat | 0.773077554 / 0.776258301 / 0.767667468 | 0.772334441 | 0.008590833 | 34 / 35 / 34 |

The TF32-repeat EMA mean is .00694620 above the first JAX EMA and .00316173
below the JAX-repeat EMA. The original Torch EMA mean also lies between those
JAX means, with a substantially wider observed evaluation range. These are
separate fits, not a pooled estimate of either framework's quality.

Live ranking depends on the checkpoint: original Torch is below both JAX means;
Torch TF32 repeat and highest are above JAX first and below JAX repeat. The
TF32-repeat live mean is .01783506 above JAX first and .01108554 below JAX repeat.
Thus a single general Torch-versus-JAX live gap is not supported by this panel.

Highest EMA has a .00277033 higher mean than TF32-repeat EMA in these readings;
highest and TF32-repeat live means are nearly equal, with much greater variation
for highest live. Highest live has 33–35 references without termination per
reading versus 36–37 for TF32-repeat live, so the scalar means do not imply
uniform task improvement. All five fits have higher EMA than live mean returns.
This adds a fixed-checkpoint quality reading, not an estimate
across independent training seeds. The previously measured highest fit took
1.730 times the TF32-repeat complete training process. The measured recipe's
working choice remains Torch + NablaEngine + TF32/compile/CUDA Graphs + EMA,
with official JAX v0.2 as the algorithm/configuration reference.

The initial .01208337 range is not an error bar for this panel. Original Torch
EMA/live ranges reach .01572581/.02208829; highest live reaches .02418596 and
JAX-first live .02085155. Even the same original-Torch live checkpoint has a
different range under this interleaved, different-device context. These outcomes
retain the conclusion that COM is repeatable while the complete evaluator is not.
No checkpoint has a matched original-SDK three-repeat comparison in this second
allocation, so differences from historical scores are not that checkpoint's
isolated causal response to the COM patch. Scores from different SDK conditions
must not be substituted for the common-evaluator readings above.

The complete reevaluation process takes 1,175.39 s (19.59 min). Compilation is
12.38 s, full priming 32.67 s, and the thirty synchronized measured calls total
984.43 s (32.606–33.072 s each). Construction, checks and serialization account
for additional process time. Peak host RSS is 7.60 GiB; the maximum phase-boundary
device sample is 2.951 GiB. These are an amortized common-evaluator workload,
not a comparison of JAX and Torch inference runtimes: both sets of weights run
through the same JAX network and MuJoCo Warp physics.
