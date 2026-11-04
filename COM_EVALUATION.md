# COM range-sum backport evaluation

The COM-only upstream backport removes the measured component's run-to-run bit
variation. In the paired full-evaluator experiment, the three-return range falls
from .02563164 to .01208337. Native physics and full returns still vary. This is
a bounded improvement under the stated conditions, not whole-simulator
determinism or a unique explanation of the Torch/JAX live-return gap.

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
kernel timings. Full-evaluator time is about 2.1% lower in this ordered comparison.
Complete worker processes take 221.28/218.93 s but include different validation
work, so their ratio is not a throughput estimate. Peak host RSS is 7.12/7.08 GiB;
maximum phase-boundary device samples are 1.977/1.971 GiB, not continuous peaks.

Both SDK conditions and CPU analysis complete successfully with no failed or
skipped execution phase. Strict native numerical mismatches remain separate
diagnostic results. The gate permits a subsequent common-evaluator comparison
of existing checkpoints; it does not change the measured identity of any older
score or establish which learner caused a remaining quality difference.
