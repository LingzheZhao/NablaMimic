# Policy averaging and the current paper recipe

`configs/ours_ema.args` adds the three changes described in the current
[NablaMimic paper](https://sailab.ai/nablamimic/assets/nablamimic.pdf): policy
parameter EMA, training contributions after episode resets within a window, and
100 outer updates of learning-rate warmup. It preserves the released workload:
all LAFAN1 clips, 4096 training and collector environments, 64-step windows,
1000 replay rows, eight action-target regression minibatches, and residual joint
setpoints. The released `configs/ours.args` retains its original resolved
parameters, including masking after the first done and 500-update warmup. Both
recipes include the same shared `configs/base/action_target.args` workload.

Training always uses the live encoder and decoder: collection, differentiable
rollouts, action targets, and regression never use the EMA policy. The EMA starts
as a copy of the initialized actor and updates once after the complete regression
scan, with

```
beta = min(0.999, (1 + n) / (10 + n))
ema = beta * ema + (1 - beta) * live
```

Here `n` is the number of previously completed EMA updates; the first decay is
0.1. Only the trainable `enc` and `dec` trees are averaged. The current live
`anchor_dec` shape marker and optional `obs_norm` statistics are copied. Disabling
the window mask uses the existing detached reset state and restarted discount;
it does not connect gradients across episodes. Warmup and cosine-decay arguments
count outer updates, so the regression and critic schedulers multiply them by
their own optimizer steps per outer update.

Each new checkpoint keeps `actor` as the live policy. EMA runs additionally save
`actor_ema` and `ema_updates`; all new checkpoints contain `evaluation_policy`,
`completed_updates`, and `ema_updates` (zero when disabled). The trainer checks
that the EMA count matches completed outer updates before saving. A numbered
`ckpt/it_N.pkl` is saved before outer update N and therefore contains N completed
updates. The historical `params.pkl` iteration label is emitted after that
update; use its explicit `completed_updates`, not its label, for effort. Training
optimizers and their learning-rate counters remain associated with the live
policy.

This EMA training path supports fresh runs only. It rejects `--resume`,
`--shac-strict`, and a nonempty output directory. The upstream checkpoint does
not preserve the full RNG stream, physical continuation, collector, replay, and
data-iteration state. Saving EMA weights and its counter does not make that
training continuation exact. The unit tests check numerical EMA continuation
after serialization, not full training recovery.

`nablamimic.evaluate` defaults to EMA for the new recipe; `--policy live` measures
the live weights separately. Asking for missing or inconsistent EMA state fails
instead of falling back to live weights. Results include the selected policy and
both update counts, with distinct `itN/nominal/ema` and `itN/nominal/live` keys.
Select numbered checkpoints at predeclared training-time targets, taking the
first saved checkpoint at or after each target and reporting its actual time.
Do not select the best-performing checkpoint. Evaluation uses the donor's
deterministic nominal MuJoCo Warp protocol: ten solver iterations, every clip
from its first frame, no randomization or observation/reset noise, and each
clip's missing post-termination tail contributes zero to its full normalized
return. Keep every clip and its executed support visible.

For the comparison to the running released-recipe baseline, retain the same
externally pinned dependency image, reference cache, robot assets, and
capacity-corrected NablaEngine condition: 288 solver rows and the same tape
representation. Keep that engine correction separate from these three recipe
changes; do not silently substitute another engine revision, hardware class,
dataset, or action definition. Record actual code and dependency identifiers,
resolved configurations, inputs, failed conditions, and output roots beside
each run. The source change itself is independent of BM integration.

Implementing this recipe does not establish an eight-minute reproduction. The
paper reports a three-seed mean curve on RTX PRO 6000 and interpolates its first
crossing of normalized return 0.82. A single seed on CMP 170HX supports a
checkpoint-specific comparison. Report setup, training, evaluation, and total
wall time separately. The donor's native `wall_s` excludes explicit actor,
collector, and EMA compilation, but includes lazy critic/statistics compilation
and checkpoint work; retain that accounting when comparing with the existing
baseline and disclose its difference from the paper's compilation-excluded
training time. Training rewards alone do not replace the executed evaluation.

## Optional evaluation initialization

`nablamimic.evaluate --forward-only` omits the unused adjoint data buffers,
backward context, backward callbacks, and backward warmup for nominal evaluation.
The flag is opt-in for controlled comparison; the default and all training paths
retain their differentiable initialization. Both modes use the same forward
model, capacity sizing, four physical substeps, and forward warmup. Patched
MuJoCo Warp's global gradient registration remains unchanged, because this
optimization does not assume it is independent of forward kernel dispatch.
Differentiating the forward-only instance fails explicitly.

Evaluation outputs declare `forward_only` at both the root and checkpoint record,
and reject merging distinct initialization modes. Each process still creates
fresh physics and retains sticky capacity flags. Validate fixed-checkpoint
scalar, per-clip, and observed-trajectory agreement before adopting the option;
removing unused initialization is not proof of numerical equivalence. Measure
process wall time for this comparison: the existing `seconds` reading starts
after environment initialization and therefore omits the intended saving.
