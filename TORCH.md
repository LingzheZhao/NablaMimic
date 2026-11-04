# PyTorch training on NablaEngine

`nablamimic.torch_train` implements the current `configs/ours.args` action-target
recipe through NablaEngine's Torch interface. The control body, encoder, targets,
functional Adam updates, collector transition and statistics use `torch.compile`.
Explicit CUDA Graphs capture the complete differentiable window and reusable
updates. Random inputs are sampled outside capture. Compilation/capture errors
stop the run; no automatic eager fallback changes the execution condition.

The code follows the official 0.2.0 arguments and evaluation interface. In the
pinned provisioned environment, all 84 CPU tests, 32 mathematical-update
comparisons and 36 capacity-source checks passed. On an RTX 6000 Ada 48 GB with
the declared 96-contact/416-row Engine, the native JAX/Torch window comparison
passed all 137 checks. Three separate compiled-versus-CUDA-Graph comparisons
passed all 1,923 checks each: FP32, TF32-permitted, and FP32 with EMA disabled.
These comparisons use eight environments, four controls and two updates.

The complete 4,096-by-64 FP32 workload with replay length 1,000 also completed
twelve updates. Native invalidity counters stayed zero, all five graph replay
counts matched, and actual precision stayed highest/TF32-disabled. Peak Torch
allocated/reserved memory was 31.81/38.24 GiB and did not grow across the twelve
logged updates. None of these five GPU validation stages failed or was skipped.
They establish their stated native, capture/storage and bounded execution
scopes; they do not establish full-size compiler equivalence or long-training
quality on this revision. Earlier fixed-snapshot experiments remain separate
readings. The [validation studies](studies/pytorch/README.md) define the checks.

## Fixed 1,500-update comparison

The [tested Torch revision](https://github.com/LingzheZhao/NablaMimic/commit/b13375e22948d505579a627e8fc1bce99cccb31d)
completed 1,500 updates on an RTX 6000 Ada 48 GB with `compile-graph` and
`nets_tf32`. The official JAX comparisons use
[the same v0.2 recipe](https://github.com/OamIcs/NablaMimic/commit/bfbb2a6773b4776c17402a34a93ec7589d9c7c18).
All three executions used the same physical GPU at different times, the same
96-contact/416-row Engine and forty-clip reference collection. The fifty shared
configuration settings matched: 4,096 environments, horizon 64, replay 1,000,
eight regression minibatches, 100-update warmup and EMA 0.999. Initial actor,
EMA, critic, normalizer and reset-bin state matched; subsequent random streams
remain framework-specific. The two JAX readings are same-seed technical repeats,
not two independent training seeds.

| Condition | Checkpoint clock (min) | Complete training process (min) | EMA return / completed | Live return / completed |
| --- | ---: | ---: | ---: | ---: |
| Torch | 14.185 | 16.744 | 0.835171 / 39 of 40 | 0.745769 / 35 of 40 |
| Official JAX, first | 26.838 | 28.693 | 0.828801 / 39 of 40 | 0.793606 / 37 of 40 |
| Official JAX, repeat | 26.929 | 28.806 | 0.839127 / 40 of 40 | 0.785574 / 35 of 40 |

The complete Torch training process was 1.71–1.72 times faster, including cold
imports, compilation/capture and weight saves. Its setup/capture took 133.48
seconds. Torch checkpoint auditing ran separately (2.26 seconds); the JAX
training processes include their final checkpoint/backend audit. Additional
wall-time saves and terminal replay-state serialization were disabled for Torch
(`--save-wall-minutes 6001 --no-full-state-at-end`), preserving only numbered
checkpoints every 250 updates. These are fixed-update-count timings, not times
to a quality threshold. The ratio compares the implementations as executed:
JAX additionally computes actor setpoint-output-change RMS/quantiles and
pre/post full-window critic-fit diagnostics every update; this Torch run omits
them. Their cost, reset/restore differences and compilation/graph contributions
have not been isolated. Neither all of the gain nor all of the gap can be
assigned to any one of these differences without profiling.

The checkpoint clocks also include different compilation boundaries. JAX first
compiles critic/statistics and EMA after that clock starts; Torch compiles and
captures those operations beforehand. In the later 1000-to-1250 update interval,
checkpoint-clock differences are 142.045 seconds for Torch and 268.502/268.217
seconds for the two JAX readings. From 1250 to 1500 they are 142.462 seconds and
268.359/269.367 seconds, respectively. These later intervals retain a 1.884–1.891
time ratio, including each implementation's diagnostics, logging and saves;
they do not isolate equal-computation kernel speed.

All 1,500 logged updates were finite, all nine native invalidity counters were
zero, and optimizer, native-work and five graph replay counts matched. Seven
numbered checkpoints passed their audits. Peak Torch allocated/reserved memory
was 31.81/38.24 GiB without growth across the logged updates; JAX allocator peaks
were 45.78 GiB. These framework allocator definitions differ and exclude some
native/Warp allocations, so they are not total-device peak measurements.
Device-wide samples at fifteen-second intervals reached 38.88 GiB for Torch
and 46.79–46.83 GiB for JAX. Those samples miss the first 309 seconds of Torch
and the first 62/68 seconds of the JAX training processes. Torch has the lower
sampled maximum; these observations do not bound either full-process peak.

Each policy was evaluated in a separate process by the original official
MuJoCo Warp evaluator with highest FP32 matmul, ten solver iterations, no noise
or randomization and full-reference normalization for all forty clips. The EMA point reading
lies between the two JAX readings; live return is lower than both. Torch EMA
has one tilt termination in `walk2_subject3` at 200.34 seconds. Live has five
height terminations, including `fallAndGetUp1_subject4` at 7.42 seconds; partial
supports remain in the recorded outcomes. Height/tilt labels are reference-relative
tracking termination criteria. A height label means root-height error exceeds
0.25 metres; it does not by itself establish a mechanical fall. This one Torch fit establishes no
statistical quality equivalence, general backend superiority, eight-minute
crossing or independent physical-fidelity acceptance. The
[JAX comparison record](https://github.com/LingzheZhao/NablaMimic/pull/1)
keeps the two baseline checkpoint identities separate.

## EMA and precision follow-up

A second TF32 fit completes 1,500 updates with EMA .835851 (40/40 references)
and live .761672 (37/40). A highest-precision intervention reaches EMA .831668
(39/40) and live .768678 (34/40), taking 1.730 times the complete-process duration
of that repeat with the same terminal full-state saving. Both retain the same
31.81/38.24 GiB allocator peaks. These same-seed point readings use different
physical cards of the same model and do not establish a quality distribution;
the mixed result does not support changing the TF32 default.

[The diagnostic results](TORCH_DIAGNOSTICS.md) retain all seven checkpoint
readings, per-clip execution supports, fixed-weight evaluation variation,
production-size strict numerical failures, a learner update with real late
Adam state, and the completed precision intervention. Precision differences
already appear before physics, but they do not uniquely explain the final
EMA–live gap. Identical weights also vary across evaluations, so the curve's
last interval does not establish late training deterioration. No schedule
change is inferred from these bounded observations.

## Environment and training

Install the optional Torch dependency into a provisioned, pinned environment:

```bash
pip install -e '.[torch,test]'
```

A compatible separately pinned NablaEngine build must provide `torch_ffi`, the
G1 differentiable kinematics interface and native validity diagnostics. Keep its
source and build identity with the run. This extra does not change the official
`engine` dependency pin or install a capacity-modified Engine. The retained
capacity qualification specifically checks a declared 96-contact/416-row build;
it does not establish that a different Engine version or capacity is equivalent.

Within the provisioned GPU environment, with the ordinary reference/asset paths
configured as for the JAX trainer:

```bash
python -m nablamimic.torch_train --config configs/ours_torch.args --out runs/torch_s0
```

`ours_torch.args` includes official `ours.args`, then selects `compile-graph`.
The actor average uses `--actor-ema 0.999 --actor-ema-warmup`, one update after the
complete eight-minibatch actor-regression group; the live actor remains the
training/collector policy. `--actor-ema 0` disables averaging. The Torch EMA
follows the official floating-leaf rule and the requested decay/warmup settings.
Other training variants remain constrained by the Torch trainer's explicit
validation; BONES, SHAC and resumed execution are not implemented here.

Network matmuls permit TF32 by default (`--matmul-precision nets_tf32`). Select
`--matmul-precision highest` to disable TF32 before compilation and capture.
The effective backend settings are recorded before setup, after setup and at
termination. The native physics options, BF16 regression-input storage, reward,
gradient clipping and optimizer formulas are unchanged by this switch. Precision
permission does not imply bitwise agreement across frameworks or compilers.

`--initial-checkpoint PATH` imports actor/critic weights from a fresh current
official JAX numbered checkpoint with `iter=0` and `env_steps=0`. Its current
action-target fields, finite FP32 arrays, network dimensions and initial
normalizer are checked. An optional initial `actor_ema` must equal the live actor.
This is not a checkpoint converter or a training continuation: optimizer,
replay, collector and RNG state start fresh in Torch. Same seed labels do not
produce matching JAX and Torch random streams.

Training saves NumPy weight checkpoints with `actor`, optional `actor_ema`,
`critic`, `cnorm`, `iter`, `env_steps` and `wall_s`, plus Torch run metadata.
A terminal full-state artifact also records optimizer, scheduling, RNG and replay
state for inspection; restoration from it has not been implemented or validated.
Every run requires a fresh output directory. The measured training clock excludes
compilation/capture and the terminal save. The trainer total includes setup and
that save, but there is no isolated serialization timer. Their difference must
not be reported as serialization time; the complete process additionally
includes imports and reference loading before trainer construction.

## Official evaluation

The official JAX evaluator reads the Torch NumPy weight layout directly. Select
the key explicitly and preserve the two readings separately:

```bash
python -m nablamimic.evaluate --run runs/torch_s0 --iters 1500 \
  --actor-key actor_ema --out runs/torch_s0/eval_ema_1500.json
python -m nablamimic.evaluate --run runs/torch_s0 --iters 1500 \
  --actor-key actor --out runs/torch_s0/eval_live_1500.json
```

No changes to official JAX training, evaluation, network/environment mathematics
or argument resolution are required by this port. The private earlier nominal
evaluation instrumentation and historical result readers are not part of this
PyTorch implementation. Report checkpoint identity, complete/partial horizons,
falls and tracking fidelity; survival alone does not establish motion quality.
