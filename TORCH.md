# PyTorch training on NablaEngine

`nablamimic.torch_train` implements the current `configs/ours.args` action-target
recipe through NablaEngine's Torch interface. The control body, encoder, targets,
functional Adam updates, collector transition and statistics use `torch.compile`.
Explicit CUDA Graphs capture the complete differentiable window and reusable
updates. Random inputs are sampled outside capture. Compilation/capture errors
stop the run; no automatic eager fallback changes the execution condition.

The code follows the official 0.2.0 arguments and evaluation interface. This
revision has not yet been validated by a new provisioned CPU/GPU run. Experiments
performed with earlier fixed snapshots do not validate this revised source or
establish its training quality, speed or cross-framework numerical equivalence.
The [validation studies](studies/pytorch/README.md) state the necessary checks.

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
compilation/capture and reports terminal full-state serialization separately.

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
