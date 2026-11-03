# Validate the Torch implementation

These programs run in explicitly provisioned pinned CPU/GPU environments. They
write fresh outputs beside the run and do not modify the reference cache or
Engine source. The current code is based on the official 0.2.0 recipe and APIs;
earlier fixed-snapshot experiment results are not acceptance evidence for it.
No quality or speed result is claimed before these checks and new training have
actually run on this revision.

## CPU mathematics

Run the core tests with the current JAX/Optax definitions and fresh arrays:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m pytest \
  tests/test_configs.py tests/test_torch_env.py tests/test_torch_numerics.py \
  tests/test_torch_execution.py tests/test_torch_precision.py \
  tests/test_torch_checkpoint.py tests/test_engine_contract.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python studies/pytorch/update_check.py \
  --source-revision "$SOURCE_REVISION" --out "$OUTPUT_ROOT/update-mathematics"
```

The update study extracts the current official action-target, critic, sampler
and EMA expressions from `train.py`, shares the exact sampled indices with
Torch, and compares one complete outer update. CPU tests also check gradient
clipping, quaternion derivatives, actor-input and parameter VJPs, BF16 storage,
Adam, schedules, official EMA options, precision flags and current zero-update
checkpoint validation. Small CPU tests do not qualify native CUDA physics,
production memory use or full-size compiled numerical equivalence.

## Native and compiled execution

The Engine build is an explicit input. The following capacity gate is specific
to a compatible 96-contact/416-row build, with 1,504 tape words and thirteen zone
words. The Torch port does not modify these native kernels. Record the Engine
base/capacity revisions and fixed dependency environment externally, then expose
the chosen source through the provisioned environment.

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python studies/pytorch/engine_capacity_check.py \
  --cpu --engine-capacity-revision "$ENGINE_REVISION" \
  --out "$OUTPUT_ROOT/engine-capacity"
```

Use fresh `it_000000.pkl` weights produced by current official JAX training. That
numbered checkpoint has `iter` and `env_steps` at zero; no older metadata reader
is used. Both interfaces must use exactly the same model, reference data and
explicit inputs in the following native check:

```bash
python studies/pytorch/training_check.py \
  --out "$OUTPUT_ROOT/native-training" --source-revision "$SOURCE_REVISION" \
  --reference-cache "$MOTION_CACHE_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --engine-root "$ENGINE_ROOT" --engine-base-revision "$ENGINE_BASE_REVISION" \
  --engine-capacity-revision "$ENGINE_REVISION" \
  --capacity-validator studies/pytorch/engine_capacity_check.py \
  --expected-max-contacts 96 --expected-constraint-rows 416 \
  --expected-tape-words 1504 --expected-zone-words 13
python studies/pytorch/graph_update_check.py \
  --out "$OUTPUT_ROOT/compiled-graph" --source-revision "$SOURCE_REVISION" \
  --reference-cache "$MOTION_CACHE_DIR" --initial-checkpoint "$INITIAL_CHECKPOINT" \
  --config configs/ours_torch.args --reference-execution compile --matmul-precision fp32 \
  --engine-root "$ENGINE_ROOT" --engine-base-revision "$ENGINE_BASE_REVISION" \
  --engine-capacity-revision "$ENGINE_REVISION" \
  --capacity-validator studies/pytorch/engine_capacity_check.py \
  --expected-max-contacts 96 --expected-constraint-rows 416 \
  --expected-tape-words 1504 --expected-zone-words 13
```

`training_check.py` compares current JAX and Torch state/observation transitions,
rewards and differentiable windows under shared inputs. `graph_update_check.py`
compares two complete updates, including collector/replay histories, targets,
optimizers, EMA, statistics and RNG. Its `compile` reference isolates graph
capture/storage; an `eager` reference is a separate compiler-plus-capture check.
Run the graph comparison separately with `--matmul-precision tf32` for the default
training condition. Do not substitute a passing precision condition for another.
The existing numerical margins and exact discrete-support checks remain fixed;
all nine native invalid counters must be zero. Keep failures visible.

The bounded graph check uses eight environments and four steps. After it passes,
exercise the complete 4,096-by-64 workload before a long run:

```bash
python -m nablamimic.torch_train --config configs/ours_torch.args \
  --initial-checkpoint "$INITIAL_CHECKPOINT" --matmul-precision highest \
  --iters 12 --no-full-state-at-end --out "$OUTPUT_ROOT/full-smoke"
```

Check finite metrics, all native guards, twelve outer updates, twelve EMA updates,
96 actor and 240 critic optimizer updates, actual graph replay counts and bounded
memory. This smoke run checks execution and memory; it is not a fidelity result
or proof of full-workload compiler equality. Evaluate later trained NumPy weights
with official `nablamimic.evaluate --actor-key actor_ema` or `--actor-key actor`.
