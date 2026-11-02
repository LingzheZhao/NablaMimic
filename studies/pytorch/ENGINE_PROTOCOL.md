# Native Engine and PyTorch execution check

This is a provisioned, single-GPU interface and compiler check. It does not measure
training convergence, motion fidelity, or training throughput. Run only the fixed
committed probe and source with a pinned environment and a fresh output root.

Use the donor G1 model built by `EnvConfig(solver_iterations=10,
residual_action=True)`, with MuJoCo sensors disabled as in the donor Engine setup.
Both interfaces receive the same model object and foot-floor pair ordering.
The corrected native source has 64 retained contacts, 288 rows, no friction-loss
rows, 29 limited joints, a 1,472-word tape record, nine zone words, and packed
contact/row/limit/iteration widths of 7/9/8/8 bits. The source validator, generated
kernel constants, architecture flags, model dimensions and packed-constant byte
agreement are required checks. Native math remains the supplied Engine source.

The input is frame zero and the midpoint of every one of the forty current
50 Hz reference clips, in sorted clip order: 80 worlds. Control is the next
reference joint pose clipped to actuator limits, warm starts are zero, and the
fixed NumPy seed 20261103 selects friction in [0.3, 1.2] and torso COM offsets
within +/-[0.025, 0.05, 0.05] metres. Seed 7 selects fixed float32 cotangents.
Native integration uses four 5 ms substeps and ten solver iterations, with
`clamp_kv=0`, `normal=3`, no soft clamp, and gradient decay 1.

Required comparisons are native forward state/tape/kinematics and full native
VJP; Torch autograd versus JAX VJP; standalone kinematic outputs and VJP; eager
versus fullgraph Inductor outputs and gradients; and explicit native plus
compiled forward/backward CUDA Graph replay after changed inputs. Replay checks
roll batch inputs by one and seven rows; the compiled graphs also receive
changed cotangents. Capture and its warm-up run before initializing JAX CUDA.
`fullgraph=True`, the Inductor backend, and disabled error suppression are
mandatory. Unsupported capture or compilation fails the probe; there is no
eager fallback. Inductor's automatic CUDA graphs are disabled so the explicit
capture is the graph under test.

Native arrays, model constants, outputs, gradients and packed tape require exact
byte equality. Scalar loss reductions alone use rtol 1e-5 and atol 1e-6. All
physical float arrays and gradients must be finite. Tape integer bit fields may
look like nonfinite float32 values, so tape equality is byte based and finiteness
is checked on its decoded continuous fields. Unsupported worlds, contact/row
overflow, nonfinite backward rows/cotangents and replay mismatch counters must
be zero. Solver iteration limits are recorded; they are not silently dropped.

Each run uses private native, JAX, Inductor, Triton and CUDA caches. A declared
read-only JAX library seed can be copied into the new native cache; Torch builds
its own model-specific library there. Compilation uses a short owned `/tmp`
directory, removed after exit. Framework memory fractions default to 0.45 each;
these are allocator settings, not a cap on every native GPU allocation. The
probe records framework memory readings and all failed stages in `result.json`.
Its NPZ artifacts preserve exact inputs, cotangents, native state and gradients.

Example from a provisioned pinned environment, with source, data and additive
Torch overlay already selected explicitly:

```sh
python studies/pytorch/engine_check.py \
  --out "$PROBE_OUTPUT" --reference-cache "$REFERENCE_CACHE" \
  --engine-root "$ENGINE_ROOT" --capacity-validator "$ENGINE_ROOT/validate_rows288.py" \
  --engine-base-revision "$ENGINE_BASE_REVISION" \
  --engine-capacity-revision "$ENGINE_CAPACITY_REVISION"
```

`PYTHONPATH` must select the fixed donor source and immutable Torch overlay.
`LAFAN1_G1_DIR` selects the forty original CSV identities and `MENAGERIE_DIR`
selects the pinned robot assets. `CUDA_VISIBLE_DEVICES` must expose one allocated
CMP 170HX. The expected GPU model can be changed only as a declared run condition.

