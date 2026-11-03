# Frozen-weight reset and orientation isolation

This short diagnostic tests specific possible implementation differences. It
does not train, tune, reproduce an entire trajectory, estimate a policy failure
rate, or establish the cause of the final training-quality difference.

Run the two Python files together in the pinned provisioned CUDA environment,
with one GPU and the declared donor source, Engine and reference cache. Every run
requires a fresh output directory. The program writes its protocol before
initializing either framework. All failed comparisons and completed conditions
remain in the output.

## Fixed comparisons

Floating values and VJPs must be finite and satisfy
`abs(candidate-reference) <= 1e-5 + 1e-4 * abs(reference)`.
Integer and boolean values must match exactly. Margins are identical for every
precision and checkpoint; there is no exception for small rewards or gradients.

The orientation diagnostic runs first. Each explicitly named observed trace
supplies 64 rows: 32 uniformly spaced eligible rows and 32 additional rows with
the lowest independently computed body-orientation reward. Eligibility requires
finite active current and previous rows, no recorded capacity loss or invalid
lineage, and matching previous frame/current source frame. It records exclusions
and selected row indices. Error-tail enrichment is diagnostic and does not
estimate trajectory-level quality.

1. Both production reward implementations receive the same recorded fourteen
   body quaternions, reference frame and nominal domain context. Compare the
   body-orientation term and its VJP with respect to all body quaternions.
2. Independently compute that yaw-aligned reward in NumPy float64. Central finite
   differences use step 1e-6 for every component of every tracked body's
   quaternion, without renormalizing the ambient-coordinate perturbations.
   Compare both production values and gradients to this oracle.
3. Share post-step qpos/qvel, then use MJX and native Engine kinematics
   independently. Compare body-orientation reward and its qpos VJP, with explicit
   reports for action indices 21, 27 and 28. Actual joint names and coordinate
   addresses are read from the runtime model.
4. Use the preceding row's qpos/qvel and the current row's recorded setpoint for
   one native step in each framework. Warm start is zero because it was not
   recorded; friction is one and COM is zero as in nominal evaluation. This is a
   declared counterfactual, not exact replay. Compare reward and control VJP,
   including the three wrist indices, and require zero native invalidity
   counters and positive forward/backward diagnostic counts.

VJPs use a unit cotangent per world, summing rewards so that batch averaging does
not hide local gradient differences. The orientation diagnostic is FP32; its
trace/checkpoint identities remain separate.

The reset and restore diagnostic uses all forty clips at start, midpoint, end,
and one seeded reset frame each: 160 rows. It runs nominal, domain-randomization
only, and full training reset conditions. Actual JAX reset keys produce explicit
unit draws that are supplied to the actual Torch reset. Each implementation
constructs its own physical state; derived JAX body fields are not copied into
the Torch reset. The raw qpos/qvel/domain fields are themselves compared.

Restore separately shares only compact physical/history inputs. These contain
nonzero warm starts, previous actions, arbitrary continuous histories, episode
ages, reset-style histories, finite fallen states and near-height-threshold
states. Each implementation computes its own derived body fields. All states,
observations, goals, zero-action reward terms, hazards and termination outcomes
are retained.

Each explicitly supplied checkpoint selects live or EMA weights through the
production selector. The initial, final JAX and final Torch checkpoints are never
pooled. Each is tested in FP32 and production TF32 precision, with Torch eager
execution and JAX jit/vmap. Compare encoder, action, critic and reward/control
readouts on each backend's observations. Additional controls share the JAX
observations or action to separate network arithmetic from upstream state
differences. This is not a compile/graph test.

## Invocation

```bash
python reset_observation_check.py \
  --out FRESH_OUTPUT \
  --source-root FIXED_SOURCE --source-revision DECLARED_SOURCE_REVISION \
  --engine-root FIXED_ENGINE --engine-capacity-revision DECLARED_ENGINE_REVISION \
  --reference-cache CURRENT_40_CLIP_CACHE \
  --checkpoint initial live INITIAL_CHECKPOINT \
  --checkpoint jax_final ema FINAL_JAX_CHECKPOINT \
  --checkpoint torch_final ema FINAL_TORCH_CHECKPOINT \
  --orientation-trace jax_final_observed JAX_FIGHT1_OBSERVED_NPZ \
  --orientation-trace torch_final_observed TORCH_FIGHT1_OBSERVED_NPZ
```

The result contains source/environment declarations, checkpoint and trace paths,
runtime joint mapping, all individual checks and failures. Raw inputs and outputs
remain beside that result. A failed numerical check gives a nonzero exit only
after all otherwise runnable conditions have been recorded; an execution
exception records the exception and stops without inventing missing results.
