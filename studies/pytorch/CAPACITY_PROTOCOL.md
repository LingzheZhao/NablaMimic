# Qualification of 96 contacts and 416 constraint rows

The model, collision pairs, solver, objective and analytic-adjoint options remain
those of the current NablaMimic donor. Capacity qualification is separate from
motion fidelity and convergence. Ninety-six contacts is not a demonstrated upper
bound on every possible training state; overflow must continue to fail the run.

Every condition uses fixed source and dependency revisions and a fresh output
directory. Results retain inputs, failed checks and overflow rows. A condition
with insufficient coverage remains failed. A larger input pool is a new frozen
condition, not a change to the earlier result or its thresholds.

The CPU prerequisite checks the actual current model, 96/416 defaults, capability
validation, 1504-word records, all thirteen zone words, contact slots through 95,
row bits through 415, rejection of unsupported capacity overrides and rejection
of the previous tape width. This prerequisite alone proves no physical behavior.

## Required independent stages

| Stage | Comparison and acceptance |
| --- | --- |
| `baseline64` | Original 64-contact/288-row Engine, actual current model, eighty fresh reference inputs: first and midpoint frames of each of forty clips. Shared friction/COM samples and output cotangents are saved once. Ten Newton iterations, four substeps, `clamp_kv=0`, `normal=3`. Every native invalid counter is zero. |
| `candidate96` | Same input arrays, model constants and cotangents through the 96-contact/416-row Engine; same finite/invalid rules. |
| `comparison` | Input arrays and packed physical model constants match exactly. Physical outputs, decoded continuous tape fields and qpos/qvel/control VJPs satisfy finite `rtol=1e-4`, `atol=1e-5`. Active contact counts, pair/slot order, constraint-row counts, active zone bits and actuator clamp bits match exactly. Tape padding/offsets are not compared across different ABIs. Solver iteration counts are reported separately. |
| `natural_stress` | Actual collision discovery on a separately frozen manufactured-pose pool must produce valid contacts above 64, a valid 96-contact state, and raw counts above 96. One substep makes the raw `info[:,1]` count unambiguous. Contact-overflow bit 4 is present exactly when raw count exceeds 96; no other unsupported bit is allowed. Tape contact counts equal `min(raw,96)`. The first valid world of every available count 65–96 is retained for finite forward/VJP checks with zero invalid counters. All pool rows, including rejected overflow states, remain saved. |
| `highcontact_ffi_graph` | Selected real-contact states run through the same candidate build. Native outputs and VJPs agree bitwise across Torch eager/Inductor, changed-input explicit CUDA Graph replay, and Torch/JAX FFIs. Same-build packed tapes agree bitwise. Only the framework scalar loss reduction uses finite `rtol=1e-4`, `atol=1e-5`. Both control inputs and cotangents change between graph replays. |
| `synthetic_jacobian_hessian` | Collision discovery is replaced by a manufactured sequence through the actual `add_contact` function. The production contact-Jacobian block, contact grouping and Hessian routine remain intact. Counts 0,1,31,32,63,64,65,95,96,97 exercise boundaries and overflow. Both world and self contacts occupy every tile, with mixed active zones. Independently computed MuJoCo point Jacobians and a dense float64 mass-plus-constraint Hessian satisfy finite `rtol=1e-4`, `atol=1e-5`; DOF supports are exact. Contact 95 has a nonzero independent and device Hessian contribution. This manufactured fixture does not replace natural collision coverage. |

The expanded natural pool contains 43,040 states: 1,024 seeded hinge poses with
angles uniform in `[-pi,pi]` and random unit root quaternions, plus 321 opposing
hip/ankle-roll poses on `[-0.8,0.8]` in increments of 0.005. Each pose uses 32 root
heights from -2.0 to 0.9 m. Hinge angles may exceed joint limits and root positions
may deeply penetrate the floor; these are explicit capacity stress conditions.
Velocity and warm start are zero, friction is one, COM offset is zero, and control
setpoints are clipped to the model's existing control range. Execution uses
1,024-world chunks; every world is independent and all raw rows and diagnostic
counts are retained. No dynamics or contact filter changes are made.

The final `capacity_qualification.json` names all six stages above, their output
paths and passed status. It also records the donor source revision, the candidate
Engine capacity revision and the separate baseline revision. The corresponding
launch records retain the fixed image, overlay and harness identities. Top-level `status` must be `passed`
and `failures` empty. Missing stages, coverage or identities fail the aggregate.

The existing same-build native interface checks, production-TF32 compile versus
CUDA Graph training checks, independent collector-history copy invariants and
full-workload smoke test remain separate mandatory prerequisites. The original
eager-versus-compiler numerical comparisons keep their own failed results and are
not reclassified by a passing capture/storage condition.

Run the programs in separate provisioned processes so one process never loads
both native builds. Put the selected donor and Engine source on `PYTHONPATH`;
select the same pinned robot assets with `MENAGERIE_DIR`.

1. `engine_capacity_check.py --cpu` checks the current candidate model and tape.
2. `engine_capacity_probe.py --phase prepare --stress-profile expanded` writes
   the shared reference inputs and the larger stress pool.
3. Run `--phase native` once per declared Engine build with the same saved
   `valid_inputs.npz`, then `--phase compare` on those two fresh output roots.
4. Run `--phase stress --stress-profile expanded` with `stress_inputs.npz`.
5. `engine_capacity_interfaces.py` consumes the passed stress root and checks
   its selected high-contact states across execution paths.
6. `synthetic_capacity_check.py` independently checks the device Jacobian and
   Hessian using manufactured contacts and the current model.

Each program's `--help` gives its required source, revision, input and fresh-output
arguments. The metadata aggregator and execution scheduling are external to the
public donor package. Native iteration-limit, nonconvergence and singular-contact
diagnostics remain recorded. They are not contact-capacity or nonfinite-state
failures; passing this protocol does not assert solver convergence.
