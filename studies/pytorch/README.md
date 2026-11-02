# PyTorch with the same NablaEngine

This route ports the current EMA recipe to PyTorch while retaining NablaEngine's
native forward dynamics and adjoint. It is an experimental reproduction path;
validation results and convergence must be established before claiming equivalence
or a speed improvement. BM integration is outside this study.

`python -m nablamimic.torch_train --config configs/ours_torch.args --out FRESH_OUTPUT` selects
fullgraph Inductor compilation and explicit CUDA Graphs. Compilation covers the
control body, reference encoder, action targets, TD targets, collector, optimizers
and statistics. Explicit graphs capture the entire differentiable window and
backward, plus separate reusable collector, actor regression, critic and statistics
calls. Compilation/capture errors fail the run. Random inputs are sampled outside
graphs. Warmup and capture mutations are restored before training starts.

Use a fixed committed revision and pinned environment with Torch 2.9.1+cu128 and
Triton 3.5.1. Retain the original JAX, MuJoCo, Warp, robot assets and capacity-corrected
NablaEngine revision. The environment's resolved CUDA libraries must be recorded
and tested: the Torch wheel's declared vendor-library pins alone do not describe
an additive environment. Source and inputs are selected explicitly through
`PYTHONPATH`, `LAFAN1_G1_DIR`, `MOTION_CACHE_DIR` and `MENAGERIE_DIR`. Computational
checks run in a provisioned environment, with fresh output directories.

Validation consists of the CPU numerical tests, `update_check.py` (one complete
mathematical outer update using shared sampling indices), `engine_check.py` (the
native interface, compiler and changed-input CUDA Graph probe), and
`training_check.py` (paired short physical windows). The native probe's frozen
rules are in [ENGINE_PROTOCOL.md](ENGINE_PROTOCOL.md). The physical-window
comparison freezes rtol 1e-4 and atol 1e-5; the synthetic update comparison freezes
rtol 1e-4 and atol 1e-6. Finite values and exact integer counters are required.
A full-size compiled/captured smoke run must additionally demonstrate graph
replays, finite updates, correct optimizer/EMA counts and valid native counters.
`graph_update_check.py` compares two actual randomized TorchTrainer updates in
eager and compiled graph modes at N=8/H=4 with the full networks. Initial state
and RNG/noise are exact; finite float comparisons use rtol 1e-4 and atol 1e-5.
Termination, reset and frame support must match exactly; a matching termination
remains visible and is not itself a transformation failure.

The planned full comparison uses one CMP 170HX, seed 0, all forty 50 Hz LAFAN1
clips, 4,096 training and collector worlds, a 64-control window, 1,000 replay
steps, UTD 0.4, eight actor regressions of 16,384 samples, five critic epochs with
four minibatches, and three hours of training. Solver settings and the EMA recipe
remain those in `ours_ema.args`. Use `--initial-checkpoint` to share the declared
zero-update JAX actor and critic weights, with fresh optimizer and EMA state.
Torch RNG streams are different from JAX even with the same seed label.

The Torch training clock starts after setup, compilation and capture; per-update
sampling, synchronization, diagnostics, logging and weight saves are included.
The original JAX clock excludes some startup compilation but includes lazy critic
compilation, so startup, training and total durations must be reported separately.
Save at the first update after 8, 20, 60, 90 and 180 minutes, plus the normal update
interval. Sampled checkpoint crossings are not exact first-crossing times.

Current-format NumPy policy checkpoints support the existing fixed MuJoCo Warp
original and observed evaluations. Evaluate EMA and live weights separately;
keep all forty clips, failures and time supports visible. Report reward, completion
and physical tracking fidelity separately. A terminal Torch state also records
optimizers, counters, RNG, replay and collector state; resumed execution is
explicitly refused until its restoration path is validated.
