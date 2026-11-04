# COM range-sum check

Run `com_position_check.py` in a provisioned CUDA environment after installing
this revision with `scripts/setup_mujoco_warp.sh`. The measured dependency
versions are Warp 1.18.0, NumPy 2.5.4 and the pinned MuJoCo Warp adjoint source
with patches 0001–0005. This small entry needs no NablaEngine, policy checkpoint,
robot assets or scheduler integration.

```bash
python studies/mujoco_warp/com_position_check.py --out /results/fresh-com-check
```

The output directory must be fresh. Five synthetic cases cover branching,
multiple roots, the world's own mass, a zero-mass subtree, zero total mass and
positive subtree/total masses below `MJ_MINVAL`. Body masses are nonnegative,
and subtree masses are consistent with them. These are forward-only arithmetic
cases; they do not assert support for zero-mass dynamics or adjoints.

Each case captures the two COM kernels once and runs sixteen times, overwriting
sentinel outputs. Results must repeat bitwise and match independent FP64 sums
from the actual FP32 mass/position inputs at rtol 1e-5 and atol 1e-6 metres.
The report keeps two denominator references separately: independently summed
masses and the stored FP32 subtree masses. Both preserve the old nonzero-mass
division rule. The world kernel follows upstream's root-COM times root-mass
regrouping, which adds FP32 rounding compared with a direct all-body sum.

Raw arrays and a JSON result are written beside the selected output. No inputs,
physics configuration or package files are modified. This entry tests the COM
formula and repeated component execution; it does not establish full simulator
determinism, gradient correctness or long-rollout quality.
