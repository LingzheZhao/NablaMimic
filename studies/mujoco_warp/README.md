# COM range-sum check

The portable forward-only CUDA check is maintained in the
[pinned SDK fork](https://github.com/LingzheZhao/mujoco_warp/blob/d39d6ad691e63ce94e66df69d509eff42047c2d9/studies/com_position_check.py).
Install the SDK commit selected by `scripts/setup_mujoco_warp.sh`. With its
default checkout location, run in a provisioned CUDA environment:

```bash
python third_party/mujoco_warp_fork/studies/com_position_check.py --out /results/fresh-com-check
```

The output directory must be fresh. The entry needs no NablaEngine, checkpoint
or robot assets. See the [SDK documentation](https://github.com/LingzheZhao/mujoco_warp/blob/d39d6ad691e63ce94e66df69d509eff42047c2d9/NABLAMIMIC.md)
for the fixed formula, repetition checks and dependency conditions. It checks
COM arithmetic, not full-simulator determinism, dynamics of zero-mass models,
gradient correctness or long-rollout quality.
