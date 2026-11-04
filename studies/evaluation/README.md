# Saved subtree-sum order analysis

`subtree_sum_orders.py` checks a saved layer with one parent and exactly three
children against all six sequential FP32 child-addition orders. Each addition
rounds to FP32; comparisons use exact bits separately for x, y and z. Warp 1.18
implements vector atomic addition through scalar atomics, so the three
coordinates need not share one child order.

This is an offline CPU arithmetic check, not a simulator or GPU reproducer.
It does not capture inputs, restore native state, classify solver flags or
reproduce the full evaluator experiments. The caller must establish that the
declared children are the only contributors to the declared parent in the saved
layer. Arrays must be native FP32 with shape `[world, body, 3]`, immediately
before and after that layer. At this point `com_pos` holds mass-weighted position
sums, before division by subtree mass.

Run with Python 3.12+ and NumPy (the measured environment uses NumPy 2.5.4):

```bash
python studies/evaluation/subtree_sum_orders.py \
  --before /data/rep-000/after_tree_depth_003.npy /data/rep-001/after_tree_depth_003.npy \
  --after /data/rep-000/after_tree_depth_002.npy /data/rep-001/after_tree_depth_002.npy \
  --parent 1 --children 2 8 14 \
  --source-revision <checked-out-revision> --out /results/fresh-order-analysis
```

Supply before/after lists in matching repetition order. The output directory
must be fresh and separate from inputs. The JSON retains each candidate's
value/bits and all matching orders per coordinate, including ambiguous matches.
It also checks that all pre-layer arrays agree bitwise across repetitions and
that bodies other than the declared parent remain unchanged. Nonfinite values
fail; subnormal inputs/intermediates prevent a successful match status because
CUDA atomic flushing would require separate analysis. Exit status zero requires
all checks to pass. No input file is modified.

Membership means only that the observed result is compatible with at least one
of these scalar FP32 serializations. It does not identify actual GPU scheduling,
uniquely attribute downstream trajectory/return changes, or estimate a failure
rate from repeated coordinates. Saved experiment arrays remain outside the
repository. See [the report](../../EVALUATION_REPEATABILITY.md) for conditions
and causal limits.

The small tests use freshly constructed arithmetic examples:

```bash
python -m pytest -q tests/test_subtree_sum_orders.py
```
