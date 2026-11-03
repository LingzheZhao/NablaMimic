# Decoder position feedback

The optional decoder input includes the current reference torso position minus
the measured torso position, expressed in the measured torso frame. It precedes
the existing six orientation values. The motion encoder still receives only
the 680 reference features; the critic keeps its 825 inputs. Reward, termination,
residual joint targets and the original orientation-noise draw are unchanged.
The three position values receive uniform observation noise of half-width
0.25 m from a separately folded key.

Start a fresh run with the current upstream recipe:

```bash
python -m nablamimic.train --config configs/ours_position_feedback.args --out runs/position_s0
```

The configuration includes `ours.args`, adding only
`--decoder-position-feedback`. It therefore uses the upstream EMA settings
(`--actor-ema 0.999 --actor-ema-warmup`), the unmasked objective and 100-update
learning-rate warmup. The option requires `--anchor-to-decoder`.

Checkpoints retain the official schema without an added feedback field.
Configuration `obs_mode` explicitly names `anchor_ori` or `anchor_pos_ori`;
evaluation rejects missing or unknown modes and checks the live and averaged
actors' six- or nine-value decoder markers and first-layer input widths.
The optional CLI record `decoder_position_feedback`, when present, must agree
with `obs_mode`. A current official baseline therefore uses its existing
configuration and native actor layout directly. Resume checks the loaded actor
layout against the current CLI condition. Native iteration and EMA representation
are unchanged.

The upstream evaluator selects averaged weights by default when they exist.
Use separate output files when explicitly comparing weight selections:

```bash
python -m nablamimic.evaluate --run runs/position_s0 --iters 250 --actor-key actor_ema --out runs/position_s0/eval_ema.json
python -m nablamimic.evaluate --run runs/position_s0 --iters 250 --actor-key actor --out runs/position_s0/eval_live.json
```

CPU checks use fresh synthetic observations and networks:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_position_feedback.py
```

These checks cover position sensitivity, preserved orientation noise, direct
versus hoisted encoder action and input gradients, unchanged critic/reward/
termination definitions, metadata rejection and inheritance of the current
recipe. They do not establish physical tracking quality on the updated upstream
implementation. No new training or execution measurement accompanies this
source adaptation. This option is not integrated into a Torch trainer, policy
exporter, BONES workflow or BeyondMimic.
