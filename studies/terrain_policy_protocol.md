# Single-reference terrain height-scan pilot

The input is the screened 537-frame motion_07 jumpsit reference at 50 Hz, with
its matched terrain4 CoACD collision and a pelvis-root physical model. Training
uses the current action-cotangent regression recipe, eight actor updates and EMA,
with the explicit reduced dimensions in configs/terrain_heightscan_*.args.
One eager N32 H8 update measures actual peak CUDA allocation before the N64 H16
1000-update learning condition. Both require a new output root. A run that fails
retains its condition and receives a distinct output root on retry.

The baseline supplies ten future reference frames spaced by one 20 ms control,
covering 0.18 seconds. The reference-path condition spaces them by five controls,
covering 0.9 seconds, and adds each future reference torso's displacement in the
current reference torso's full orientation frame. The normalized-action-domain
condition adds the source's limit of ±20 to that reference-path condition. These
are independent training conditions with distinct observation contracts. Every
checkpoint retains its own condition and separate EMA/live readings.

Every evaluated checkpoint is read separately as EMA and live. Each policy read
executes 20 starts from reference frame zero, one nominal start and 19 deterministic
pose perturbations from generator seed 0. Translation perturbations are uniform
within 0.01 m per axis and rotation perturbations within 0.02 rad per axis. Joint
positions, velocity and physical parameters use their nominal values. Evaluations
retain every trajectory through its first tracking termination or the complete
reference, requesting 536 transitions and allowing at least 600 controls.

For this pilot, a working checkpoint requires at least 18 of 20 complete references,
mean world tracked-body origin error at most 0.10 m and mean torso world-position
error at most 0.10 m within each passing trial. Each trial receives its own decision
before counting the eighteen required passing trials. Partial trials retain their actual
time support and errors. Tracking termination and the task-specific posture-fall reading are separate.
A posture fall is pelvis up-dot below 0.5 or world pelvis height below 0.25 m
for at least 0.2 seconds continuously, measured as eleven consecutive 50 Hz
state samples spanning ten intervals, including the terminal state. An observed
posture fall excludes that trial from the working-policy count. Tracking termination that truncates this
support leaves the posture-fall result unavailable. This upright jumpsit rule
measures posture. The seat interval from 4.5 to 5.5 seconds reports torso/pelvis
world errors and ankle world heights without additional passing thresholds.

Each measured scan read has a paired constant-scan read with the same weights,
initial physical state and requested horizon. The checkpoint's training-time sensor
contract remains fixed while the evaluation condition records the intervention.
Report paired completion and errors on their common supports as well as each full
support. The comparison measures sensitivity to withholding the scan under this
single-reference distribution. Motion phase and reference cues may also predict
terrain, which limits conclusions about general perceptive control.

## Normalized action domain condition

The fresh-seed condition in `configs/terrain_heightscan_action_domain.args` uses
an explicit normalized action limit of 20, following InstinctMJ SuperMimic's
imagined and real collection steps. The final noise- and delta-modified action
is clamped before physics. Its bounded value is shared by the previous-action
history, the existing reward formula and recorded actions. Action-target anchors
use the bounded noise-free policy output, while regression predicts the raw
network output. This prevents saturated raw outputs from copying themselves into
the target. Baseline configurations retain their declared unrestricted domain.

The observed training input ranges and checkpoint variance concentrated in the
previous-action features support investigating raw-action history feedback. The
fixed zero-input 128-step network recurrence reached a maximum action magnitude
of 4.616 in both compared conditions and never activated its command limit. That
diagnostic did not demonstrate divergence. A completed training run and bounded
inputs do not establish the working-policy criterion above.

## Actor target-relative position condition

The independent condition in `configs/terrain_heightscan_anchor_position.args`
appends three target-relative torso-position coordinates to the actor history
input, after its current height scan. It uses the simulator's current physical
torso pose and the reference torso position, expressed in the physical torso's
full orientation frame. The goal encoder and its final six rotation coordinates
retain their layout. The critic already observes this position and receives no
additional copy. With the scan, actor history width is 1010, goal width 716 and
critic width 932. This condition adds simulator-based target-relative feedback
to the height-scan controller. It uses N64 H32 with explicit compiled CUDA graphs.

## Selected checkpoint result

The selected 2500-update EMA checkpoint meets this pilot's rule on terrain4 and
motion_07. With the 77-ray height scan and simulator-based target-relative torso
position, all twenty starts complete the reference and eighteen meet their own
world-body and torso error bounds with no observed posture fall. The nominal
start has mean world-body error 0.05736 m and torso error 0.04153 m. With the same
EMA weights and constant scans, three starts complete and zero meet the rule.
The live weights meet the rule on zero starts.

These readings use N128 H32, compiled CUDA graphs and seed 0, with the exact
[NablaMimic source](https://github.com/LingzheZhao/NablaMimic/tree/1e27f8b)
and the declared Engine495 environment of the run. The selected weights and
readings remain attached to update 2500. At update 3000, the EMA measured-scan
group completes nineteen starts but only two meet the error rule. That later
checkpoint is not selected. Replaying the delivered 2500-update package with its
relocated inputs reproduces all eighty original readings using the same starts. The demonstrated scope is one terrain and one reference with
height-scan and simulator-relative position observations. Ten-motion coverage
and the original depth-camera interface require separate implementation and
measurements.

To reproduce the selected policy's evaluation, use that pinned source and the
run's pinned dependency environment. Provide the resolved training configuration
and its bound reference, physical scene and sensor artifacts at the declared
paths. Evaluate the saved EMA weights with the existing command below. This
command writes EMA and live results separately. Read the `ema` entries for the
selected policy. The current development source has additional required task
parameters, so use the pinned source for this saved configuration.

```bash
python -m nablamimic.torch_terrain_evaluate \
  --config "$TRAIN_CONFIG" \
  --checkpoint "$EMA2500_CHECKPOINT" \
  --execution batch --starts 20 \
  --output "$NEW_EVALUATION_ROOT"
```
