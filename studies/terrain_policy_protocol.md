# Single-reference terrain height-scan pilot

The input is the screened 537-frame motion_07 jumpsit reference at 50 Hz, with
its matched terrain4 CoACD collision and a pelvis-root physical model. Training
uses the current action-cotangent regression recipe, eight actor updates and EMA,
with the explicit reduced dimensions in configs/terrain_heightscan_*.args.
One eager N32 H8 update measures actual peak CUDA allocation before the N64 H16
1000-update learning condition. Both require a new output root. A run that fails
retains its condition and receives a distinct output root on retry.

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
