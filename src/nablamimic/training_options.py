"""Shared CLI and declared recipe constants, independent of simulation backends."""
import argparse

FIXED = dict(robot="mjlab", obs_mode="anchor_ori", contacts="all", reward_frame="world", hist_len=10,
             state_select="live", matmul_precision="nets_tf32", actor_hidden=[2048, 1024, 512, 256],
             actor_latent=32, critic_hidden=[256, 128])

def build_parser():
    ap = argparse.ArgumentParser(prog="python -m nablamimic.train",
                                 description="Train NablaMimic or the SHAC baseline (see configs/).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="take the training arguments from this configuration file (overrides may follow on the first "
                         "start); they are locked into OUT/config.json with the source revision")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from, or 'latest' (the newest OUT/ckpt/it_*.pkl): actor, critic, "
                         "normalizer, optimizer and reset-bin state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", nargs="+", default=["all"], help="LAFAN1 clip names, or all")
    ap.add_argument("--motion-lib", default=None, metavar="DIR",
                    help="train on a sharded reference library (nablamimic.bones.convert output) instead of --clips: "
                         "a working set of --ws-shards shards (each a random sample of the library) is on the GPU and is "
                         "replaced by the next shards, in a fixed random order, every --ws-swap updates; the replay and "
                         "the reset-bin statistics restart with each working set")
    ap.add_argument("--ws-shards", type=int, default=4)
    ap.add_argument("--ws-swap", type=int, default=1000)
    ap.add_argument("--num-envs", type=int, default=4096, help="BPTT windows per update")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--iters", type=int, default=400000)
    ap.add_argument("--hours", type=float, default=None,
                    help="wall-clock budget (training time, as log.jsonl's wall_s): stop at the first update after it "
                         "with a last checkpoint")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95, help="TD(lambda) of the critic targets")
    ap.add_argument("--lr", type=float, default=2e-4, help="actor (without --action-target-lr) and critic learning rate")
    ap.add_argument("--critic-epochs", type=int, default=5)
    ap.add_argument("--critic-minibatches", type=int, default=4)
    ap.add_argument("--v-bound", action="store_true",
                    help="clip the critic's values where they bootstrap (window ends, episode ends inside a window, "
                         "hence also the critic's lambda-return targets) to +-g_max, the largest possible discounted "
                         "return: stops a value runaway that the window objective can feed (BONES-SEED)")
    ap.add_argument("--clip-end-boot", default="next", choices=["next", "pre"],
                    help="value an episode end at a clip end bootstraps with: next = V of the clip's last frame (goal "
                         "window frozen at it, never a critic training input); pre = V of the transition's start state")
    ap.add_argument("--shac-strict", action="store_true",
                    help="SHAC with its published update rule and gradient guard: the actor update clips only the "
                         "global norm (1.0), no element-wise clip; at the first update whose pre-clip actor gradient "
                         "norm is non-finite or > 1e6 (SHAC's own guard, which raises) the update is not applied, a "
                         "checkpoint and OUT/HALTED are written and training stops; with the direct actor gradient "
                         "only (no --action-target)")
    ap.add_argument("--max-episode-steps", type=int, default=500)
    ap.add_argument("--physics", default="engine", choices=["engine", "warp"],
                    help="engine: NablaEngine, the fused differentiable engine (installed separately); warp: MuJoCo "
                         "Warp with its analytic adjoint (patched, scripts/setup_mujoco_warp.sh)")
    ap.add_argument("--solver-iterations", type=int, default=10, help="Newton iterations of the training physics")
    ap.add_argument("--no-remat", action="store_true",
                    help="keep every step's residuals instead of rematerializing the window in the backward")
    ap.add_argument("--mjw-adj-normal", type=int, default=0, choices=[0, 1, 2, 3],
                    help="warp: derivative of the sphere / capsule self-contact normal v/|v| in the backward: 0 exact, "
                         "1 constant, 2 regularized 1/max(|v|, 1 mm), 3 constant where |v| < 1 mm (mjw_options)")
    ap.add_argument("--mjw-adj-parallel", type=int, default=0, choices=[0, 1],
                    help="warp: near-parallel capsule pairs keep their segment parameters in the backward")
    ap.add_argument("--mjw-ift-solve", type=int, default=0, choices=[0, 1, 2],
                    help="warp: IFT solve of patch 0004 (1: scaled right-hand side, 2: no augmented border)")
    ap.add_argument("--actor-obs-norm", action="store_true",
                    help="normalize the actor input [history, goal] by running statistics (EMA 0.01 per update)")
    ap.add_argument("--anchor-to-decoder", action="store_true",
                    help="the anchor-relative orientation goes to the decoder beside the history; the motion encoder "
                         "sees only the future reference frames")
    ap.add_argument("--enc-hoist", default="off", choices=["off", "on"],
                    help="with --anchor-to-decoder: the encoder outputs of a window in two batched calls before the "
                         "scan instead of once per step (the same policy, faster)")
    ap.add_argument("--action-target", type=float, nargs=3, default=None, metavar=("M", "K", "DELTA"),
                    help="instead of one gradient step per update: from each window take the per-sample action "
                         "gradient dJ/da_t (clipped as --act-cot-clip), form target actions a + eta*g with eta set so "
                         "the RMS setpoint change is DELTA rad, keep the last K windows' (observation, target) in a GPU "
                         "ring buffer and take M regression minibatch steps per update (PODS-style)")
    ap.add_argument("--action-target-lr", type=float, default=None,
                    help="learning rate of the regression steps' Adam (same clipping, betas and schedule as the actor's)")
    ap.add_argument("--action-target-delta-floor", type=float, default=1.0,
                    help="<1 with --lr-decay-steps: the step DELTA follows the same cosine, down to this fraction")
    ap.add_argument("--policy-ema", action="store_true",
                    help="average encoder and decoder once after each outer action-target update; use EMA weights "
                         "for evaluation, live weights for training. Fresh runs only: exact training-state resume "
                         "is not implemented. First decay uses n=0, with n the preceding EMA update count")
    ap.add_argument("--critic-pre-state", action="store_true",
                    help="fit the critic on the state each transition starts from (whose lambda-return the target is); "
                         "default: the state after the transition")
    ap.add_argument("--term-penalty", type=float, nargs=3, default=None, metavar=("W", "DZ0", "DTILT0"),
                    help="add -W*dt*[(relu(dz-DZ0)/(0.25-DZ0))^2 + (relu(dtilt-DTILT0)/(0.8-DTILT0))^2] to the training "
                         "reward (dz, dtilt: the termination test's height and tilt errors); evaluation unchanged")
    ap.add_argument("--mask-after-done", action="store_true",
                    help="an episode end inside a window ends that env's contribution to the actor objective")
    ap.add_argument("--act-noise", type=float, default=0.0,
                    help="Gaussian action noise (action units) executed in the training windows")
    ap.add_argument("--act-cot-clip", type=float, default=None,
                    help="per-sample clipping of the action cotangent to this many times the batch median, in the "
                         "window's backward at every step; with --action-target also on the targets")
    ap.add_argument("--residual-action", action="store_true",
                    help="joint setpoints = the next reference frame's joint angles + scale * action (instead of the "
                         "default pose q0 + scale * action)")
    ap.add_argument("--action-rate-space", default="setpoint", choices=["setpoint", "raw"],
                    help="the reward's action-rate term: on the commanded unclipped setpoints over the action "
                         "scale (setpoint: with --residual-action the absolute-action term of the same setpoints, the "
                         "first step after a reset against the default pose), or on the raw policy outputs (raw); the "
                         "same for absolute actions. The evaluators use setpoint")
    ap.add_argument("--reset-sampling", default="uniform", choices=["uniform", "supermimic"],
                    help="supermimic: 1 s bins weighted by the discounted (lambda-)return from them, mixing 50%% "
                         "uniform, 40%% softmax(-25 G_b/G_max), 10%% the bins' slow-moving past softmax weight")
    ap.add_argument("--uniform-over", default="frames", choices=["frames", "bins"],
                    help="the uniform part of the reset distribution: over valid frames, or over 1 s bins")
    ap.add_argument("--replay-steps", type=int, default=0,
                    help=">0: SuperMimic-style data collection: --collect-envs envs run the policy with "
                         "N(0, collect-noise^2) action noise and the reset sampling, 1/utd steps per update on average, "
                         "into a FIFO replay of replay-steps x collect-envs states; every update's windows start from "
                         "states drawn uniformly from the filled part of the replay")
    ap.add_argument("--collect-envs", type=int, default=0, help="0: same as --num-envs")
    ap.add_argument("--collect-noise", type=float, default=0.1)
    ap.add_argument("--utd", type=float, default=0.4, help="policy updates per collector step")
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help=">0: cosine decay of the learning rates to --lr-decay-to x lr over this many updates, then "
                         "constant")
    ap.add_argument("--lr-decay-to", type=float, default=0.03)
    ap.add_argument("--lr-warmup-steps", type=int, default=0,
                    help=">0 with --lr-decay-steps: linear warmup from 0 over this many updates, then the cosine decay "
                         "(warmup included in --lr-decay-steps)")
    ap.add_argument("--ckpt-every", type=int, default=250, help="numbered checkpoints for the evaluation")
    return ap
