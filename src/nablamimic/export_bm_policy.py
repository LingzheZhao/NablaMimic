"""Export the actor of a train.py checkpoint as a BM tracking policy file (format "bm-nabla-tracking-policy-v1").

A checkpoint (ckpt/it_N.pkl) pickles optax states and needs this package's environment to load. The policy file holds
only what executing the policy needs, readable with numpy alone (np.load(path, allow_pickle=False)):

  encoder.0.weight ... encoder.8.bias   the goal encoder in torch Linear layout: weight [out, in], bias [out]
  decoder.0.weight ... decoder.8.bias   the decoder, likewise
  declaration                           a 0-d unicode array holding one JSON document (the fields below)

Key mapping. The actor is {"enc": [(w, b), ...], "dec": [(w, b), ...], "anchor_dec": zeros(6)} (nets.actor_init with
anchor_to_decoder=6); mlp_apply computes x @ w + b with ELU between layers, w of shape [in, out]. Layer i of "enc"
becomes encoder.{2i}.weight = w.T and encoder.{2i}.bias = b, i = 0..4, so the indices 0, 2, 4, 6, 8 are the Linear
layers of a flat torch Sequential(Linear, ELU, Linear, ELU, ..., Linear); "dec" maps to decoder.* the same way. The
encoder output is squashed by tanh; the decoder input is [tanh(encoder output), history, anchor orientation]
(nets.actor_apply). "anchor_dec" is a shape marker that never enters the computation and is not exported; nor are
optimiser states or reset-bin statistics. --critic-out writes the critic and its input normaliser to a second npz
(format "bm-nabla-critic-v1"): critic.{0,2,4}.weight/bias in the same torch layout, normalizer.mean and
normalizer.variance as the checkpoint's cnorm holds them (it holds no update count: the declaration states
normalizer_count "absent"), a 0-d "format" entry and a declaration naming the checkpoint, the run, the completed updates,
the declared source revision and the input layout of env.TrackingEnv.critic_obs.

The declaration states the architecture, the input layout, the action map, the timing and the training domain. Every
value comes from the checkpoint, the run's config.json, the model build_model compiles for that configuration (joint
names, nominal pose, gains, torque limits, setpoint range: the values the policy trained with), the reference manifest
(export_reference_manifest) or a required argument; a missing or inconsistent value is refused, never defaulted.

Only the current upstream JAX train.py checkpoint/configuration schema is supported.
The run must be fresh, single-device and not resumed, diverged or guard-halted.
A numbered it_N.pkl is saved before update N: it holds N completed outer updates,
and env_steps must equal N * num_envs * horizon. The native checkpoint has no
completed-update or averaging counter. With actor_ema > 0 it holds actor_ema and,
under this fresh-run contract, that actor has exactly N averaging updates.
No checkpoint fields are added or converted. Torch checkpoints are outside this
command's supported schema.

  python -m nablamimic.export_bm_policy --checkpoint RUN/ckpt/it_011667.pkl --run-config RUN --policy live \\
      --artifact-id POLICY_ID --source-artifact-id CHECKPOINT_ID --reference-manifest MANIFEST.json \\
      --engine-revision ENGINE_REVISION --engine-contact-capacity 64 --engine-constraint-row-capacity 288 --out FILE.npz
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

from .env import (ACTION_EFFORT_FRACTION, ANCHOR, ANCHOR_DIM, BM_BODIES, DT, FUTURE, GOAL_ANCHOR, GOAL_FRAME_DIM, HIST,
                  PROPRIO_DIM, SCALE, SUBSTEPS, TIMESTEP, ActionParameters)
from .export_reference_manifest import read_manifest
from .motion import REFERENCE_FPS

FORMAT = "bm-nabla-tracking-policy-v1"
CRITIC_FORMAT = "bm-nabla-critic-v1"
ANCHOR_WIDTH = GOAL_ANCHOR.stop - GOAL_ANCHOR.start
# The model compiled_action_parameters builds: env.build_model compiles the fixed scene (train.FIXED).
TRAINING_MODEL = dict(robot="mjlab", contacts="all")
# Top-level config.json fields that train.py writes beside the parsed arguments: train.FIXED and its own bookkeeping.
NON_ARGUMENT_FIELDS = frozenset({
    "robot", "obs_mode", "contacts", "reward_frame", "hist_len", "state_select", "matmul_precision", "actor_hidden",
    "actor_latent", "critic_hidden", "devices", "processes", "git_commit", "git_dirty", "gpu", "engine", "segments",
    "argv", "config_args", "config_file"})
# Core fields written by the current upstream 0.2 JAX producer. A known optional
# CLI extension may add records without changing which core fields are required.
CORE_CONFIG_FIELDS = frozenset({
    "act_cot_clip", "act_noise", "action_rate_space", "action_target", "action_target_delta_floor",
    "action_target_lr", "actor_ema", "actor_ema_warmup", "actor_hidden", "actor_latent", "actor_obs_norm",
    "anchor_to_decoder", "ckpt_every", "clip_end_boot", "clips", "collect_envs", "collect_noise", "config",
    "contacts", "critic_epochs", "critic_hidden", "critic_minibatches", "critic_pre_state", "devices",
    "enc_hoist", "gamma", "git_commit", "git_dirty", "gpu", "hist_len", "horizon", "hours", "iters", "lam",
    "lr", "lr_decay_steps", "lr_decay_to", "lr_warmup_steps", "mask_after_done", "matmul_precision",
    "max_episode_steps", "mjw_adj_normal", "mjw_adj_parallel", "mjw_ift_solve", "motion_lib", "no_remat",
    "num_envs", "obs_mode", "out", "physics", "processes", "replay_steps", "reset_sampling",
    "residual_action", "resume", "reward_frame", "robot", "seed", "shac_strict", "solver_iterations",
    "state_select", "term_penalty", "uniform_over", "utd", "v_bound", "ws_shards", "ws_swap",
})

# Per-frame proprioception (env.TrackingEnv.proprio), in order: (name, width, factor applied to the quantity).
PROPRIOCEPTION_TERMS = (
    ("joint_position_minus_nominal_plus_encoder_bias", 29, 1.0),
    ("joint_velocity_scaled", 29, SCALE["joint_vel"]),
    ("root_angular_velocity_in_body_scaled", 3, SCALE["ang_vel"]),
    ("gravity_in_body", 3, 1.0),
    ("previous_executed_action", 29, 1.0),
)
# Per reference row of the goal (env.TrackingEnv._goal_features), in order.
GOAL_TERMS = (
    ("reference_joint_position_minus_nominal", 29, 1.0),
    ("reference_joint_velocity_scaled", 29, SCALE["joint_vel"]),
    ("reference_gravity_in_body", 3, 1.0),
    ("reference_root_angular_velocity_in_body_scaled", 3, SCALE["ang_vel"]),
    ("reference_root_linear_velocity_in_heading_frame", 3, 1.0),
    ("reference_root_height", 1, 1.0),
)
assert sum(w for _, w, _ in PROPRIOCEPTION_TERMS) == PROPRIO_DIM
assert sum(w for _, w, _ in GOAL_TERMS) == GOAL_FRAME_DIM


def _terms(terms):
    return [dict(name=n, width=w, factor=float(f)) for n, w, f in terms]


def _floats32(x):
    """The float32 values the environment uses (TrackingEnv casts with jnp.float32), as exact JSON numbers."""
    return [float(v) for v in np.asarray(x, np.float32)]


def checkpoint_label(path):
    """N of a numbered checkpoint ckpt/it_N.pkl (params.pkl carries a different label and is refused)."""
    m = re.fullmatch(r"it_(\d+)\.pkl", Path(path).name)
    if m is None:
        raise ValueError(f"{path}: expected a numbered checkpoint it_N.pkl")
    return int(m.group(1))


def load_run_config(path):
    """config.json of a run, given the file or the run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text()), path


def training_model(cfg):
    """{robot, contacts} of the run; refused unless the fixed scene env.build_model compiles."""
    model = dict(robot=_require(cfg, "robot"), contacts=_require(cfg, "contacts"))
    if model != TRAINING_MODEL:
        raise ValueError(f"model {model}: the exporter compiles the fixed scene {TRAINING_MODEL} only")
    return model


def run_markers(run_dir):
    """The current trainer's DONE marker; divergence or a guard halt is refused."""
    run_dir = Path(run_dir)
    for name in ("DIVERGED", "HALTED"):
        if (run_dir / name).exists():
            raise ValueError(f"{run_dir} holds {name}; this exporter requires an unhalted, nondiverged run")
    markers = {}
    if (run_dir / "DONE").exists():
        done = (run_dir / "DONE").read_text().strip()
        if not done.isdecimal():
            raise ValueError("DONE must hold the current trainer's nonnegative final iteration")
        markers["DONE"] = dict(last_update=int(done))
    return markers


def run_arguments(cfg):
    """The run's arguments as config.json records them. With --config, train.py records argv (the arguments used,
    without --config, --out and --resume), config_args (those of the file) and config_file; without --config it
    records none of them, and the parsed arguments are the top-level fields other than NON_ARGUMENT_FIELDS."""
    recorded = {k: cfg[k] for k in ("argv", "config_args", "config_file") if k in cfg}
    if "argv" in cfg:
        return dict(arguments_source="argv_recorded_at_first_start_with_config_file", **recorded)
    return dict(arguments_source="parsed_arguments_in_config_json_run_started_without_config_file", **recorded,
                parsed_arguments={k: v for k, v in cfg.items() if k not in NON_ARGUMENT_FIELDS})


def _require(cfg, key):
    if key not in cfg:
        raise ValueError(f"config.json has no {key!r}")
    return cfg[key]


def validate_run_config(cfg):
    """Validate the current upstream JAX producer's complete fresh-run schema."""
    from .train import build_parser

    arguments = {action.dest for action in build_parser()._actions} - {"help"}
    missing = CORE_CONFIG_FIELDS - cfg.keys()
    extra = cfg.keys() - (arguments | NON_ARGUMENT_FIELDS)
    if missing or extra:
        raise ValueError(f"config.json fields: missing {sorted(missing)}, unsupported {sorted(extra)}")
    if cfg["resume"] is not None or cfg["devices"] != 1 or cfg["processes"] != 1 or cfg["shac_strict"]:
        raise ValueError("Export requires a fresh single-device run without resume or the SHAC guard")
    if "segments" in cfg:
        segments = cfg["segments"]
        if not isinstance(segments, list) or len(segments) != 1 or segments[0]["resume"] is not None:
            raise ValueError("Export requires exactly one fresh training segment")
    for key in ("num_envs", "horizon", "devices", "processes"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    decay = cfg["actor_ema"]
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 <= decay < 1:
        raise ValueError("actor_ema must be a finite decay in [0, 1)")
    if type(cfg["actor_ema_warmup"]) is not bool:
        raise ValueError("actor_ema_warmup must be boolean")
    recorded = {"argv", "config_args", "config_file"} & cfg.keys()
    if recorded and recorded != {"argv", "config_args", "config_file"}:
        raise ValueError("The current trainer records argv, config_args and config_file together")
    configured = cfg["config"] is not None
    if configured != bool(recorded) or configured != ("segments" in cfg):
        raise ValueError("Configuration-file and segment records must agree with the current producer")
    if configured:
        if cfg["config_file"] != cfg["config"] or "engine" in cfg:
            raise ValueError("Configured runs record one config file and the engine in their fresh segment")
    elif "engine" not in cfg:
        raise ValueError("A run without a configuration file must record its engine at the top level")


def update_counts(checkpoint, cfg, label):
    """Current numbered checkpoint count; EMA count is derived for a fresh run."""
    validate_run_config(cfg)
    expected = {"actor", "critic", "cnorm", "st_a", "st_c", "bin_q", "iter", "wall_s", "env_steps"}
    if cfg["action_target"]:
        expected.add("st_r")
    if cfg["actor_ema"] > 0:
        expected.add("actor_ema")
    if not isinstance(checkpoint, dict) or checkpoint.keys() != expected:
        raise ValueError(f"Checkpoint must have the current train.py fields {sorted(expected)}")
    for name, value in (("label", label), ("iter", checkpoint["iter"]), ("env_steps", checkpoint["env_steps"])):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    per_update = cfg["num_envs"] * cfg["devices"] * cfg["horizon"]
    if checkpoint["iter"] != label or checkpoint["env_steps"] != label * per_update:
        raise ValueError("Numbered checkpoint label, iter and completed window transitions disagree")
    wall = checkpoint["wall_s"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not np.isfinite(wall) or wall < 0:
        raise ValueError("wall_s must be a finite nonnegative elapsed time")
    return label, "numbered_checkpoint_label_and_env_steps", label if cfg["actor_ema"] > 0 else 0


def select_actor(checkpoint, policy):
    """Explicit native actor or actor_ema selection; an absent EMA is an error."""
    if policy not in ("live", "ema"):
        raise ValueError(f"Unsupported policy selection: {policy}")
    key = "actor_ema" if policy == "ema" else "actor"
    if key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain requested {key} weights")
    return checkpoint[key]


def actor_arrays(actor, cfg, nu):
    """{key: array} in torch Linear layout and the architecture record; refuses another actor structure."""
    if not isinstance(actor, dict) or set(actor) != {"enc", "dec", "anchor_dec"}:
        keys = sorted(actor) if isinstance(actor, dict) else type(actor).__name__
        raise ValueError(f"actor fields {keys}: expected enc, dec and anchor_dec (an actor trained with "
                         "--anchor-to-decoder, without --actor-obs-norm; PPO policies are not supported)")
    if _require(cfg, "anchor_to_decoder") is not True or _require(cfg, "actor_obs_norm") is not False:
        raise ValueError("config.json must declare anchor_to_decoder true and actor_obs_norm false")
    anchor = np.asarray(actor["anchor_dec"])
    if anchor.shape != (ANCHOR_WIDTH,):
        raise ValueError(f"anchor_dec has shape {anchor.shape}, expected ({ANCHOR_WIDTH},)")
    hidden = [int(h) for h in _require(cfg, "actor_hidden")]
    latent = int(_require(cfg, "actor_latent"))
    enc_in, history = FUTURE * GOAL_FRAME_DIM, HIST * PROPRIO_DIM
    dec_in = latent + history + ANCHOR_WIDTH
    widths = {"enc": [enc_in, *hidden, latent], "dec": [dec_in, *hidden, nu]}
    arrays = {}
    for tree, prefix in (("enc", "encoder"), ("dec", "decoder")):
        layers = actor[tree]
        if len(layers) != len(widths[tree]) - 1:
            raise ValueError(f"{tree} has {len(layers)} layers, expected {len(widths[tree]) - 1}")
        for i, layer in enumerate(layers):
            w, b = (np.asarray(x) for x in layer)
            n_in, n_out = widths[tree][i], widths[tree][i + 1]
            if w.shape != (n_in, n_out) or b.shape != (n_out,):
                raise ValueError(f"{tree} layer {i}: kernel {w.shape}, bias {b.shape}, expected ({n_in}, {n_out})")
            for x in (w, b):
                if x.dtype != np.float32 or not np.all(np.isfinite(x)):
                    raise ValueError(f"{tree} layer {i}: parameters must be finite float32, got {x.dtype}")
            arrays[f"{prefix}.{2 * i}.weight"] = np.ascontiguousarray(w.T)
            arrays[f"{prefix}.{2 * i}.bias"] = np.array(b)
    architecture = dict(
        encoder=dict(input=enc_in, hidden=hidden, output=latent, output_squash="tanh",
                     input_content="goal_reference_rows"),
        decoder=dict(inputs=[dict(name="latent", width=latent), dict(name="proprioception_history", width=history),
                             dict(name="anchor_orientation", width=ANCHOR_WIDTH)],
                     input=dec_in, hidden=hidden, output=nu),
        activation="ELU",
        layout="flat_sequential: keys <encoder|decoder>.<N>.weight [out, in] and .bias [out], N = 0, 2, 4, ... the "
               "Linear layers with ELU between them",
        dtype="float32",
        evaluation_matmul_precision="highest")
    return arrays, architecture


def critic_input_terms(nu):
    """The critic input in env.TrackingEnv.critic_obs order: (name, width, content)."""
    return (
        ("proprioception", PROPRIO_DIM, "the actor's per-frame proprioception terms of the current state, unnoised"),
        ("goal", FUTURE * GOAL_FRAME_DIM, "the next FUTURE reference rows' goal features, as the actor's goal rows"),
        ("anchor_orientation", ANCHOR_WIDTH, "first two columns of robot_rotation^T reference_rotation of the torso "
                                             "anchor at the current row (entries 3..8 of anchor_relative_pose)"),
        ("root_error", 4, "root height minus the reference row's root height, then the root linear velocity in the "
                          "yaw (heading) frame"),
        ("anchor_relative_pose", ANCHOR_DIM, "reference torso anchor position in the robot anchor frame (3), then the "
                                             "first two columns of the relative rotation (6)"),
        ("domain_draws", 1 + nu + 3, "(friction - 0.75) / 0.45, encoder bias / 0.01 per joint, torso COM offset / "
                                     "0.05 (the episode's domain randomisation draws)"),
    )


def critic_arrays(checkpoint, cfg, nu):
    """{key: array} for the critic and its input normaliser, and the architecture record; refuses another structure."""
    terms = critic_input_terms(nu)
    width = sum(w for _, w, _ in terms)
    hidden = [int(h) for h in _require(cfg, "critic_hidden")]
    widths = [width, *hidden, 1]
    layers = checkpoint.get("critic")
    if not isinstance(layers, (list, tuple)) or len(layers) != len(widths) - 1:
        raise ValueError(f"critic: expected {len(widths) - 1} layers (widths {widths})")
    arrays = {}
    for i, layer in enumerate(layers):
        w, b = (np.asarray(x) for x in layer)
        if w.shape != (widths[i], widths[i + 1]) or b.shape != (widths[i + 1],):
            raise ValueError(f"critic layer {i}: kernel {w.shape}, bias {b.shape}, expected "
                             f"({widths[i]}, {widths[i + 1]})")
        for x in (w, b):
            if x.dtype != np.float32 or not np.all(np.isfinite(x)):
                raise ValueError(f"critic layer {i}: parameters must be finite float32, got {x.dtype}")
        arrays[f"critic.{2 * i}.weight"] = np.ascontiguousarray(w.T)
        arrays[f"critic.{2 * i}.bias"] = np.array(b)
    norm = checkpoint.get("cnorm")
    if not isinstance(norm, (list, tuple)) or len(norm) != 2:
        raise ValueError("cnorm: expected the pair (mean, variance)")
    for name, x in zip(("mean", "variance"), (np.asarray(v) for v in norm)):
        if x.shape != (width,) or x.dtype != np.float32 or not np.all(np.isfinite(x)):
            raise ValueError(f"critic normaliser {name}: expected finite float32 ({width},), got {x.dtype} {x.shape}")
        arrays[f"normalizer.{name}"] = np.array(x)
    if not np.all(arrays["normalizer.variance"] > 0):
        raise ValueError("critic normaliser variance must be positive")
    return arrays, terms, hidden


def build_critic_file(checkpoint, cfg, *, label, checkpoint_name, run_dir, source_revision, nu):
    """(arrays, declaration) of the critic and its input normaliser (format "bm-nabla-critic-v1")."""
    if not source_revision:
        raise ValueError("a source revision is required")
    recorded = cfg.get("git_commit")
    if recorded and recorded != source_revision:
        raise ValueError(f"source revision {source_revision} differs from config.json's git_commit {recorded}")
    completed, _, _ = update_counts(checkpoint, cfg, label)
    arrays, terms, hidden = critic_arrays(checkpoint, cfg, nu)
    width = sum(w for _, w, _ in terms)
    pre_state = bool(_require(cfg, "critic_pre_state"))
    declaration = dict(
        format=CRITIC_FORMAT,
        checkpoint=str(checkpoint_name), run=str(run_dir), checkpoint_label=label, completed_updates=completed,
        source_revision=source_revision, run_git_commit=recorded,
        training_framework="jax",
        input_layout=[[n, w] for n, w, _ in terms],
        input_terms=[dict(name=n, width=w, content=c) for n, w, c in terms],
        input_width=width, activation="ELU", hidden=hidden, output=1,
        layout="critic.<N>.weight [out, in] and .bias [out], N = 0, 2, 4: the Linear layers of a flat sequence with "
               "ELU between them and none after the last; the input is normalised first",
        dtype="float32", training_matmul_precision=_require(cfg, "matmul_precision"),
        normalizer=dict(
            stored=["mean", "variance"],
            application="clip((x - mean) / sqrt(variance + 1e-5), -10, 10)",
            update="once per outer update on the window's post-step critic inputs: mean <- 0.99 mean + 0.01 "
                   "batch_mean; variance <- 0.99 variance + 0.01 max(batch_mean_of_squares - batch_mean^2, 1e-6)",
            initial="mean 0, variance 1"),
        normalizer_count="absent",
        fitted_inputs="pre_step_state" if pre_state else "post_step_state",
    )
    return arrays, declaration


def check_action_parameters(act):
    if not isinstance(act, ActionParameters):
        raise TypeError("action parameters must be env.ActionParameters")
    n = len(act.joint_names)
    for name in ("nominal", "kp", "effort", "scale", "setpoint_lower", "setpoint_upper"):
        x = np.asarray(getattr(act, name))
        if x.shape != (n,) or not np.all(np.isfinite(x)):
            raise ValueError(f"action parameter {name}: shape {x.shape}, expected finite ({n},)")
    if not np.all(np.asarray(act.setpoint_lower) < np.asarray(act.setpoint_upper)):
        raise ValueError("setpoint_lower must be below setpoint_upper")
    return n


def reference_record(manifest, cfg):
    """The training domain's reference: the manifest's identity and clips, consistent with config.json's clips."""
    if manifest["bodies"] != list(BM_BODIES):
        raise ValueError(f"manifest bodies {manifest['bodies']} differ from the training bodies {list(BM_BODIES)}")
    if float(manifest["frame_rate_hz"]) != REFERENCE_FPS:
        raise ValueError(f"manifest frame rate {manifest['frame_rate_hz']} differs from {REFERENCE_FPS}")
    if _require(cfg, "motion_lib"):
        raise ValueError("runs on a sharded motion library (--motion-lib) are not supported")
    clips = _require(cfg, "clips")
    names = [c["name"] for c in manifest["clips"]]
    if clips == ["all"]:
        if manifest["clip_selection"] != "all":
            raise ValueError("the run trained on all clips; the manifest lists a selection")
    elif names != list(clips):
        raise ValueError(f"the manifest's clips {names} differ from the run's clips {clips} (order included)")
    return dict(artifact_id=manifest["artifact_id"], frame_rate_hz=float(manifest["frame_rate_hz"]),
                bodies=manifest["bodies"], clip_selection=manifest["clip_selection"],
                clip_order=manifest["clip_order"], clips=names, frames=[c["frames"] for c in manifest["clips"]])


def source_run_record(cfg):
    """Arguments and source state declared by the current JAX trainer."""
    record = dict(**run_arguments(cfg), training_framework="jax", git_commit=_require(cfg, "git_commit"),
                  git_dirty=_require(cfg, "git_dirty"))
    record["gpu"] = _require(cfg, "gpu")
    return record


def engine_records(cfg):
    """The engine identities config.json records (train.py writes them at the top level, or per segment with
    --config); refused when there are none."""
    records = [s["engine"] for s in cfg.get("segments", []) if "engine" in s]
    if "engine" in cfg:
        records.insert(0, cfg["engine"])
    if not records:
        raise ValueError("config.json records no engine identity")
    return records


def build_policy_file(checkpoint, cfg, act, manifest, *, policy, label, artifact_id, source_artifact_id,
                      engine_revision, contact_capacity, constraint_row_capacity, end_markers, source_checkpoint="",
                      source_config=""):
    """(arrays, declaration) for one checkpoint; pure apart from its arguments."""
    for name, value in (("artifact id", artifact_id), ("source artifact id", source_artifact_id),
                        ("engine revision", engine_revision)):
        if not value:
            raise ValueError(f"a {name} is required")
    for name, value in (("contact capacity", contact_capacity), ("constraint row capacity", constraint_row_capacity)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"the engine {name} must be a positive integer")
    if _require(cfg, "physics") != "engine":
        raise ValueError(f"physics {cfg['physics']!r}: only NablaEngine-trained runs are exported")
    model = training_model(cfg)
    if not isinstance(end_markers, dict):
        raise TypeError("end_markers must be the mapping run_markers() returns")
    if end_markers.keys() - {"DONE"}:
        raise ValueError("Only an unhalted, nondiverged run can be exported")
    if "DONE" in end_markers and label > end_markers["DONE"]["last_update"]:
        raise ValueError("Checkpoint label is later than the run's declared final update")
    if int(_require(cfg, "hist_len")) != HIST or _require(cfg, "obs_mode") != "anchor_ori":
        raise ValueError("config.json must declare hist_len 10 and obs_mode anchor_ori (the inputs of env.py)")
    residual = _require(cfg, "residual_action")
    if type(residual) is not bool:
        raise ValueError(f"residual_action {residual!r} is not a boolean")
    nu = check_action_parameters(act)
    completed, count_source, averaging = update_counts(checkpoint, cfg, label)
    arrays, architecture = actor_arrays(select_actor(checkpoint, policy), cfg, nu)
    baseline = dict(kind="next_reference_joint_position", row="min(current_row + 1, clip_last_row)") if residual \
        else dict(kind="nominal_joint_position")
    declaration = dict(
        format=FORMAT,
        artifact_id=artifact_id,
        policy_source=dict(
            kind="imported_nablamimic", source_artifact_id=source_artifact_id,
            counts=dict(completed_outer_updates=completed, averaging_updates=averaging),
            completed_outer_updates_source=count_source,
            selected_weights="averaged" if policy == "ema" else "live",
            checkpoint_label=label, checkpoint_evaluation_policy=None,
            averaging=dict(decay=cfg["actor_ema"], warmup=cfg["actor_ema_warmup"],
                           count_source="fresh_run_one_average_per_completed_outer_update"),
            source_checkpoint=str(source_checkpoint), source_run_configuration=str(source_config),
            run_end_markers=end_markers),
        architecture=architecture,
        inputs=dict(
            proprioception=dict(terms=_terms(PROPRIOCEPTION_TERMS), width=PROPRIO_DIM,
                                root_body="pelvis", quaternion_order="wxyz"),
            history=dict(frames=HIST, order="frame_major_oldest_first", update="shift_append_newest_last",
                         reset_fill="first_frame_tiled_with_previous_action_zero"),
            previous_action="executed_with_exploration_noise",
            goal=dict(terms=_terms(GOAL_TERMS), row_width=GOAL_FRAME_DIM, rows=FUTURE,
                      row_index="min(current_row + k, clip_last_row), k = 0 .. rows - 1",
                      past_clip_end="last_row_hold_including_velocities", order="row_major"),
            anchor=dict(body=ANCHOR, relation="robot_rotation_transposed_times_reference_rotation",
                        reference_row="current_row", encoding="first_two_columns",
                        entries=["R00", "R10", "R20", "R01", "R11", "R21"]),
            normalisation="none"),
        action=dict(
            joint_names=list(act.joint_names),
            nominal_joint_position=_floats32(act.nominal),
            scale=_floats32(act.scale),
            scale_rule=f"{ACTION_EFFORT_FRACTION} * effort / kp",
            kp=[float(v) for v in act.kp], effort=[float(v) for v in act.effort],
            baseline=baseline,
            setpoint="clip(baseline + scale * action, setpoint_lower, setpoint_upper)",
            setpoint_lower=_floats32(act.setpoint_lower), setpoint_upper=_floats32(act.setpoint_upper),
            encoder_bias_in_setpoint=False,
            values="float32 values the environment uses; kp and effort as the compiled model holds them"),
        timing=dict(control_seconds=DT, physics_seconds=TIMESTEP, substeps=SUBSTEPS,
                    solver_iterations=int(_require(cfg, "solver_iterations"))),
        training_domain=dict(
            physics="NablaEngine",
            engine=dict(declared_revision=engine_revision, reported=engine_records(cfg),
                        contact_capacity=contact_capacity, constraint_row_capacity=constraint_row_capacity),
            model=model,
            reference=reference_record(manifest, cfg),
            training_matmul_precision=_require(cfg, "matmul_precision"),
            source_run=source_run_record(cfg)),
    )
    return arrays, declaration


def compiled_action_parameters(cfg):
    """ActionParameters of the model the run trained on (env.build_model of its configuration, which always compiles
    the fixed scene: other robots or contact sets are refused)."""
    from .env import EnvConfig, action_parameters, build_model
    training_model(cfg)
    m, kp = build_model(EnvConfig(solver_iterations=int(_require(cfg, "solver_iterations")),
                                  residual_action=bool(_require(cfg, "residual_action")),
                                  action_rate_space=_require(cfg, "action_rate_space")))
    if m.opt.iterations != int(cfg["solver_iterations"]) or m.opt.timestep != TIMESTEP:
        raise ValueError("compiled model iterations or timestep differ from the run's")
    return action_parameters(m, kp)


def write_policy_file(arrays, declaration, out, *, file_format=None):
    """Write the npz once: a temporary file beside out, then a hard link (refused if out exists). file_format adds a
    0-d "format" entry (the critic file has one; the policy file states its format in the declaration only)."""
    out = Path(out)
    if out.suffix != ".npz":
        raise ValueError(f"{out}: the policy file must end in .npz")
    if out.exists():
        raise FileExistsError(out)
    if "declaration" in arrays or "format" in arrays:
        raise ValueError("'declaration' and 'format' are reserved")
    extra = {} if file_format is None else dict(format=np.array(file_format))
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "xb") as f:
            np.savez(f, **arrays, **extra, declaration=np.array(json.dumps(declaration, indent=1)))
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nablamimic.export_bm_policy",
                                 description="Export a train.py checkpoint's actor as a BM tracking policy file "
                                             f"({FORMAT}): torch-layout weights and a JSON declaration in one npz.")
    ap.add_argument("--checkpoint", required=True, help="RUN/ckpt/it_N.pkl")
    ap.add_argument("--run-config", required=True,
                    help="the run's config.json, or the run directory holding it (the checkpoint must lie in its ckpt/)")
    ap.add_argument("--policy", required=True, choices=["live", "ema"],
                    help="live selects actor; ema selects actor_ema and refuses an absent averaged actor")
    ap.add_argument("--artifact-id", required=True, help="the identity the policy file is published under")
    ap.add_argument("--source-artifact-id", required=True, help="the published identity of the checkpoint")
    ap.add_argument("--reference-manifest", required=True,
                    help="the manifest of the reference cache the run trained on (export_reference_manifest)")
    ap.add_argument("--engine-revision", required=True,
                    help="the declared NablaEngine source revision of the run (config.json does not record one)")
    ap.add_argument("--engine-contact-capacity", type=int, required=True, help="contacts per world of that build")
    ap.add_argument("--engine-constraint-row-capacity", type=int, required=True,
                    help="constraint rows per world of that build")
    ap.add_argument("--out", required=True, help="the policy file (.npz); refused if it exists")
    ap.add_argument("--critic-out", help=f"also write the checkpoint's critic and its input normaliser ({CRITIC_FORMAT},"
                                         " .npz); refused if it exists")
    ap.add_argument("--source-revision", help="with --critic-out: the source revision the run trained with (recorded, "
                                              "refused if config.json records another git_commit)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.suffix != ".npz" or out.exists():
        ap.error(f"--out must name a new .npz file: {out}")
    critic_out = None if args.critic_out is None else Path(args.critic_out)
    if critic_out is not None and (critic_out.suffix != ".npz" or critic_out.exists()
                                   or critic_out.resolve() == out.resolve()):
        ap.error(f"--critic-out must name a new .npz file other than --out: {critic_out}")
    if (critic_out is None) != (args.source_revision is None):
        ap.error("--critic-out and --source-revision go together")
    label = checkpoint_label(args.checkpoint)
    cfg, cfg_path = load_run_config(args.run_config)
    run_dir = cfg_path.resolve().parent
    if Path(args.checkpoint).resolve().parent != (run_dir / "ckpt").resolve():
        ap.error(f"--checkpoint must lie in the run directory's ckpt/: {run_dir / 'ckpt'}")
    validate_run_config(cfg)
    markers = run_markers(run_dir)
    manifest = read_manifest(args.reference_manifest)
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    act = compiled_action_parameters(cfg)
    arrays, declaration = build_policy_file(
        checkpoint, cfg, act, manifest, policy=args.policy, label=label,
        artifact_id=args.artifact_id, source_artifact_id=args.source_artifact_id, engine_revision=args.engine_revision,
        contact_capacity=args.engine_contact_capacity, constraint_row_capacity=args.engine_constraint_row_capacity,
        end_markers=markers, source_checkpoint=Path(args.checkpoint).resolve(), source_config=cfg_path.resolve())
    if critic_out is not None:
        critic, critic_declaration = build_critic_file(
            checkpoint, cfg, label=label, checkpoint_name=Path(args.checkpoint).name, run_dir=run_dir,
            source_revision=args.source_revision, nu=len(act.joint_names))
    write_policy_file(arrays, declaration, out)
    if critic_out is not None:
        write_policy_file(critic, critic_declaration, critic_out, file_format=CRITIC_FORMAT)
        print(f"{critic_out}: critic {critic_declaration['input_width']}-{critic_declaration['hidden']}-1 after "
              f"{critic_declaration['completed_updates']} outer updates, normaliser mean and variance, no count")
    src = declaration["policy_source"]
    print(f"{out}: {len(arrays)} arrays, {src['selected_weights']} weights after "
          f"{src['counts']['completed_outer_updates']} outer updates ({src['counts']['averaging_updates']} averaging "
          f"updates), baseline {declaration['action']['baseline']['kind']}")


if __name__ == "__main__":
    main()
