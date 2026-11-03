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
(nets.actor_apply). "anchor_dec" is a shape marker that never enters the computation and is not exported; nor are the
critic, its normaliser, optimiser states or reset-bin statistics.

The declaration states the architecture, the input layout, the action map, the timing and the training domain. Every
value comes from the checkpoint, the run's config.json, the model build_model compiles for that configuration (joint
names, nominal pose, gains, torque limits, setpoint range: the values the policy trained with), the reference manifest
(export_reference_manifest) or a required argument; a missing or inconsistent value is refused, never defaulted.

The run directory is the directory of config.json, and the checkpoint must lie in its ckpt/. train.py writes its end
markers there: DONE, DIVERGED or HALTED. DIVERGED (a non-finite loss, checked every 10 updates on the
logged update) is refused: the checkpoint at that label holds the weights that produced the non-finite loss and the nine
updates before it were not checked. HALTED (the --shac-strict guard) is recorded, not refused: the guard masks the
rejected actor and optimiser update (jnp.where on the guard's flag) before the halt save, so that save holds the weights
after its label's completed updates; only actor-input statistics advance, and actors with them are refused anyway. The
declaration records the markers present (a run resumed after a halt can hold HALTED and DONE).

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

from .env import (ACTION_EFFORT_FRACTION, ANCHOR, BM_BODIES, DT, FUTURE, GOAL_ANCHOR, GOAL_FRAME_DIM, HIST, PROPRIO_DIM,
                  SCALE, SUBSTEPS, TIMESTEP, ActionParameters)
from .export_reference_manifest import read_manifest
from .motion import REFERENCE_FPS
from .params import evaluation_actor

FORMAT = "bm-nabla-tracking-policy-v1"
ANCHOR_WIDTH = GOAL_ANCHOR.stop - GOAL_ANCHOR.start
# The model compiled_action_parameters builds: env.build_model compiles the fixed scene (train.FIXED).
TRAINING_MODEL = dict(robot="mjlab", contacts="all")
# Top-level config.json fields that train.py writes beside the parsed arguments: train.FIXED and its own bookkeeping.
NON_ARGUMENT_FIELDS = frozenset({
    "robot", "obs_mode", "contacts", "reward_frame", "hist_len", "state_select", "matmul_precision", "actor_hidden",
    "actor_latent", "critic_hidden", "devices", "processes", "git_commit", "git_dirty", "gpu", "engine", "segments",
    "argv", "config_args", "config_file"})

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
    """The end markers train.py wrote in the run directory, as {name: content}; refuses a run that diverged."""
    run_dir = Path(run_dir)
    if (run_dir / "DIVERGED").exists():
        raise ValueError(f"{run_dir} holds DIVERGED (non-finite loss at update {(run_dir / 'DIVERGED').read_text()!r}, "
                         "checked every 10 updates): its last checkpoints may follow the divergence; not exported")
    markers = {}
    if (run_dir / "DONE").exists():
        markers["DONE"] = dict(last_update=int((run_dir / "DONE").read_text()))
    if (run_dir / "HALTED").exists():
        markers["HALTED"] = json.loads((run_dir / "HALTED").read_text())
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


def update_counts(checkpoint, cfg, label):
    """(completed outer updates, its source, averaging updates). Checkpoints of this fork record completed_updates and
    ema_updates; released-recipe checkpoints do not, and their count is train.py's own rule: env_steps counts window
    transitions, num_envs * devices * horizon per outer update."""
    if int(checkpoint["iter"]) != label:
        raise ValueError(f"checkpoint iter {checkpoint['iter']} differs from its file label {label}")
    per_update = int(_require(cfg, "num_envs")) * int(_require(cfg, "devices")) * int(_require(cfg, "horizon"))
    env_steps = int(checkpoint["env_steps"])
    if env_steps % per_update:
        raise ValueError(f"env_steps {env_steps} is not a whole number of outer updates of {per_update} transitions")
    derived = env_steps // per_update
    if "completed_updates" in checkpoint:
        completed, source = int(checkpoint["completed_updates"]), "checkpoint_completed_updates"
        if completed != derived:
            raise ValueError(f"completed_updates {completed} differs from env_steps / {per_update} = {derived}")
    else:
        completed, source = derived, "env_steps_over_window_transitions_per_update"
        if completed != label:
            raise ValueError(f"derived completed updates {completed} differ from the checkpoint label {label}")
    if "ema_updates" in checkpoint:
        averaging = checkpoint["ema_updates"]
        if type(averaging) is not int or averaging < 0:
            raise ValueError(f"ema_updates {averaging!r} is not a nonnegative integer")
    elif "actor_ema" in checkpoint or cfg.get("policy_ema", False):
        raise ValueError("the run averages the policy but the checkpoint records no ema_updates")
    else:
        averaging = 0  # the run declares no averaging (no policy_ema) and the checkpoint holds no averaged weights
    return completed, source, averaging


def select_actor(checkpoint, policy):
    """The declared weights through params.evaluation_actor, which refuses an absent or inconsistent EMA."""
    if policy == "ema" and "evaluation_policy" not in checkpoint:
        raise ValueError("the checkpoint declares no evaluation policy (released-recipe checkpoint): it holds live "
                         "weights only; use --policy live")
    return evaluation_actor(checkpoint, policy)


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
            checkpoint_label=label, checkpoint_evaluation_policy=checkpoint.get("evaluation_policy"),
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
            source_run=dict(**run_arguments(cfg), git_commit=_require(cfg, "git_commit"),
                            git_dirty=_require(cfg, "git_dirty"), gpu=_require(cfg, "gpu"))),
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


def write_policy_file(arrays, declaration, out):
    """Write the npz once: a temporary file beside out, then a hard link (refused if out exists)."""
    out = Path(out)
    if out.suffix != ".npz":
        raise ValueError(f"{out}: the policy file must end in .npz")
    if out.exists():
        raise FileExistsError(out)
    if "declaration" in arrays:
        raise ValueError("'declaration' is reserved")
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "xb") as f:
            np.savez(f, **arrays, declaration=np.array(json.dumps(declaration, indent=1)))
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
                    help="the weights to export (params.evaluation_actor: ema refuses an absent or inconsistent EMA)")
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
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.suffix != ".npz" or out.exists():
        ap.error(f"--out must name a new .npz file: {out}")
    label = checkpoint_label(args.checkpoint)
    cfg, cfg_path = load_run_config(args.run_config)
    run_dir = cfg_path.resolve().parent
    if Path(args.checkpoint).resolve().parent != (run_dir / "ckpt").resolve():
        ap.error(f"--checkpoint must lie in the run directory's ckpt/: {run_dir / 'ckpt'}")
    markers = run_markers(run_dir)
    manifest = read_manifest(args.reference_manifest)
    with open(args.checkpoint, "rb") as f:
        checkpoint = pickle.load(f)
    arrays, declaration = build_policy_file(
        checkpoint, cfg, compiled_action_parameters(cfg), manifest, policy=args.policy, label=label,
        artifact_id=args.artifact_id, source_artifact_id=args.source_artifact_id, engine_revision=args.engine_revision,
        contact_capacity=args.engine_contact_capacity, constraint_row_capacity=args.engine_constraint_row_capacity,
        end_markers=markers, source_checkpoint=Path(args.checkpoint).resolve(), source_config=cfg_path.resolve())
    write_policy_file(arrays, declaration, out)
    src = declaration["policy_source"]
    print(f"{out}: {len(arrays)} arrays, {src['selected_weights']} weights after "
          f"{src['counts']['completed_outer_updates']} outer updates ({src['counts']['averaging_updates']} averaging "
          f"updates), baseline {declaration['action']['baseline']['kind']}")


if __name__ == "__main__":
    main()
