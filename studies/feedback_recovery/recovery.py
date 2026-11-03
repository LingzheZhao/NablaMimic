"""Frozen-policy XY recovery measurements from shared, freshly executed donor states."""

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic import nets, sim
from nablamimic.env import BM_BODIES, DT, W, EnvConfig, TrackingEnv
from nablamimic.evaluate import validate_feedback
from nablamimic.motion import all_clip_names, load_library
from nablamimic.params import evaluation_actor

HERE = Path(__file__).resolve().parent
INSTRUMENTATION = HERE.parent / "evaluation" / "instrumentation"


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    if settings["paired_routes"] != ["encoder", "decoder"] or settings["policy"] not in ("ema", "live"):
        raise ValueError("Select paired encoder/decoder routes and a supported policy")
    for name in ("checkpoint_update", "prefix_controls", "maximum_controls"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    offsets = settings["world_x_offsets_m"]
    if not offsets or offsets[0] != 0.0 or len(set(offsets)) != len(offsets) or not np.isfinite(offsets).all():
        raise ValueError("Select unique finite world-X offsets starting with sham zero")
    seeds = settings["training_seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Select nonnegative integer training seeds")
    times = settings["readout_seconds"]
    if not times or not np.isfinite(times).all() or min(times) < DT:
        raise ValueError("Select finite readout times of at least one control")
    if any(not np.isclose(seconds / DT, round(seconds / DT), rtol=0, atol=1e-6) for seconds in times):
        raise ValueError("Readout times must lie on the native control clock")
    return settings


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, INSTRUMENTATION / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def translated_compact(compact, displacement):
    if not np.isfinite(displacement):
        raise ValueError("World X displacement must be finite")
    return {**compact, "qpos": compact["qpos"].at[..., 0].add(displacement)}


def check_branch_inputs(env, sham_state, branch_state, displacement):
    """Require matching policy/physical inputs after the shared restoration path."""
    expected = translated_compact(env.compact(sham_state), displacement)
    actual = env.compact(branch_state)
    for name in expected:
        left = jax.tree_util.tree_leaves(expected[name])
        right = jax.tree_util.tree_leaves(actual[name])
        if len(left) != len(right):
            raise ValueError(f"Branch state structure differs at {name}")
        for x, y in zip(left, right):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=f"Branch input differs at {name}")
    keys = jax.random.split(jax.random.PRNGKey(0), 1)
    _, sham_goal = jax.vmap(env.policy_obs)(sham_state, keys)
    _, branch_goal = jax.vmap(env.policy_obs)(branch_state, keys)
    np.testing.assert_array_equal(np.asarray(sham_goal[..., :-9]), np.asarray(branch_goal[..., :-9]))
    np.testing.assert_array_equal(np.asarray(sham_goal[..., -6:]), np.asarray(branch_goal[..., -6:]))


def planned_controls(frame, last, maximum):
    return min(maximum, max(0, int(last) - int(frame)))


def summarize(valid, terminated, planned, remaining, maximum):
    valid = np.asarray(valid, bool)
    terminated = np.asarray(terminated, bool)
    executed = len(valid)
    if valid.shape != terminated.shape or executed > planned:
        raise ValueError("Trajectory support differs from planned controls")
    native_termination = bool(terminated.any())
    reference_end_reached = bool(executed == remaining and executed > 0)
    reference_completed = bool(reference_end_reached and valid.all() and not native_termination)
    return dict(planned_controls=int(planned), executed_controls=executed,
                valid_controls=int(valid.sum()), native_termination=native_termination,
                reference_completed=reference_completed, reference_end_reached=reference_end_reached,
                full_requested_horizon_metrics_available=bool(executed == maximum and valid.all()),
                requested_horizon_seconds=maximum * DT,
                end=("native_termination" if native_termination else "invalid_physics" if not valid.all()
                     else "reference_completed" if reference_completed else "maximum_controls" if executed == maximum
                     else "partial"))


def paired_diagnostic(sham, perturbed, maximum, readout_seconds):
    """Describe own-checkpoint excess errors on a visible, matched physical prefix."""
    available = min(len(sham), len(perturbed))
    support = 0
    for index in range(available):
        if int(sham[index]["frame"][0]) != int(perturbed[index]["frame"][0]):
            raise ValueError("Paired continuations have different reference frames")
        if not (sham[index]["valid"][0] and perturbed[index]["valid"][0]):
            break
        support += 1
    report = dict(shared_valid_prefix_controls=support, sham_executed_controls=len(sham),
                  perturbed_executed_controls=len(perturbed), full_requested_horizon_available=support == maximum, requested_horizon_seconds=maximum * DT)
    fields = {"world_body_mpjpe_m": 0, "donor_aligned_body_mpjpe_m": 1,
              "torso_xy_m": 6, "torso_abs_yaw_rad": 9}
    report["excess_errors"] = {}
    for name, index in fields.items():
        values = np.asarray([float(perturbed[t]["metrics"][0, index] - sham[t]["metrics"][0, index])
                             for t in range(support)])
        report["excess_errors"][name] = dict(
            shared_prefix_mean=float(values.mean()) if support else None,
            shared_prefix_integral=float(values.sum() * DT) if support else None,
            at_seconds={str(seconds): float(values[int(round(seconds / DT)) - 1])
                        if support >= int(round(seconds / DT)) else None for seconds in readout_seconds})
    return report


def policy_probe(actor, env, state):
    hist, goal = jax.vmap(env.policy_obs)(state, jax.random.split(jax.random.PRNGKey(0), 1))
    k = actor["anchor_dec"].shape[-1]
    latent = nets.actor_encode(actor, goal[..., :-k])
    action = nets.actor_decode(actor, latent, hist, goal[..., -k:])
    return dict(history=hist, goal=goal, latent=latent, action=action,
                setpoint=jax.vmap(env.ctrl_of)(action, state.frame))


def make_transition(env, record):
    @jax.jit
    def transition(actor, state, tables):
        e = env.bind(tables)
        functions = sim.vmapped(env, tables)
        probe = policy_probe(actor, e, state)
        next_state = functions["step"](state, probe["action"])
        _, terms = functions["reward"](next_state, probe["action"], state.prev_a, prev=state)
        terminated = functions["terminated"](next_state)
        values = jnp.stack([terms[name] for name in record["terms"]], -1)
        trace = record["record"](e, next_state, jnp.ones(1, bool), terminated, state, values)
        trace.update(latent=probe["latent"], policy_action=probe["action"],
                     position_feedback=probe["goal"][..., -9:-6], orientation_feedback=probe["goal"][..., -6:])
        return next_state, trace

    return transition


def capacity_record(audit, slots):
    """Read an already observed refresh (one slot) or control (four slots)."""
    from mujoco_warp._src.types import OverflowType
    count = int(audit.cursor.numpy()[0])
    if count != slots or bool(audit.overrun.numpy()[0]) or audit.nnz_request is None:
        raise RuntimeError("Capacity observation support unavailable")
    d = audit.sim.df
    contacts = audit.contacts.numpy()[:slots]
    rows = audit.rows.numpy()[:slots]
    nnz = audit.nnz.numpy()[:slots]
    flags = audit.flags.numpy()[:slots]
    diagnostic = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
    valid = bool(not np.any(contacts > d.naconmax) and not np.any(rows >= d.njmax)
                 and (not audit.sim.m.is_sparse or not np.any(nnz > d.njmax_nnz))
                 and not np.any(flags & ~diagnostic))
    return dict(valid=valid, requested_contacts=contacts, requested_rows=rows,
                requested_nnz=nnz, overflow_flags=flags)


def native_refresh(physics, snapshot, compact, audit):
    """Identical restore/refresh operation for every displacement, including sham."""
    import warp as wp
    from nablamimic.warp_physics import mjw
    snapshot.restore()
    native = physics.sim
    d = native.df
    # Restore qvel/warm explicitly as well as coordinates; ctrl and native time come from the snapshot.
    buffers = []
    for name in ("qpos", "qvel", "qacc_warmstart"):
        value = np.asarray(compact[name], np.float32)
        source = wp.array(value, dtype=wp.float32, device=d.qpos.device)
        buffers.append(source)
        wp.copy(getattr(d, name), source)
    xi = compact["xi"]
    friction = wp.array(np.asarray(xi.friction), dtype=wp.float32, device=d.qpos.device)
    com = wp.array(np.asarray(xi.com), dtype=wp.float32, device=d.qpos.device)
    native._dr(friction, com)
    warm, time = d.qacc_warmstart.numpy().copy(), d.time.numpy().copy()
    audit.reset()
    audit.nnz_request = None
    mjw.forward(native.m, d)
    audit.append(d)
    wp.synchronize_device(d.qpos.device)
    np.testing.assert_array_equal(d.qacc_warmstart.numpy(), warm)
    np.testing.assert_array_equal(d.time.numpy(), time)
    return capacity_record(audit, 1)


def write_trace(path, rows):
    if rows:
        np.savez_compressed(path, **{key: np.concatenate([row[key] for row in rows], axis=0)
                                   for key in rows[0]})
    else:
        np.savez_compressed(path, frame=np.empty(0, np.int32), valid=np.empty(0, bool))


def advance(actor, state, count, transition, audit, trace_path):
    rows = []
    try:
        for _ in range(count):
            audit.reset()
            state, trace = transition(actor, state)
            state, trace = jax.block_until_ready((state, trace))
            row = {key: np.asarray(value) for key, value in trace.items()}
            row.update(valid=np.array([False]), capacity_observed=np.array([False]),
                       requested_contacts=np.full((1, 4), -1, np.int32),
                       requested_rows=np.full((1, 4), -1, np.int32),
                       requested_nnz=np.full((1, 4), -1, np.int32),
                       overflow_flags=np.full((1, 4), -1, np.int32))
            # Keep the returned physical sample even if reading its capacity evidence fails.
            rows.append(row)
            capacity = capacity_record(audit, 4)
            row.update(valid=np.array([capacity["valid"] and bool(row["finite"][0])]),
                       capacity_observed=np.array([True]),
                       requested_contacts=capacity["requested_contacts"].reshape(1, 4),
                       requested_rows=capacity["requested_rows"].reshape(1, 4),
                       requested_nnz=capacity["requested_nnz"].reshape(1, 4),
                       overflow_flags=capacity["overflow_flags"].reshape(1, 4))
            if bool(row["terminated"][0]) or not bool(row["valid"][0]):
                break
    except Exception as error:
        try:
            write_trace(trace_path, rows)
            trace_path.with_suffix(".failure.json").write_text(json.dumps(dict(
                error_type=type(error).__name__, error=str(error), returned_physical_controls=len(rows),
                failed_call_execution_unknown=not rows or bool(rows[-1]["capacity_observed"][0])), indent=2) + "\n")
        except Exception as flush_error:
            error.add_note(f"Partial trace write also failed: {flush_error}")
        raise
    return state, rows


def load_actors(runs, settings):
    selected_update = settings["checkpoint_update"]
    if list(runs) != settings["paired_routes"]:
        raise ValueError("Run routes differ from selected study input")
    actors, configs = {}, {}
    for route, run in runs.items():
        cfg = json.loads((run / "config.json").read_text())
        if cfg.get("position_feedback") != route or cfg.get("clips") != ["all"]:
            raise ValueError("Run must declare its route and all forty clips")
        if cfg.get("actor_obs_norm", False):
            raise ValueError("The matched recovery study requires unnormalized actor inputs")
        if cfg.get("seed") not in settings["training_seeds"]:
            raise ValueError("Run seed differs from selected study input")
        with (run / "ckpt" / f"it_{selected_update:06d}.pkl").open("rb") as handle:
            checkpoint = pickle.load(handle)
        validate_feedback(checkpoint, cfg)
        if checkpoint.get("completed_updates") != selected_update or checkpoint.get("iter") != selected_update:
            raise ValueError("Checkpoint update differs from selected study input")
        actors[route] = jax.tree_util.tree_map(jnp.asarray, evaluation_actor(checkpoint, settings["policy"]))
        configs[route] = cfg
    if configs["encoder"]["seed"] != configs["decoder"]["seed"]:
        raise ValueError("Routes must belong to the same training seed")
    for field in ("residual_action", "action_rate_space", "solver_iterations", "seed"):
        if configs["encoder"].get(field) != configs["decoder"].get(field):
            raise ValueError(f"Route configurations differ at {field}")
    return actors, configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--encoder-run", type=Path, required=True)
    parser.add_argument("--decoder-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from nablamimic import mjw_options
    mjw_options.configure(False, 0, 1e-3, 0, 0.0, 0)
    import mujoco
    import warp as wp
    from nablamimic.evaluation_state import WarpEvaluationState
    settings = load_settings(args.resolved_config)
    offsets = settings["world_x_offsets_m"]
    prefix_controls = settings["prefix_controls"]
    maximum = settings["maximum_controls"]
    actors, configs = load_actors(dict(encoder=args.encoder_run, decoder=args.decoder_run), settings)
    cfg = configs["encoder"]
    if cfg["seed"] != args.training_seed:
        raise ValueError("Checkpoint seed differs from the requested paired study")
    if not cfg.get("residual_action") or cfg.get("action_rate_space") != "setpoint":
        raise ValueError("The matched runs require reference residual actions and setpoint rate cost")
    args.out.mkdir(parents=True, exist_ok=False)
    summary_path = args.out / "conditions.jsonl"
    (args.out / "inputs.json").write_text(json.dumps(dict(
        runs={"encoder": str(args.encoder_run), "decoder": str(args.decoder_run)},
        seed=cfg["seed"], study_input=settings, resolved_config=str(args.resolved_config), fresh_execution=True), indent=2) + "\n")
    helper = load_helper("observed_helpers")
    observer = load_helper("evaluation_capacity")
    audits = observer.install(sim)
    names = all_clip_names()
    if len(names) != 40:
        raise ValueError("Expected the original forty LAFAN1 clips")
    nets.set_matmul_precision("highest")
    library = load_library(names, bodies=BM_BODIES)
    env = TrackingEnv(library, EnvConfig(position_feedback="encoder", dr=False, obs_noise=False,
                                         reset_noise=False, residual_action=True,
                                         action_rate_space="setpoint", solver_iterations=cfg["solver_iterations"]))
    # Both position routes use identical observations; actor marker dimensions select their network route.
    env.warp = sim.make_warp(env, [1], train=False, forward_only=True)
    physics, audit = env.warp[1], audits[0]
    if (int(physics.sim.m.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            or physics.sim.m.callback.control or not physics.sim.m.opt.run_collision_detection):
        raise ValueError("Nominal recovery requires active collision detection and no sleeping/control callback")
    initial_native = WarpEvaluationState(physics.sim.m, physics.sim.df, wp.get_device("cuda:0"))
    tables = env.tables()
    transition = make_transition(env, dict(record=helper.make_record(), terms=tuple(W)))
    reset = jax.jit(jax.vmap(env.reset))
    restore = jax.jit(jax.vmap(env.restore))
    keys = jax.random.split(jax.random.PRNGKey(1), 1)
    dummy = reset(keys, jnp.asarray(env.starts[:1], jnp.int32))
    compiled = {route: transition.lower(actor, dummy, tables).compile() for route, actor in actors.items()}

    def step_for(route):
        return lambda actor, state: compiled[route](actor, state, tables)

    def save_summary(row):
        with summary_path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    for clip_index, name in enumerate(names):
        last = int(env.ref_last[int(env.starts[clip_index])])
        for donor, donor_actor in actors.items():
            label = f"clip{clip_index:02d}-{donor}"
            folder = args.out / label
            folder.mkdir()
            initial_native.restore()
            state = reset(keys, jnp.asarray([env.starts[clip_index]], jnp.int32))
            prefix_count = min(prefix_controls, last - int(state.frame[0]))
            state, prefix = advance(donor_actor, state, prefix_count, step_for(donor), audit, folder / "prefix.npz")
            write_trace(folder / "prefix.npz", prefix)
            available = (len(prefix) == prefix_controls and bool(prefix[-1]["valid"][0])
                         and not bool(prefix[-1]["terminated"][0]) and int(state.frame[0]) < last)
            donor_info = dict(clip=name, donor=donor, seed=cfg["seed"], prefix_executed=len(prefix),
                              release_frame=int(state.frame[0]), donor_available=available)
            if not available:
                for target in actors:
                    for displacement in offsets:
                        save_summary({**donor_info, "target": target, "displacement_world_x_m": displacement,
                                      "status": "donor_unavailable"})
                continue
            compact = jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), env.compact(state))
            np.savez_compressed(folder / "donor-state.npz",
                                **{name: np.asarray(value) for name, value in compact.items() if name != "xi"},
                                xi_friction=np.asarray(compact["xi"].friction),
                                xi_enc_bias=np.asarray(compact["xi"].enc_bias), xi_com=np.asarray(compact["xi"].com))
            sham_state = restore(compact)
            snapshot = WarpEvaluationState(physics.sim.m, physics.sim.df, wp.get_device("cuda:0"))
            direct_state, direct_rows = advance(donor_actor, state, 1, step_for(donor), audit, folder / "direct-first-control.npz")
            write_trace(folder / "direct-first-control.npz", direct_rows)
            remaining = last - int(state.frame[0])
            count = planned_controls(state.frame[0], last, maximum)
            for target, actor in actors.items():
                target_rows = {}
                for offset_index, displacement in enumerate(offsets):
                    branch = f"{target}-offset{offset_index:02d}"
                    shifted = translated_compact(compact, displacement)
                    pre_capacity = native_refresh(physics, snapshot, shifted, audit)
                    branch_state = restore(shifted)
                    check_branch_inputs(env, sham_state, branch_state, displacement)
                    release = jax.device_get(policy_probe(actor, env, branch_state))
                    np.savez_compressed(folder / f"{branch}-release.npz", **release,
                                        qpos=np.asarray(branch_state.dx.qpos), qvel=np.asarray(branch_state.dx.qvel),
                                        qacc_warmstart=np.asarray(branch_state.dx.qacc_warmstart),
                                        frame=np.asarray(branch_state.frame), body_pos=np.asarray(branch_state.dx.xpos),
                                        prev_a=np.asarray(branch_state.prev_a), ep=np.asarray(branch_state.ep))
                    pre_ok = pre_capacity["valid"] and all(np.isfinite(value).all() for value in release.values())
                    pre_ok = pre_ok and all(np.isfinite(np.asarray(value)).all()
                                           for value in jax.tree_util.tree_leaves(shifted))
                    np.savez_compressed(folder / f"{branch}-refresh.npz", **pre_capacity)
                    if pre_ok:
                        _, rows = advance(actor, branch_state, count, step_for(target), audit, folder / f"{branch}.npz")
                    else:
                        rows = []
                    write_trace(folder / f"{branch}.npz", rows)
                    target_rows[displacement] = rows
                    if target == donor and displacement == 0.0 and rows:
                        identity = {key: float(np.max(np.abs(rows[0][key] - direct_rows[0][key])))
                                    for key in ("qpos", "qvel", "policy_action", "setpoint", "metrics")}
                        identity = {name: value if np.isfinite(value) else None for name, value in identity.items()}
                        (folder / "sham-first-control-identity.json").write_text(
                            json.dumps(identity, indent=2, allow_nan=False) + "\n")
                    report = summarize([r["valid"][0] for r in rows], [r["terminated"][0] for r in rows], count, remaining, maximum)
                    if not pre_ok:
                        report["end"] = "invalid_before_first_action"
                    save_summary({**donor_info, "target": target, "displacement_world_x_m": displacement,
                                  "pre_action_valid": bool(pre_ok), "trace": f"{label}/{branch}.npz", **report})
                paired = {str(displacement): paired_diagnostic(target_rows[0.0], target_rows[displacement],
                                                              maximum, settings["readout_seconds"])
                          for displacement in offsets if displacement != 0.0}
                (folder / f"{target}-perturbation-minus-sham.json").write_text(
                    json.dumps(paired, indent=2, allow_nan=False) + "\n")
            del snapshot


if __name__ == "__main__":
    main()
