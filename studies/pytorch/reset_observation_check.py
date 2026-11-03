"""Frozen-input reset/restore and observation isolation; no training or tuning.

Both implementations construct their own derived physical state. The reset
comparison calls each production reset, matching JAX's explicitly exported
uniform draws. The restore comparison shares only compact inputs, never derived
body fields. Network-only controls share observations and frozen weights.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import time


RTOL, ATOL, SEED = 1e-4, 1e-5, 20261103
CONDITIONS = {
    "nominal": dict(dr=False, reset_noise=False, obs_noise=False),
    "domain_randomization": dict(dr=True, reset_noise=False, obs_noise=False),
    "training_reset": dict(dr=True, reset_noise=True, obs_noise=True),
}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--engine-root", type=Path, required=True)
    p.add_argument("--engine-capacity-revision", required=True)
    p.add_argument("--reference-cache", type=Path, required=True)
    p.add_argument("--checkpoint", nargs=3, action="append", required=True,
                   metavar=("LABEL", "POLICY", "PATH"))
    p.add_argument("--orientation-trace", nargs=2, action="append", required=True,
                   metavar=("LABEL", "OBSERVED_NPZ"))
    return p


def main():
    args = parser().parse_args()
    labels = [item[0] for item in args.checkpoint]
    if len(labels) != len(set(labels)) or any(not x.replace("_", "").replace("-", "").isalnum() for x in labels):
        raise ValueError("Checkpoint labels must be unique path-safe identifiers")
    if any(item[1] not in ("live", "ema") for item in args.checkpoint):
        raise ValueError("Each checkpoint policy must be live or ema")
    trace_labels = [item[0] for item in args.orientation_trace]
    if len(trace_labels) != len(set(trace_labels)) or any(not x.replace("_", "").replace("-", "").isalnum() for x in trace_labels):
        raise ValueError("Orientation trace labels must be unique path-safe identifiers")
    if os.environ.get("DK_ALLOW_UNSUPPORTED"):
        raise ValueError("Unsupported-physics bypass is forbidden")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  argv=sys.argv, source_revision=args.source_revision,
                  engine_capacity_revision=args.engine_capacity_revision,
                  source_root=str(args.source_root.resolve()), engine_root=str(args.engine_root.resolve()),
                  reference_cache=str(args.reference_cache.resolve()), checks=[], failures=[],
                  checkpoints={}, conditions={}, scope=__doc__,
                  orientation_protocol=dict(
                      traces=[dict(label=label, path=str(Path(path).resolve())) for label, path in args.orientation_trace],
                      precision="FP32", sample_count=64,
                      selection="32 evenly spaced adjacent valid rows plus 32 lowest orientation-reward rows",
                      finite_difference_dtype="float64", finite_difference_step=1e-6,
                      finite_difference="central ambient component differences; all 14 body quaternions",
                      vjp="unit cotangent per world, sum of rewards",
                      control="previous-row physical state/current-row setpoint, zero warm; counterfactual, not exact replay"),
                  protocol=dict(rtol=RTOL, atol=ATOL, finite_required=True,
                                discrete_exact=True, seed=SEED, batch=160,
                                frames="all forty clips: start, midpoint, end, one seeded reset frame",
                                conditions=CONDITIONS, precisions=["fp32", "tf32"],
                                torch_execution="eager", jax_execution="jit/vmap",
                                no_state_copy="Only compact inputs are shared for restore; reset paths are independent",
                                failed_conditions_retained=True,
                                untested=["training updates", "CUDA graphs", "long-horizon policy quality"]))
    started = time.monotonic()

    def save():
        write_json(root / "result.json", record)

    def check(name, passed, **fields):
        record["checks"].append(dict(name=name, passed=bool(passed), **fields))
        if not passed:
            record["failures"].append(name)
        save()
        return bool(passed)

    def require(name, passed, **fields):
        if not check(name, passed, **fields):
            raise AssertionError(name)

    save()  # Freeze protocol before importing frameworks or loading inputs.
    try:
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_CLIENT_MEM_FRACTION"] = ".45"
        os.environ["MOTION_CACHE_DIR"] = str(args.reference_cache.resolve())
        for variable, child in (("DK_CACHE_DIR", "native"), ("JAX_COMPILATION_CACHE_DIR", "jax"),
                                ("CUDA_CACHE_PATH", "cuda")):
            path = root / "cache" / child
            path.mkdir(parents=True)
            os.environ[variable] = str(path)
        sys.path.insert(0, str(args.source_root.resolve() / "src"))
        sys.path.insert(0, str(args.engine_root.resolve() / "src"))
        sys.path.insert(0, str(args.source_root.resolve() / "studies" / "pytorch"))

        import numpy as np
        import torch
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step as engine_step
        from nablamimic import torch_env, torch_nets as tn
        from nablamimic.env import BM_BODIES, EnvConfig, TrackingEnv, Xi, build_model
        from nablamimic.motion import all_clip_names, load_library
        from nablamimic.params import evaluation_actor
        from engine_check import host_inputs

        require("source_import_location", Path(torch_env.__file__).resolve().is_relative_to(args.source_root.resolve()))
        require("engine_import_location", Path(tf.__file__).resolve().is_relative_to(args.engine_root.resolve()))
        require("one_cuda_device", torch.cuda.is_available() and torch.cuda.device_count() == 1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(.45)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = EnvConfig(solver_iterations=10, residual_action=True, action_rate_space="setpoint", **CONDITIONS["nominal"])
        host_inputs(args.reference_cache, build_model(cfg)[0])  # Validate existing caches before load_library.
        names = all_clip_names()
        require("forty_clips", len(names) == 40)
        library = load_library(names, bodies=BM_BODIES)
        tenv = torch_env.TorchTrackingEnv(library, cfg)
        info = tf.init(model=tenv.m, foot_pairs=tenv.foot_pairs)
        tf.set_diagnostics(True)
        require("native_capacity", info["maxcon"] == 96 and engine_step.NEFC == 416 and info["rec_words"] == 1504,
                max_contacts=int(info["maxcon"]), rows=int(engine_step.NEFC), record_words=int(info["rec_words"]))
        require("native_kinematics_available", info["kinematics_error"] is None)
        record["engine"] = {key: info.get(key) for key in ("version", "arch", "nq", "nv", "nu", "maxcon", "rec_words")}

        import jax
        import jax.numpy as jnp
        from nablamimic import nets as jn

        require("jax_cuda", jax.default_backend() == "gpu")
        jn.set_matmul_precision("highest")
        jenv = TrackingEnv(library, cfg)
        record["versions"] = dict(torch=torch.__version__, torch_cuda=torch.version.cuda, jax=jax.__version__,
                                  gpu=torch.cuda.get_device_name(0))
        record["joint_mapping"] = [dict(action_index=i, joint_name=tenv.m.joint(int(tenv.m.actuator_trnid[i, 0])).name,
                                         qpos_index=int(tenv.m.jnt_qposadr[tenv.m.actuator_trnid[i, 0]]),
                                         qvel_index=int(tenv.m.jnt_dofadr[tenv.m.actuator_trnid[i, 0]]))
                                   for i in range(tenv.nu)]
        require("joint_coordinate_mapping", all(row["qpos_index"] == 7+row["action_index"]
                and row["qvel_index"] == 6+row["action_index"] for row in record["joint_mapping"]))

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def tt(value):
            return torch.as_tensor(np.array(value, copy=True), device="cuda")

        def compare(name, left, right):
            a, b = cpu(left), cpu(right)
            shape = a.shape == b.shape
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            exact = a.dtype.kind in "biu" or b.dtype.kind in "biu"
            fields = dict(shape_jax=list(a.shape), shape_torch=list(b.shape), finite=finite, exact=exact)
            passed = False
            if shape and finite:
                error = np.abs(a.astype(np.float64) - b.astype(np.float64))
                limit = np.zeros_like(error) if exact else ATOL + RTOL * np.abs(a.astype(np.float64))
                bad = error > limit
                passed = not bool(bad.any())
                fields.update(max_abs_difference=float(error.max(initial=0)), bad_elements=int(bad.sum()),
                              bad_rows=np.flatnonzero(bad.reshape((len(a), -1)).any(1)).tolist() if a.ndim else [],
                              max_tolerance_ratio=float((error / np.maximum(limit, 1e-300)).max(initial=0)) if not exact else None)
                if bad.any():
                    index = tuple(int(i) for i in np.unravel_index(error.argmax(), error.shape))
                    fields.update(worst_index=list(index), jax_at_worst=float(a[index]), torch_at_worst=float(b[index]))
            check(name, passed, **fields)

        def compare_dict(prefix, left, right):
            require(prefix + ".keys", set(left) == set(right))
            for key in sorted(left):
                compare(prefix + "." + key, left[key], right[key])

        def state_fields(state):
            return dict(qpos=state.dx.qpos, qvel=state.dx.qvel, warm=state.dx.qacc_warmstart,
                        xpos=state.dx.xpos, xquat=state.dx.xquat, cvel=state.dx.cvel,
                        subtree_com=state.dx.subtree_com, frame=state.frame, ep=state.ep,
                        hist=state.hist, prev_a=state.prev_a, friction=state.xi.friction,
                        enc_bias=state.xi.enc_bias, com=state.xi.com)

        for key in tenv.TABLES:
            compare("reference." + key, getattr(jenv, key), getattr(tenv, key))
        for key in ("q0", "action_scale", "ctrl_lo", "ctrl_hi", "soft_lo", "soft_hi", "body_ids", "foot_pairs"):
            compare("model." + key, getattr(jenv, key), getattr(tenv, key))

        rng = np.random.default_rng(SEED)
        rows = []
        for clip, start in zip(names, library.starts):
            length = int(library.last[start] - start + 1)
            for label, local_frame in (("start", 0), ("midpoint", (length - 1) // 2),
                                       ("end", length - 1), ("reset", int(rng.integers(length)))):
                rows.append(dict(clip=clip, position=label, local_frame=local_frame, frame=int(start) + local_frame))
        frames = np.asarray([row["frame"] for row in rows], dtype=np.int32)
        count = len(frames)
        record["rows"] = rows
        require("declared_batch", count == 160)
        keys = jax.random.split(jax.random.PRNGKey(SEED), count)
        goal_keys = jax.random.split(jax.random.PRNGKey(SEED + 1), count)

        def unit_draws(key):
            kq, kv, kj, kx, kn, _ = jax.random.split(key, 6)
            kp, kr = jax.random.split(kq)
            kf, kb, kc = jax.random.split(kx, 3)
            def unit(k, shape):
                return jax.random.uniform(k, shape, minval=-1.0, maxval=1.0)
            return dict(position=unit(kp, (3,)), rotation=unit(kr, (3,)), velocity=unit(kv, (6,)),
                        joints=unit(kj, (29,)), friction=unit(kf, ()), enc_bias=unit(kb, (29,)),
                        com=unit(kc, (3,)), proprio=unit(kn, (93,)))

        noise = jax.jit(jax.vmap(unit_draws))(keys)
        goal_noise = jax.jit(jax.vmap(lambda key: jax.random.uniform(key, (6,), minval=-1.0, maxval=1.0)))(goal_keys)
        np.savez(root / "shared_random_inputs.npz", keys=cpu(keys), goal_keys=cpu(goal_keys),
                 frames=frames, goal_noise=cpu(goal_noise), **{k: cpu(v) for k, v in noise.items()})
        tnoise = {key: tt(cpu(value)) for key, value in noise.items()}
        tgoal_noise = tt(cpu(goal_noise))
        checkpoints = []
        for label, policy, path in args.checkpoint:
            with Path(path).open("rb") as handle:
                checkpoint = pickle.load(handle)
            actor = evaluation_actor(checkpoint, policy)
            require(label + ".actor_contract", "anchor_dec" in actor and actor["anchor_dec"].shape == (6,) and "obs_norm" not in actor)
            record["checkpoints"][label] = dict(path=str(Path(path).resolve()), policy=policy,
                completed_updates=int(checkpoint["completed_updates"]), ema_updates=int(checkpoint["ema_updates"]),
                critic="same checkpoint live critic; independent of selected actor EMA")
            checkpoints.append((label, actor, checkpoint["critic"], checkpoint["cnorm"]))
        save()

        from orientation_gradient_check import run_orientation_check
        record["orientation"] = run_orientation_check(
            root=root, traces=args.orientation_trace, jenv=jenv, tenv=tenv,
            jax=jax, jnp=jnp, torch=torch, np=np, cpu=cpu, tt=tt, check=check, compare=compare)
        save()

        for condition, flags in CONDITIONS.items():
            cfg = EnvConfig(solver_iterations=10, residual_action=True, action_rate_space="setpoint", **flags)
            jenv.cfg = tenv.cfg = cfg
            record["conditions"][condition] = asdict(cfg)
            case_root = root / condition
            case_root.mkdir()
            js = jax.block_until_ready(jax.jit(jax.vmap(jenv.reset))(keys, jnp.asarray(frames)))
            with torch.no_grad():
                ts = tenv.reset(tt(frames), tnoise)
            # Shared compact restore controls deliberately contain nontrivial history,
            # warm starts, previous actions and episode ages. No body fields are copied.
            compact = {key: cpu(value).copy() for key, value in state_fields(js).items() if key in torch_env.COMPACT_KEYS}
            compact["warm"] = rng.normal(0, .1, (count, 35)).astype(np.float32)
            compact["hist"] = rng.normal(0, .3, (count, 10, 93)).astype(np.float32)
            compact["prev_a"] = rng.normal(0, .2, (count, 29)).astype(np.float32)
            compact["ep"] = (np.arange(count, dtype=np.int32) * 17) % 501
            compact["ep"][::4] = 0
            compact["hist"][::4] = compact["hist"][::4, :1]
            compact["prev_a"][::4] = 0
            # Explicit finite fallen and near-height-threshold states remain in scope.
            compact["qpos"][1::4, 2] = np.asarray(library.qpos)[frames[1::4], 2] + .30
            compact["qpos"][2::4, 2] = np.asarray(library.qpos)[frames[2::4], 2] + .249
            np.savez(case_root / "shared_compact.npz", **compact)
            jcompact = dict(qpos=jnp.asarray(compact["qpos"]), qvel=jnp.asarray(compact["qvel"]),
                            qacc_warmstart=jnp.asarray(compact["warm"]), frame=jnp.asarray(compact["frame"]),
                            ep=jnp.asarray(compact["ep"]), hist=jnp.asarray(compact["hist"]),
                            prev_a=jnp.asarray(compact["prev_a"]), key=keys,
                            xi=Xi(*(jnp.asarray(compact[k]) for k in ("friction", "enc_bias", "com"))))
            jr = jax.block_until_ready(jax.jit(jax.vmap(jenv.restore))(jcompact))
            with torch.no_grad():
                tr = tenv.restore({key: tt(value) for key, value in compact.items()})

            def jfeatures(state, key):
                h, g = jenv.policy_obs(state, key)
                reward, terms = jenv.reward(state, jnp.zeros(29), state.prev_a, state)
                linear, angular = jenv._body_vel(state.dx)
                return dict(proprio=jenv.proprio(state.dx, state.xi, state.prev_a), history=h,
                            goal=g, goal_clean=jenv.goal(state), anchor=jenv._anchor_rel(state.dx, state.frame),
                            critic_obs=jenv.critic_obs(state), body_linear=linear, body_angular=angular,
                            hazard=jenv.hazard_features(state), terminated=jenv.terminated(state),
                            clip_ended=state.frame >= jenv.ref_last[state.frame],
                            zero_action_ctrl=jenv.ctrl_of(jnp.zeros(29), state.frame), zero_action_reward=reward,
                            **{"zero_action_reward_" + key: value for key, value in terms.items()})

            def tfeatures(state):
                h, g = tenv.policy_obs(state, tgoal_noise)
                action = torch.zeros((count, 29), device="cuda")
                reward, terms = tenv.reward(state, action, state["prev_a"], state)
                linear, angular = tenv.body_velocity(state)
                return dict(proprio=tenv.proprio(state), history=h, goal=g,
                            goal_clean=torch.cat((tenv.goal_ref(state["frame"]), tenv.anchor_rel(state)[:, 3:9]), -1),
                            anchor=tenv.anchor_rel(state), critic_obs=tenv.critic_obs(state),
                            body_linear=linear, body_angular=angular, hazard=tenv.hazard_features(state),
                            terminated=tenv.terminated(state), clip_ended=tenv.clip_ended(state),
                            zero_action_ctrl=tenv.ctrl_of(action, state["frame"]), zero_action_reward=reward,
                            **{"zero_action_reward_" + key: value for key, value in terms.items()})

            jfeature_fn = jax.jit(jax.vmap(jfeatures))
            for operation, jstate, tstate in (("reset", js, ts), ("restore", jr, tr)):
                prefix = condition + "." + operation
                target = case_root / operation
                target.mkdir()
                jfields = state_fields(jstate)
                compare_dict(prefix + ".state", jfields, tstate)
                np.savez(target / "jax_state.npz", **{k: cpu(v) for k, v in jfields.items()})
                np.savez(target / "torch_state.npz", **{k: cpu(v) for k, v in tstate.items()})
                jf = jax.block_until_ready(jfeature_fn(jstate, goal_keys))
                with torch.no_grad():
                    tv = tfeatures(tstate)
                compare_dict(prefix + ".features", jf, tv)
                np.savez(target / "jax_features.npz", **{k: cpu(v) for k, v in jf.items()})
                np.savez(target / "torch_features.npz", **{k: cpu(v) for k, v in tv.items()})
                record["conditions"][condition][operation + "_termination_counts"] = dict(
                    jax=int(cpu(jf["terminated"]).sum()), torch=int(cpu(tv["terminated"]).sum()))
                for precision in ("fp32", "tf32"):
                    jn.set_matmul_precision("highest" if precision == "fp32" else "nets_tf32")
                    torch.set_float32_matmul_precision("highest")
                    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
                    torch.backends.cudnn.allow_tf32 = False
                    record.setdefault("precision_settings", {})[precision] = dict(
                        torch_matmul_precision=torch.get_float32_matmul_precision(),
                        torch_cuda_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        torch_cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                        jax_nets_precision=jn.NET_PRECISION)

                    def jnetwork(actor, critic, norm, history, goal, critic_obs):
                        return dict(latent=jn.actor_encode(actor, goal[:, :-6]),
                                    action=jn.actor_apply(actor, history, goal),
                                    critic=jn.critic_apply(critic, jn.normalize(norm, critic_obs)))

                    jnetwork_fn = jax.jit(jnetwork)
                    for label, actor, critic, norm in checkpoints:
                        net_prefix = prefix + "." + precision + "." + label
                        ja, jc, jnrm = (jax.tree.map(jnp.asarray, x) for x in (actor, critic, norm))
                        ta, tc, tnrm = (tn.tree_from_jax(x, "cuda") for x in (actor, critic, norm))
                        joutput = jax.block_until_ready(jnetwork_fn(ja, jc, jnrm, jf["history"], jf["goal"], jf["critic_obs"]))
                        with torch.no_grad():
                            def tnetwork(history, goal, critic_obs):
                                return dict(latent=tn.actor_encode(ta, goal[:, :-6]),
                                            action=tn.actor_apply(ta, history, goal),
                                            critic=tn.critic_apply(tc, tn.normalize(tnrm, critic_obs)))
                            toutput = tnetwork(tv["history"], tv["goal"], tv["critic_obs"])
                            control = tnetwork(*(tt(cpu(jf[k])) for k in ("history", "goal", "critic_obs")))
                        compare_dict(net_prefix + ".own_observations", joutput, toutput)
                        compare_dict(net_prefix + ".shared_observations_control", joutput, control)
                        ja_action = joutput["action"]
                        jreward, jterms = jax.jit(jax.vmap(jenv.reward))(jstate, ja_action, jstate.prev_a, jstate)
                        with torch.no_grad():
                            treward, tterms = tenv.reward(tstate, toutput["action"], tstate["prev_a"], tstate)
                            shared_reward, shared_terms = tenv.reward(tstate, tt(cpu(ja_action)), tstate["prev_a"], tstate)
                        jpolicy = dict(reward=jreward, ctrl=jax.vmap(jenv.ctrl_of)(ja_action, jstate.frame), **jterms)
                        tpolicy = dict(reward=treward, ctrl=tenv.ctrl_of(toutput["action"], tstate["frame"]), **tterms)
                        shared_policy = dict(reward=shared_reward, ctrl=tenv.ctrl_of(tt(cpu(ja_action)), tstate["frame"]), **shared_terms)
                        compare_dict(net_prefix + ".own_actions", jpolicy, tpolicy)
                        compare_dict(net_prefix + ".shared_action_control", jpolicy, shared_policy)
                        destination = target / precision / label
                        destination.mkdir(parents=True)
                        for name, values in (("jax_network", joutput), ("torch_network", toutput),
                                             ("torch_shared_obs_network", control), ("jax_policy", jpolicy),
                                             ("torch_policy", tpolicy), ("torch_shared_action_policy", shared_policy)):
                            np.savez(destination / (name + ".npz"), **{k: cpu(v) for k, v in values.items()})
                jn.set_matmul_precision("highest")
                torch.set_float32_matmul_precision("highest")
                torch.backends.cuda.matmul.allow_tf32 = False
            save()
        record["status"] = "failed" if record["failures"] else "passed"
    except BaseException as error:
        record["status"] = "failed"
        record["exception"] = repr(error)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        save()
    if record["failures"]:
        raise SystemExit("Frozen reset/restore checks failed; all completed conditions retained")


if __name__ == "__main__":
    main()
