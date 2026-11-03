"""The BM policy file exporter on a tiny synthetic checkpoint (CPU; the compiled-model part needs Menagerie)."""

import contextlib
import copy
import io
import json
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic import export_bm_policy as ebp
from nablamimic.env import BM_BODIES, ActionParameters
from nablamimic.motion import SCENE
from nablamimic.nets import actor_apply, actor_init, critic_apply, mlp_init, normalize

HIDDEN, LATENT, NU = (8, 4), 3, 29
HIDDEN4 = (8, 7, 6, 5)  # four hidden layers, as the trained actor has: Linear indices 0, 2, 4, 6, 8
ENV = types.SimpleNamespace(goal_dim=10 * 68 + 6, hist_dim=10 * 93, nu=NU)  # the real input widths
LABEL, PER_UPDATE = 12, 16 * 1 * 4
CRITIC_HIDDEN, CRITIC_IN = (7, 5), 93 + 680 + 6 + 4 + 9 + 33  # env.TrackingEnv.critic_dim


def numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, np.float32), jax.device_get(tree))


def released_checkpoint(seed=0, hidden=HIDDEN):
    """A checkpoint as the released recipe saves it: no evaluation policy, completed or EMA counts."""
    actor = numpy_tree(actor_init(jax.random.PRNGKey(seed), ENV, hidden=hidden, latent=LATENT, anchor_to_decoder=6))
    critic = numpy_tree(mlp_init(jax.random.PRNGKey(seed + 100), [CRITIC_IN, *CRITIC_HIDDEN, 1]))
    rng = np.random.default_rng(seed)
    cnorm = (rng.normal(size=CRITIC_IN).astype(np.float32), rng.uniform(0.5, 2.0, CRITIC_IN).astype(np.float32))
    return dict(actor=actor, critic=critic, cnorm=cnorm, iter=LABEL, wall_s=1.0, env_steps=LABEL * PER_UPDATE)


def averaged_checkpoint():
    ck = released_checkpoint()
    ck.update(actor_ema=numpy_tree(actor_init(jax.random.PRNGKey(7), ENV, hidden=HIDDEN, latent=LATENT,
                                              anchor_to_decoder=6)),
              ema_updates=LABEL, completed_updates=LABEL, evaluation_policy="ema")
    return ck


def run_config(**override):
    cfg = dict(physics="engine", num_envs=16, devices=1, horizon=4, hist_len=10, obs_mode="anchor_ori",
               anchor_to_decoder=True, actor_obs_norm=False, actor_hidden=list(HIDDEN), actor_latent=LATENT,
               residual_action=True, action_rate_space="setpoint", solver_iterations=10, robot="mjlab",
               contacts="all", matmul_precision="nets_tf32", clips=["all"], motion_lib=None, git_commit=None,
               critic_hidden=list(CRITIC_HIDDEN), critic_pre_state=True,
               git_dirty=False, gpu="test card", argv=["--residual-action"],
               segments=[dict(engine=dict(version="0.1.0", key="library-key", arch="sm_80"))])
    cfg.update(override)
    return cfg


def torch_run_config(**override):
    """config.json as torch_train.py writes it: vars(args) with "config", no argv or git_dirty, its own bookkeeping
    and a top-level engine record."""
    cfg = run_config(training_framework="torch", config="configs/ours_torch.args", out="RUN", resume=None,
                     execution="compile-graph", initial_checkpoint="JAX_RUN/ckpt/it_000000.pkl", git_commit="abc123",
                     torch_version="2.9.1", torch_cuda="12.8", random_stream="torch generators", clock="setup excluded",
                     inputs=dict(MOTION_CACHE_DIR="CACHE"), declared_environment="image",
                     engine=dict(version="0.1.0", key="torch-library-key", arch="sm_80", maxcon=96))
    for key in ("argv", "git_dirty", "segments"):
        del cfg[key]
    cfg.update(override)
    return cfg


def action_parameters():
    i = np.arange(NU, dtype=np.float64)
    kp = 40.0 + i
    effort = 25.0 + 0.5 * i
    return ActionParameters(tuple(f"joint_{k}" for k in range(NU)), 0.01 * i, kp, effort, 0.25 * effort / kp,
                            -1.0 - 0.1 * i, 1.0 + 0.1 * i)


def manifest(**override):
    m = dict(format="nablamimic-reference-manifest-v1", artifact_id="reference-test", frame_rate_hz=50.0,
             bodies=list(BM_BODIES), clip_selection="all", clip_order="sorted_clip_names",
             clips=[dict(name="dance1_subject1", frames=30), dict(name="walk1_subject1", frames=40)])
    m.update(override)
    return m


def export(checkpoint=None, cfg=None, act=None, man=None, policy="live", label=LABEL, **kw):
    args = dict(artifact_id="policy-test", source_artifact_id="checkpoint-test", engine_revision="engine-test",
                contact_capacity=64, constraint_row_capacity=288, end_markers={})
    args.update(kw)
    return ebp.build_policy_file(released_checkpoint() if checkpoint is None else checkpoint,
                                 run_config() if cfg is None else cfg, act or action_parameters(),
                                 manifest() if man is None else man, policy=policy, label=label, **args)


def numpy_forward(arrays, hist, goal):
    """The exported file executed with numpy alone: Linear indices 0, 2, ..., ELU between, tanh latent."""
    def mlp(prefix, x):
        n = sorted(int(k.split(".")[1]) for k in arrays if k.startswith(prefix) and k.endswith(".weight"))
        for j, i in enumerate(n):
            x = x @ arrays[f"{prefix}.{i}.weight"].T + arrays[f"{prefix}.{i}.bias"]
            if j < len(n) - 1:
                x = np.where(x > 0, x, np.expm1(x))
        return x
    z = np.tanh(mlp("encoder", goal[:, :-6]))
    return mlp("decoder", np.concatenate([z, hist, goal[:, -6:]], -1))


class ArrayTests(unittest.TestCase):
    def test_arrays_are_the_kernels_transposed(self):
        ck = released_checkpoint()
        arrays, _ = export(ck)
        expected = {f"{p}.{n}.{s}" for p in ("encoder", "decoder") for n in (0, 2, 4) for s in ("weight", "bias")}
        self.assertEqual(set(arrays), expected)
        for tree, prefix in (("enc", "encoder"), ("dec", "decoder")):
            for i, (w, b) in enumerate(ck["actor"][tree]):
                np.testing.assert_array_equal(arrays[f"{prefix}.{2 * i}.weight"], w.T)
                np.testing.assert_array_equal(arrays[f"{prefix}.{2 * i}.bias"], b)
                self.assertTrue(arrays[f"{prefix}.{2 * i}.weight"].flags["C_CONTIGUOUS"])
                self.assertEqual(arrays[f"{prefix}.{2 * i}.weight"].dtype, np.float32)
        self.assertEqual(arrays["encoder.0.weight"].shape, (HIDDEN[0], 680))
        self.assertEqual(arrays["decoder.0.weight"].shape, (HIDDEN[0], LATENT + 930 + 6))
        self.assertEqual(arrays["decoder.4.weight"].shape, (NU, HIDDEN[-1]))

    def test_four_hidden_layers_reach_indices_six_and_eight(self):
        ck = released_checkpoint(hidden=HIDDEN4)
        arrays, decl = export(ck, cfg=run_config(actor_hidden=list(HIDDEN4)))
        expected = {f"{p}.{n}.{s}" for p in ("encoder", "decoder") for n in (0, 2, 4, 6, 8)
                    for s in ("weight", "bias")}
        self.assertEqual(set(arrays), expected)
        for tree, prefix in (("enc", "encoder"), ("dec", "decoder")):
            self.assertEqual(len(ck["actor"][tree]), 5)
            for i, (w, b) in enumerate(ck["actor"][tree]):
                np.testing.assert_array_equal(arrays[f"{prefix}.{2 * i}.weight"], w.T)
                np.testing.assert_array_equal(arrays[f"{prefix}.{2 * i}.bias"], b)
        self.assertEqual(arrays["encoder.6.weight"].shape, (HIDDEN4[3], HIDDEN4[2]))
        self.assertEqual(arrays["encoder.8.weight"].shape, (LATENT, HIDDEN4[3]))
        self.assertEqual(arrays["decoder.8.weight"].shape, (NU, HIDDEN4[3]))
        self.assertEqual(decl["architecture"]["decoder"]["hidden"], list(HIDDEN4))
        rng = np.random.default_rng(1)
        hist = rng.normal(size=(4, 930)).astype(np.float32)
        goal = rng.normal(size=(4, 686)).astype(np.float32)
        with jax.default_matmul_precision("highest"):
            want = np.asarray(actor_apply(jax.tree_util.tree_map(jnp.asarray, ck["actor"]), hist, goal))
        np.testing.assert_allclose(numpy_forward(arrays, hist, goal), want, rtol=1e-5, atol=1e-6)

    def test_numpy_execution_equals_the_actor(self):
        ck = released_checkpoint()
        arrays, _ = export(ck)
        rng = np.random.default_rng(0)
        hist = rng.normal(size=(5, 930)).astype(np.float32)
        goal = rng.normal(size=(5, 686)).astype(np.float32)
        with jax.default_matmul_precision("highest"):
            want = np.asarray(actor_apply(jax.tree_util.tree_map(jnp.asarray, ck["actor"]), hist, goal))
        np.testing.assert_allclose(numpy_forward(arrays, hist, goal), want, rtol=1e-5, atol=1e-6)

    def test_averaged_weights_selected(self):
        ck = averaged_checkpoint()
        arrays, decl = export(ck, policy="ema")
        np.testing.assert_array_equal(arrays["decoder.2.weight"], ck["actor_ema"]["dec"][1][0].T)
        self.assertEqual(decl["policy_source"]["selected_weights"], "averaged")
        self.assertEqual(decl["policy_source"]["counts"], dict(completed_outer_updates=LABEL, averaging_updates=LABEL))
        self.assertEqual(decl["policy_source"]["completed_outer_updates_source"], "checkpoint_completed_updates")
        arrays, decl = export(ck, policy="live")
        np.testing.assert_array_equal(arrays["decoder.2.weight"], ck["actor"]["dec"][1][0].T)
        self.assertEqual(decl["policy_source"]["selected_weights"], "live")

    def test_file_round_trip_without_pickle(self):
        arrays, decl = export()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "policy.npz"
            ebp.write_policy_file(arrays, decl, out)
            with np.load(out, allow_pickle=False) as z:
                self.assertEqual(set(z.files), set(arrays) | {"declaration"})
                self.assertEqual(z["declaration"].dtype.kind, "U")
                self.assertEqual(json.loads(z["declaration"].item()), decl)
                for k, v in arrays.items():
                    np.testing.assert_array_equal(z[k], v)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["policy.npz"])
            with self.assertRaises(FileExistsError):
                ebp.write_policy_file(arrays, decl, out)
            with self.assertRaises(ValueError):
                ebp.write_policy_file(arrays, decl, Path(d) / "policy.npy")


class DeclarationTests(unittest.TestCase):
    def test_fields(self):
        act = action_parameters()
        _, d = export(act=act)
        self.assertEqual((d["format"], d["artifact_id"]), (ebp.FORMAT, "policy-test"))
        src = d["policy_source"]
        self.assertEqual((src["kind"], src["source_artifact_id"]), ("imported_nablamimic", "checkpoint-test"))
        self.assertEqual(src["counts"], dict(completed_outer_updates=LABEL, averaging_updates=0))
        self.assertEqual(src["completed_outer_updates_source"], "env_steps_over_window_transitions_per_update")
        self.assertIsNone(src["checkpoint_evaluation_policy"])
        arch = d["architecture"]
        self.assertEqual((arch["encoder"]["input"], arch["encoder"]["hidden"], arch["encoder"]["output"]),
                         (680, list(HIDDEN), LATENT))
        self.assertEqual(arch["encoder"]["output_squash"], "tanh")
        self.assertEqual([x["name"] for x in arch["decoder"]["inputs"]],
                         ["latent", "proprioception_history", "anchor_orientation"])
        self.assertEqual((arch["decoder"]["input"], arch["decoder"]["output"], arch["activation"]),
                         (LATENT + 936, NU, "ELU"))
        inp = d["inputs"]
        self.assertEqual(sum(t["width"] for t in inp["proprioception"]["terms"]), 93)
        self.assertEqual([t["factor"] for t in inp["proprioception"]["terms"]], [1.0, 0.2, 0.5, 1.0, 1.0])
        self.assertEqual(inp["proprioception"]["terms"][-1]["name"], "previous_executed_action")
        self.assertEqual((inp["history"]["frames"], inp["history"]["order"]), (10, "frame_major_oldest_first"))
        self.assertEqual(inp["previous_action"], "executed_with_exploration_noise")
        self.assertEqual((inp["goal"]["rows"], inp["goal"]["row_width"]), (10, 68))
        self.assertEqual(inp["goal"]["past_clip_end"], "last_row_hold_including_velocities")
        self.assertEqual((inp["anchor"]["body"], inp["anchor"]["encoding"]), ("torso_link", "first_two_columns"))
        self.assertEqual(inp["normalisation"], "none")
        a = d["action"]
        self.assertEqual(a["joint_names"], list(act.joint_names))
        self.assertEqual(a["scale"], [float(np.float32(v)) for v in act.scale])
        self.assertEqual(a["nominal_joint_position"], [float(np.float32(v)) for v in act.nominal])
        self.assertEqual(a["setpoint_lower"], [float(np.float32(v)) for v in act.setpoint_lower])
        self.assertEqual(a["baseline"]["kind"], "next_reference_joint_position")
        self.assertIs(a["encoder_bias_in_setpoint"], False)
        self.assertEqual(d["timing"], dict(control_seconds=0.02, physics_seconds=0.005, substeps=4,
                                           solver_iterations=10))
        dom = d["training_domain"]
        self.assertEqual((dom["engine"]["contact_capacity"], dom["engine"]["constraint_row_capacity"]), (64, 288))
        self.assertEqual(dom["engine"]["declared_revision"], "engine-test")
        self.assertEqual(dom["engine"]["reported"], [dict(version="0.1.0", key="library-key", arch="sm_80")])
        self.assertEqual(dom["reference"]["clips"], ["dance1_subject1", "walk1_subject1"])
        self.assertEqual(dom["reference"]["artifact_id"], "reference-test")
        self.assertIsNone(dom["source_run"]["git_commit"])
        json.dumps(d)  # plain JSON

    def test_run_arguments(self):
        _, d = export()
        run = d["training_domain"]["source_run"]
        self.assertEqual(run["arguments_source"], "argv_recorded_at_first_start_with_config_file")
        self.assertEqual(run["argv"], ["--residual-action"])
        self.assertNotIn("parsed_arguments", run)
        _, d = export(cfg=run_config(config_args=["--residual-action"], config_file="configs/ours.args"))
        run = d["training_domain"]["source_run"]
        self.assertEqual((run["config_args"], run["config_file"]), (["--residual-action"], "configs/ours.args"))
        cfg = run_config()
        del cfg["argv"]
        _, d = export(cfg=cfg)
        run = d["training_domain"]["source_run"]
        self.assertEqual(run["arguments_source"], "parsed_arguments_in_config_json_run_started_without_config_file")
        self.assertNotIn("argv", run)
        args = run["parsed_arguments"]
        self.assertEqual((args["num_envs"], args["horizon"], args["residual_action"]), (16, 4, True))
        for field in ("robot", "contacts", "gpu", "segments", "git_commit", "devices", "actor_hidden"):
            self.assertNotIn(field, args)

    def test_torch_run(self):
        _, d = export(averaged_checkpoint(), cfg=torch_run_config(), policy="ema")
        run = d["training_domain"]["source_run"]
        self.assertEqual(run["arguments_source"], "parsed_arguments_in_config_json_torch_train_records_no_argv")
        self.assertEqual((run["training_framework"], run["git_commit"], run["git_dirty"]),
                         ("torch", "abc123", "not_recorded_by_torch_train"))
        args = run["parsed_arguments"]
        self.assertEqual((args["config"], args["initial_checkpoint"], args["execution"]),
                         ("configs/ours_torch.args", "JAX_RUN/ckpt/it_000000.pkl", "compile-graph"))
        for field in ("training_framework", "torch_version", "clock", "inputs", "declared_environment", "engine",
                      "gpu", "git_commit"):
            self.assertNotIn(field, args)
        self.assertEqual(d["training_domain"]["engine"]["reported"][0]["key"], "torch-library-key")
        _, d = export()
        run = d["training_domain"]["source_run"]
        self.assertEqual((run["training_framework"], run["git_dirty"]), ("jax", False))

    def test_end_markers_recorded(self):
        halted = dict(iter=LABEL, grad_norm="nan", wall_s=3.0, env_steps=LABEL * PER_UPDATE, rule="SHAC guard")
        _, d = export(end_markers=dict(HALTED=halted))
        self.assertEqual(d["policy_source"]["run_end_markers"], dict(HALTED=halted))
        _, d = export()
        self.assertEqual(d["policy_source"]["run_end_markers"], {})

    def test_absolute_actions(self):
        _, d = export(cfg=run_config(residual_action=False))
        self.assertEqual(d["action"]["baseline"], dict(kind="nominal_joint_position"))

    def test_listed_clips_must_match_in_order(self):
        man = manifest(clip_selection="listed", clip_order="as_listed")
        _, d = export(cfg=run_config(clips=["dance1_subject1", "walk1_subject1"]), man=man)
        self.assertEqual(d["training_domain"]["reference"]["clip_selection"], "listed")
        with self.assertRaises(ValueError):
            export(cfg=run_config(clips=["walk1_subject1", "dance1_subject1"]), man=man)
        with self.assertRaises(ValueError):
            export(man=man)  # the run trained on all clips


class RefusalTests(unittest.TestCase):
    def assertRefused(self, **kw):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            export(**kw)

    def test_weights_selection(self):
        self.assertRefused(policy="ema")  # released-recipe checkpoint: live only
        ck = averaged_checkpoint()
        ck["ema_updates"] = LABEL - 1
        self.assertRefused(checkpoint=ck, policy="ema")
        ck = averaged_checkpoint()
        del ck["actor_ema"]
        self.assertRefused(checkpoint=ck, policy="ema")

    def test_actor_structure(self):
        ck = released_checkpoint()
        ck["actor"]["obs_norm"] = (np.zeros(3), np.ones(3))
        self.assertRefused(checkpoint=ck)
        ck = released_checkpoint()
        del ck["actor"]["anchor_dec"]
        self.assertRefused(checkpoint=ck)
        ck = released_checkpoint()
        ck["actor"]["anchor_dec"] = np.zeros(9, np.float32)
        self.assertRefused(checkpoint=ck)
        self.assertRefused(cfg=run_config(actor_hidden=[8, 5]))
        self.assertRefused(cfg=run_config(actor_latent=4))
        self.assertRefused(cfg=run_config(anchor_to_decoder=False))
        self.assertRefused(cfg=run_config(actor_obs_norm=True))
        ck = released_checkpoint()
        ck["actor"]["dec"] = ck["actor"]["dec"][:-1]
        self.assertRefused(checkpoint=ck)

    def test_parameter_values(self):
        ck = released_checkpoint()
        w, b = ck["actor"]["enc"][1]
        w = w.copy()
        w[0, 0] = np.inf
        ck["actor"]["enc"][1] = (w, b)
        self.assertRefused(checkpoint=ck)
        ck = released_checkpoint()
        w, b = ck["actor"]["dec"][0]
        ck["actor"]["dec"][0] = (w.astype(np.float64), b)
        self.assertRefused(checkpoint=ck)

    def test_counts(self):
        self.assertRefused(label=LABEL + 1)
        ck = released_checkpoint()
        ck["env_steps"] += 1
        self.assertRefused(checkpoint=ck)
        ck = averaged_checkpoint()
        ck["completed_updates"] = LABEL - 1
        self.assertRefused(checkpoint=ck)
        ck = released_checkpoint()
        ck["actor_ema"] = ck["actor"]
        self.assertRefused(checkpoint=ck)  # averaged weights without a count
        self.assertRefused(cfg=run_config(policy_ema=True))
        with self.assertRaises(ValueError):
            ebp.checkpoint_label("RUN/params.pkl")
        self.assertEqual(ebp.checkpoint_label("RUN/ckpt/it_011667.pkl"), 11667)

    def test_configuration(self):
        self.assertRefused(cfg=run_config(physics="warp"))
        self.assertRefused(cfg=run_config(robot="g1_menagerie"))
        self.assertRefused(cfg=run_config(contacts="feet"))
        self.assertRefused(end_markers=None)
        for cfg in (run_config(robot="g1_menagerie"), run_config(contacts="feet")):
            with self.assertRaises(ValueError):
                ebp.compiled_action_parameters(cfg)  # refused before any model is compiled
        self.assertRefused(cfg=run_config(segments=[]))
        self.assertRefused(cfg=run_config(hist_len=5))
        self.assertRefused(cfg=run_config(obs_mode="full"))
        self.assertRefused(cfg=run_config(residual_action=None))
        self.assertRefused(cfg=run_config(motion_lib="/data/bones"))
        self.assertRefused(cfg=run_config(training_framework="other"))
        self.assertRefused(cfg=torch_run_config(git_dirty=False))  # torch_train.py records none
        self.assertRefused(cfg=torch_run_config(argv=["--seed", "0"]))
        cfg = run_config()
        del cfg["git_dirty"]
        self.assertRefused(cfg=cfg)  # train.py records it
        cfg = run_config()
        del cfg["gpu"]
        self.assertRefused(cfg=cfg)

    def test_declared_values(self):
        for kw in (dict(artifact_id=""), dict(source_artifact_id=""), dict(engine_revision=""),
                   dict(contact_capacity=0), dict(constraint_row_capacity=None)):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                self.assertRefused(**kw)

    def test_reference(self):
        self.assertRefused(man=manifest(bodies=list(BM_BODIES)[:-1]))
        self.assertRefused(man=manifest(frame_rate_hz=30.0))

    def test_action_parameters(self):
        act = action_parameters()
        self.assertRefused(act=act._replace(scale=act.scale[:-1]))
        self.assertRefused(act=act._replace(setpoint_upper=act.setpoint_lower - 1))
        self.assertRefused(act=act._replace(nominal=np.full(NU, np.nan)))

    def test_inputs_not_mutated(self):
        ck = released_checkpoint()
        before = copy.deepcopy(ck)
        export(ck)
        for x, y in zip(jax.tree_util.tree_leaves(ck["actor"]), jax.tree_util.tree_leaves(before["actor"])):
            np.testing.assert_array_equal(x, y)


def critic_export(checkpoint=None, cfg=None, **kw):
    args = dict(label=LABEL, checkpoint_name=f"it_{LABEL:06d}.pkl", run_dir="RUN", source_revision="fork-commit", nu=NU)
    args.update(kw)
    return ebp.build_critic_file(released_checkpoint() if checkpoint is None else checkpoint,
                                 run_config() if cfg is None else cfg, **args)


class CriticTests(unittest.TestCase):
    def test_arrays_are_the_kernels_transposed(self):
        ck = released_checkpoint()
        arrays, _ = critic_export(ck)
        self.assertEqual(set(arrays), {f"critic.{n}.{s}" for n in (0, 2, 4) for s in ("weight", "bias")}
                         | {"normalizer.mean", "normalizer.variance"})
        for i, (w, b) in enumerate(ck["critic"]):
            np.testing.assert_array_equal(arrays[f"critic.{2 * i}.weight"], w.T)
            np.testing.assert_array_equal(arrays[f"critic.{2 * i}.bias"], b)
        self.assertEqual(arrays["critic.0.weight"].shape, (CRITIC_HIDDEN[0], CRITIC_IN))
        self.assertEqual(arrays["critic.4.weight"].shape, (1, CRITIC_HIDDEN[-1]))
        np.testing.assert_array_equal(arrays["normalizer.mean"], ck["cnorm"][0])
        np.testing.assert_array_equal(arrays["normalizer.variance"], ck["cnorm"][1])

    def test_numpy_execution_equals_the_critic(self):
        ck = released_checkpoint()
        arrays, _ = critic_export(ck)
        x = np.random.default_rng(2).normal(size=(5, CRITIC_IN)).astype(np.float32)
        with jax.default_matmul_precision("highest"):
            want = np.asarray(critic_apply(jax.tree_util.tree_map(jnp.asarray, ck["critic"]),
                                           normalize(tuple(map(jnp.asarray, ck["cnorm"])), x)))
        h = np.clip((x - arrays["normalizer.mean"]) / np.sqrt(arrays["normalizer.variance"] + 1e-5), -10, 10)
        for j, i in enumerate((0, 2, 4)):
            h = h @ arrays[f"critic.{i}.weight"].T + arrays[f"critic.{i}.bias"]
            if j < 2:
                h = np.where(h > 0, h, np.expm1(h))
        np.testing.assert_allclose(h, want, rtol=1e-5, atol=1e-6)

    def test_declaration(self):
        _, d = critic_export()
        self.assertEqual((d["format"], d["checkpoint"], d["run"], d["completed_updates"], d["source_revision"]),
                         (ebp.CRITIC_FORMAT, f"it_{LABEL:06d}.pkl", "RUN", LABEL, "fork-commit"))
        self.assertEqual(d["input_layout"], [["proprioception", 93], ["goal", 680], ["anchor_orientation", 6],
                                             ["root_error", 4], ["anchor_relative_pose", 9], ["domain_draws", 33]])
        self.assertEqual((d["input_width"], d["hidden"], d["activation"]), (CRITIC_IN, list(CRITIC_HIDDEN), "ELU"))
        self.assertEqual((d["normalizer_count"], d["normalizer"]["stored"]), ("absent", ["mean", "variance"]))
        self.assertEqual(d["fitted_inputs"], "pre_step_state")
        json.dumps(d)

    def test_critic_width_is_the_environment_width(self):
        from nablamimic import env as env_module
        self.assertEqual(sum(w for _, w, _ in ebp.critic_input_terms(NU)),
                         env_module.PROPRIO_DIM + env_module.ANCHOR_DIM + ENV.goal_dim + 1 + 3 + (1 + NU + 3))

    def test_refusals(self):
        def refused(**kw):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                critic_export(**kw)
        refused(source_revision="")
        refused(cfg=run_config(git_commit="another-commit"))
        refused(cfg=run_config(critic_hidden=[7, 4]))
        refused(label=LABEL + 1)
        ck = released_checkpoint()
        ck["critic"] = ck["critic"][:-1]
        refused(checkpoint=ck)
        ck = released_checkpoint()
        w, b = ck["critic"][1]
        w = w.copy()
        w[0, 0] = np.nan
        ck["critic"][1] = (w, b)
        refused(checkpoint=ck)
        ck = released_checkpoint()
        ck["cnorm"] = (ck["cnorm"][0], np.zeros(CRITIC_IN, np.float32))
        refused(checkpoint=ck)
        ck = released_checkpoint()
        ck["cnorm"] = (ck["cnorm"][0][:-1], ck["cnorm"][1][:-1])
        refused(checkpoint=ck)
        ck = released_checkpoint()
        del ck["cnorm"]
        refused(checkpoint=ck)
        _, d = critic_export(cfg=run_config(git_commit="fork-commit"))
        self.assertEqual(d["run_git_commit"], "fork-commit")


class RunDirectoryTests(unittest.TestCase):
    """main() on temporary files; compiled_action_parameters (Menagerie) is replaced by the synthetic values."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        (self.run / "ckpt").mkdir(parents=True)
        (self.run / "config.json").write_text(json.dumps(run_config()))
        self.checkpoint = self.run / "ckpt" / f"it_{LABEL:06d}.pkl"
        with open(self.checkpoint, "wb") as f:
            pickle.dump(released_checkpoint(), f)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(manifest(artifact_id="reference-from-file")))

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, out, run_config_arg=None, checkpoint=None, extra=()):
        argv = ["--checkpoint", str(checkpoint or self.checkpoint), "--run-config", str(run_config_arg or self.run),
                "--policy", "live", "--artifact-id", "policy-from-main", "--source-artifact-id", "checkpoint-from-main",
                "--reference-manifest", str(self.manifest), "--engine-revision", "engine-from-main",
                "--engine-contact-capacity", "64", "--engine-constraint-row-capacity", "288", "--out", str(out),
                *extra]
        with mock.patch.object(ebp, "compiled_action_parameters", return_value=action_parameters()) as compiled, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ebp.main(argv)
        return compiled

    def declaration(self, out):
        with np.load(out, allow_pickle=False) as z:
            return set(z.files), json.loads(z["declaration"].item())

    def test_directory_and_file_resolve_to_the_same_run(self):
        out = self.root / "from-directory.npz"
        compiled = self.main(out)
        self.assertEqual(compiled.call_args.args[0], run_config())
        files, d = self.declaration(out)
        self.assertEqual(len(files), 13)
        src = d["policy_source"]
        self.assertEqual(src["source_run_configuration"], str((self.run / "config.json").resolve()))
        self.assertEqual(src["source_checkpoint"], str(self.checkpoint.resolve()))
        self.assertEqual((src["checkpoint_label"], src["run_end_markers"]), (LABEL, {}))
        self.assertEqual((d["artifact_id"], src["source_artifact_id"]), ("policy-from-main", "checkpoint-from-main"))
        self.assertEqual(d["training_domain"]["reference"]["artifact_id"], "reference-from-file")
        self.assertEqual(d["training_domain"]["reference"]["clips"], ["dance1_subject1", "walk1_subject1"])
        self.assertEqual(d["training_domain"]["engine"]["declared_revision"], "engine-from-main")
        out_file = self.root / "from-file.npz"
        self.main(out_file, run_config_arg=self.run / "config.json")
        self.assertEqual(self.declaration(out_file)[1], d)

    def test_critic_file(self):
        out, critic = self.root / "policy.npz", self.root / "critic.npz"
        self.main(out, extra=["--critic-out", str(critic), "--source-revision", "fork-commit"])
        with np.load(critic, allow_pickle=False) as z:
            self.assertEqual(set(z.files), {f"critic.{n}.{s}" for n in (0, 2, 4) for s in ("weight", "bias")}
                             | {"normalizer.mean", "normalizer.variance", "format", "declaration"})
            self.assertEqual(z["format"].item(), ebp.CRITIC_FORMAT)
            d = json.loads(z["declaration"].item())
            for x in z.files:
                if x not in ("format", "declaration"):
                    self.assertEqual(z[x].dtype, np.float32)
        self.assertEqual((d["checkpoint"], d["run"], d["normalizer_count"]),
                         (self.checkpoint.name, str(self.run.resolve()), "absent"))
        for extra in (["--critic-out", str(self.root / "c2.npz")], ["--source-revision", "fork-commit"],
                      ["--critic-out", str(critic), "--source-revision", "fork-commit"],
                      ["--critic-out", str(self.root / "p3.npz"), "--source-revision", "fork-commit"]):
            p3 = self.root / "p3.npz"
            with self.assertRaises(SystemExit):
                self.main(p3, extra=extra)
            self.assertFalse(p3.exists())
            self.assertFalse((self.root / "c2.npz").exists())

    def test_existing_output_refused(self):
        out = self.root / "policy.npz"
        out.write_bytes(b"earlier output")
        with self.assertRaises(SystemExit):
            self.main(out)
        self.assertEqual(out.read_bytes(), b"earlier output")
        with self.assertRaises(SystemExit):
            self.main(self.root / "policy.npy")

    def test_checkpoint_outside_the_run_refused(self):
        other = self.root / "other" / "ckpt"
        other.mkdir(parents=True)
        moved = other / self.checkpoint.name
        moved.write_bytes(self.checkpoint.read_bytes())
        with self.assertRaises(SystemExit):
            self.main(self.root / "policy.npz", checkpoint=moved)
        self.assertFalse((self.root / "policy.npz").exists())

    def test_markers(self):
        (self.run / "DONE").write_text(str(LABEL))
        halted = dict(iter=LABEL, grad_norm=2e6, wall_s=3.0, env_steps=LABEL * PER_UPDATE, rule="SHAC guard")
        (self.run / "HALTED").write_text(json.dumps(halted) + "\n")
        out = self.root / "halted.npz"
        self.main(out)
        self.assertEqual(self.declaration(out)[1]["policy_source"]["run_end_markers"],
                         dict(DONE=dict(last_update=LABEL), HALTED=halted))
        (self.run / "DIVERGED").write_text(str(LABEL))
        with self.assertRaises(ValueError):
            self.main(self.root / "diverged.npz")
        self.assertFalse((self.root / "diverged.npz").exists())

    def test_torch_run_directory(self):
        (self.run / "config.json").write_text(json.dumps(torch_run_config()))
        (self.run / "DONE").write_text("completed\n")
        out = self.root / "torch.npz"
        self.main(out)
        d = self.declaration(out)[1]
        self.assertEqual(d["policy_source"]["run_end_markers"], dict(DONE=dict(content="completed")))
        self.assertEqual(d["training_domain"]["source_run"]["training_framework"], "torch")
        (self.run / "DONE").write_text("12\n")  # a train.py marker in a torch_train.py run
        with self.assertRaises(ValueError):
            self.main(self.root / "torch-mixed.npz")
        self.assertFalse((self.root / "torch-mixed.npz").exists())
        (self.run / "config.json").write_text(json.dumps(run_config()))
        (self.run / "DONE").write_text("completed\n")  # and the converse
        with self.assertRaises(ValueError):
            self.main(self.root / "jax-mixed.npz")


@unittest.skipIf(not SCENE.exists(), f"no Menagerie checkout at {SCENE} (scripts/setup_menagerie.sh)")
class CompiledModelTests(unittest.TestCase):
    def test_action_parameters_are_the_environment_values(self):
        from nablamimic.env import EnvConfig, TrackingEnv
        from nablamimic.motion import MotionLibrary
        frames, nb = 3, len(BM_BODIES)
        qpos = np.zeros((frames, 36), np.float32)
        qpos[:, 3] = 1.0
        quat = np.zeros((frames, nb, 4), np.float32)
        quat[..., 0] = 1.0
        lib = MotionLibrary(qpos=qpos, qvel=np.zeros((frames, 35), np.float32),
                            body_pos=np.zeros((frames, nb, 3), np.float32), last=np.full(frames, frames - 1),
                            names=["still"], starts=np.array([0]), fps=50.0, bodies=BM_BODIES, body_quat=quat,
                            body_lin_vel=np.zeros((frames, nb, 3), np.float32),
                            body_ang_vel=np.zeros((frames, nb, 3), np.float32))
        env = TrackingEnv(lib, EnvConfig(residual_action=True))
        act = ebp.compiled_action_parameters(run_config())
        self.assertEqual(len(act.joint_names), env.nu)
        np.testing.assert_array_equal(np.asarray(act.nominal, np.float32), np.asarray(env.q0))
        np.testing.assert_array_equal(np.asarray(act.scale, np.float32), np.asarray(env.action_scale))
        np.testing.assert_array_equal(np.asarray(act.setpoint_lower, np.float32), np.asarray(env.ctrl_lo))
        np.testing.assert_array_equal(np.asarray(act.setpoint_upper, np.float32), np.asarray(env.ctrl_hi))
        np.testing.assert_array_equal(act.kp, env.m.actuator_gainprm[:, 0])
        self.assertEqual(act.joint_names[0], "left_hip_pitch_joint")
        _, d = export(act=act)
        self.assertEqual(d["action"]["joint_names"], list(act.joint_names))


if __name__ == "__main__":
    unittest.main()
