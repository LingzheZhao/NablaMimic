"""The BM policy file exporter on a tiny synthetic checkpoint (CPU; the compiled-model part needs Menagerie)."""

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic import export_bm_policy as ebp
from nablamimic.env import BM_BODIES, ActionParameters
from nablamimic.motion import SCENE
from nablamimic.nets import actor_init, actor_apply

HIDDEN, LATENT, NU = (8, 4), 3, 29
ENV = types.SimpleNamespace(goal_dim=10 * 68 + 6, hist_dim=10 * 93, nu=NU)  # the real input widths
LABEL, PER_UPDATE = 12, 16 * 1 * 4


def numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, np.float32), jax.device_get(tree))


def released_checkpoint(seed=0):
    """A checkpoint as the released recipe saves it: no evaluation policy, completed or EMA counts."""
    actor = numpy_tree(actor_init(jax.random.PRNGKey(seed), ENV, hidden=HIDDEN, latent=LATENT, anchor_to_decoder=6))
    return dict(actor=actor, critic=[], iter=LABEL, wall_s=1.0, env_steps=LABEL * PER_UPDATE)


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
               git_dirty=False, gpu="test card", argv=["--residual-action"],
               segments=[dict(engine=dict(version="0.1.0", key="library-key", arch="sm_80"))])
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
                contact_capacity=64, constraint_row_capacity=288)
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
        self.assertRefused(cfg=run_config(segments=[]))
        self.assertRefused(cfg=run_config(hist_len=5))
        self.assertRefused(cfg=run_config(obs_mode="full"))
        self.assertRefused(cfg=run_config(residual_action=None))
        self.assertRefused(cfg=run_config(motion_lib="/data/bones"))
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
