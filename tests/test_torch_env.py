"""Fresh synthetic task-math tests; no assets, simulator, GPU or historical outputs."""

from types import SimpleNamespace
import unittest
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch backend is not installed")

import numpy as np
import torch

from nablamimic.torch_env import (COMPACT_KEYS, STATE_KEYS, TorchTrackingEnv, abs_real, gravity_b,
                                 quat_angle_sq, quat_mul, quat_rotate, quat_to_matrix, select_reset)


def fixture(*, residual=False, dr=False, obs_noise=False, reset_noise=False):
    frames, bodies = 12, 14
    qpos = np.zeros((frames, 36), np.float32)
    qpos[:, 2], qpos[:, 3] = 1, 1
    qvel = np.zeros((frames, 35), np.float32)
    offsets = torch.zeros((bodies, 3))
    offsets[:, 0] = torch.arange(bodies)*0.1
    positions = qpos[:, None, :3] + offsets.numpy()[None]
    quaternions = np.zeros((frames, bodies, 4), np.float32)
    quaternions[..., 0] = 1
    cfg = SimpleNamespace(residual_action=residual, action_rate_space="setpoint", solver_iterations=10,
                          dr=dr, obs_noise=obs_noise, reset_noise=reset_noise, term_height=0.25, term_tilt=0.8)
    reference = SimpleNamespace(m=SimpleNamespace(nq=36, nv=35, nu=29), cfg=cfg, nu=29,
        q0=np.zeros(29), action_scale=np.full(29, 0.5), ctrl_lo=np.full(29, -2), ctrl_hi=np.full(29, 2),
        soft_lo=np.full(29, -1), soft_hi=np.full(29, 1), body_ids=np.arange(bodies), anchor_k=1,
        torso_id=1, root_com_id=0, foot_pairs=np.array([0, 1]), starts=np.array([0, 6]), lengths=np.array([6, 6]),
        ref_qpos=qpos, ref_qvel=qvel, ref_body_pos=positions, ref_body_quat=quaternions,
        ref_body_lin=np.zeros((frames, bodies, 3)), ref_body_ang=np.zeros((frames, bodies, 3)),
        ref_last=np.repeat([5, 11], 6), goal_feat=np.arange(frames*68, dtype=np.float32).reshape(frames, 68))

    def kinematics(q, v, com):
        normalized = q[:, 3:7]/torch.linalg.vector_norm(q[:, 3:7], dim=-1, keepdim=True)
        q = torch.cat((q[:, :3], normalized, q[:, 7:]), -1)
        xpos = q[:, None, :3] + quat_rotate(offsets[None], normalized[:, None])
        xquat = normalized[:, None].expand(-1, bodies, -1)
        # The reference is stationary; velocity-specific assertions set cvel explicitly.
        cvel = torch.zeros((len(q), bodies, 6), dtype=q.dtype, device=q.device)
        subtree_com = q[:, None, :3].expand(-1, bodies, -1)
        return q, xpos, xquat, cvel, subtree_com

    calls = []
    def step(q, v, warm, ctrl, friction, com, **kwargs):
        calls.append((ctrl.detach().clone(), kwargs))
        qn = torch.cat((q[:, :7], q[:, 7:]+0.01*ctrl), -1)
        return (qn, v, warm, *kinematics(qn, v, com))

    env = TorchTrackingEnv.from_reference(reference, device="cpu", step_fn=step, kinematics_fn=kinematics)
    return env, calls


class TorchEnvironmentTest(unittest.TestCase):
    def test_nominal_reset_and_reference_goal_do_not_cross_clip_boundary(self):
        env, _ = fixture()
        state = env.reset(torch.tensor([4, 10]))
        self.assertEqual(tuple(state), STATE_KEYS)
        self.assertEqual(state["frame"].dtype, torch.int32)
        self.assertEqual(state["hist"].shape, (2, 10, 93))
        h, goal = env.policy_obs(state)
        self.assertEqual(h.shape, (2, 930))
        self.assertEqual(goal.shape, (2, 686))
        self.assertEqual(env.critic_obs(state).shape, (2, 825))
        torch.testing.assert_close(goal[0, :68], env.goal_feat[4])
        torch.testing.assert_close(goal[0, 68:136], env.goal_feat[5])
        torch.testing.assert_close(goal[0, 9*68:10*68], env.goal_feat[5])
        torch.testing.assert_close(goal[:, -6:], torch.tensor([[1., 0., 0., 0., 1., 0.]]).expand(2, -1))
        reward, terms = env.reward(state, state["prev_a"], state["prev_a"], prev=state)
        torch.testing.assert_close(reward, torch.full((2,), env.max_reward))
        self.assertEqual(set(terms), {"root_pos", "root_ori", "body_pos", "body_ori", "body_lin_vel",
                                     "body_ang_vel", "action_rate", "joint_limit"})

    def test_proprioception_order_scales_and_critic_context(self):
        env, _ = fixture()
        state = env.reset(torch.tensor([0]))
        state["qpos"][:, 7:] = 0.2
        state["qvel"][:, 6:] = 5.0
        state["qvel"][:, 3:6] = 2.0
        state["enc_bias"].fill_(0.01)
        state["com"].fill_(0.05)
        state["prev_a"].fill_(0.3)
        proprio = env.proprio(state)
        torch.testing.assert_close(proprio[:, :29], torch.full((1, 29), .21))
        torch.testing.assert_close(proprio[:, 29:61], torch.ones((1, 32)))
        torch.testing.assert_close(proprio[:, 61:64], torch.tensor([[0., 0., -1.]]))
        torch.testing.assert_close(proprio[:, 64:], torch.full((1, 29), .3))
        critic = env.critic_obs(state)
        torch.testing.assert_close(critic[:, -32:], torch.ones((1, 32)))
        self.assertAlmostEqual(float(critic[0, -33]), .25/.45, places=6)

    def test_residual_setpoint_change_accounts_for_reference_motion_and_reset_baseline(self):
        env, _ = fixture(residual=True)
        env.ref_qpos[1, 7:] = .1
        env.ref_qpos[2, 7:] = .3
        prev = env.reset(torch.tensor([1]))
        action = torch.zeros((1, 29))
        prev["ep"].fill_(7)
        # Random initial episode ages still use the reset baseline q0.
        torch.testing.assert_close(env.action_change(action, action, prev), torch.full_like(action, .6))
        prev["hist"][:, 1] += .1
        torch.testing.assert_close(env.action_change(action, action, prev), torch.full_like(action, .4))
        _, terms = env.reward(prev, action, action, prev=prev)
        self.assertAlmostEqual(float(terms["action_rate"]), -.002*29*.4**2, places=7)
        torch.testing.assert_close(env.ctrl_of(action, prev["frame"]), torch.full_like(action, .3))

    def test_step_uses_native_control_clock_and_updates_history_once(self):
        env, calls = fixture()
        state = env.reset(torch.tensor([0]))
        old_history = state["hist"].clone()
        action = torch.full((1, 29), .2, requires_grad=True)
        nxt = env.step(state, action)
        self.assertEqual(int(nxt["frame"][0]), 1)
        self.assertEqual(int(nxt["ep"][0]), 1)
        torch.testing.assert_close(nxt["hist"][:, :-1], old_history[:, 1:])
        torch.testing.assert_close(nxt["hist"][:, -1, -29:], action)
        torch.testing.assert_close(calls[0][0], torch.full((1, 29), .1))
        self.assertEqual(calls[0][1], dict(iterations=10, substeps=4, clamp_kv=0, normal=3,
                                         clamp_soft=None, grad_decay=1.0, kinematics=True))
        nxt["hist"].sum().backward()
        self.assertTrue(bool(torch.isfinite(action.grad).all()))
        self.assertGreater(float(action.grad.abs().sum()), 0)

    def test_random_inputs_are_explicit_and_scaled_once(self):
        env, _ = fixture(dr=True, obs_noise=True, reset_noise=True)
        with self.assertRaisesRegex(ValueError, "explicit"):
            env.reset(torch.tensor([0]))
        noise = {key: torch.ones_like(value) for key, value in env.sample_reset_noise(1).items()}
        state = env.reset(torch.tensor([0]), noise)
        torch.testing.assert_close(state["qpos"][:, :3], torch.tensor([[.05, .05, 1.01]]))
        torch.testing.assert_close(state["qpos"][:, 7:], torch.full((1, 29), .1))
        torch.testing.assert_close(state["qvel"][:, :6], torch.tensor([[.5, .5, .2, .52, .52, .78]]))
        torch.testing.assert_close(state["friction"], torch.tensor([1.2]))
        torch.testing.assert_close(state["enc_bias"], torch.full((1, 29), .01))
        torch.testing.assert_close(state["com"], torch.tensor([[.025, .05, .05]]))
        torch.testing.assert_close(state["hist"][:, 0]-env.proprio(state), env.noise_limits[None])
        noisy_goal = env.goal(state, torch.ones((1, 6)))
        clean_anchor = env.anchor_rel(state)[:, 3:9]
        torch.testing.assert_close(noisy_goal[:, -6:]-clean_anchor, torch.full((1, 6), .05))

    def test_body_origin_velocity_and_termination_are_physical_state_quantities(self):
        env, _ = fixture()
        state = env.reset(torch.tensor([0, 0]))
        state["cvel"] = state["cvel"].clone()
        state["cvel"][..., 2] = 1  # world angular z
        linear, angular = env.body_velocity(state)
        torch.testing.assert_close(angular[..., 2], torch.ones((2, 14)))
        torch.testing.assert_close(linear[0, :, 1], torch.arange(14)*.1)
        state["qpos"][0, 2] = 1.25
        state["qpos"][1, 2] = 1.2501
        torch.testing.assert_close(env.terminated(state), torch.tensor([False, True]))
        state["qpos"][:, 2] = 1
        state["qpos"][0, 3:7] = torch.tensor([0., 1., 0., 0.])
        state["qvel"][1, 0] = float("nan")
        torch.testing.assert_close(env.terminated(state), torch.tensor([True, True]))

    def test_compact_restore_and_detached_reset_selection(self):
        env, _ = fixture()
        state = env.reset(torch.tensor([0, 0]))
        compact = env.compact(state)
        self.assertEqual(tuple(compact), COMPACT_KEYS)
        self.assertEqual(compact["hist"].dtype, torch.float32)
        q = compact["qpos"].clone().requires_grad_(True)
        restored = env.restore({**compact, "qpos": q})
        self.assertEqual(tuple(restored), STATE_KEYS)
        reset = env.reset(torch.tensor([1, 1]))
        selected = select_reset(torch.tensor([True, False]), reset, restored)
        selected["xpos"].sum().backward()
        torch.testing.assert_close(q.grad[0], torch.zeros_like(q.grad[0]))
        self.assertGreater(float(q.grad[1, :3].abs().sum()), 0)

    def test_quaternion_rotation_and_zero_angle_gradient(self):
        q = torch.tensor([[2**-.5, 0., 0., 2**-.5]], requires_grad=True)
        torch.testing.assert_close(quat_rotate(torch.tensor([[1., 0., 0.]]), q),
                                   torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=0)
        angle = quat_angle_sq(q, q.detach()).sum()
        angle.backward()
        self.assertTrue(bool(torch.isfinite(q.grad).all()))
        torch.testing.assert_close(gravity_b(torch.tensor([[1., 0., 0., 0.]])), torch.tensor([[0., 0., -1.]]))

    def test_window_reuses_reset_relative_proprioception_noise(self):
        from nablamimic import torch_nets as nets
        from nablamimic.torch_train import make_control_body, make_window

        env, _ = fixture(obs_noise=True)
        batch, horizon = 2, 6
        reset_noise = {key: torch.zeros_like(value) for key, value in env.sample_reset_noise(batch).items()}
        start = env.reset(torch.zeros(batch, dtype=torch.int32), reset_noise)
        reset = env.reset(torch.zeros(batch, dtype=torch.int32), reset_noise)
        # Tiny functional policies exercise the real control body and its resets.
        actor = dict(enc=[(torch.zeros(env.goal_dim-6, 1), torch.zeros(1))],
                     dec=[(torch.zeros(1+env.hist_dim+6, env.nu), torch.zeros(env.nu))],
                     anchor_dec=torch.zeros(6))
        critic = [(torch.zeros(env.critic_dim, 1), torch.zeros(1))]
        normalizer = (torch.zeros(env.critic_dim), torch.ones(env.critic_dim))
        def encode(parameters, frame, reset_frame):
            def sequence(f):
                frames = torch.minimum(f[None]+torch.arange(horizon)[:, None], env.ref_last[f][None])
                return nets.actor_encode(parameters, env.goal_ref(frames.flatten())).reshape(horizon, batch, 1)
            return sequence(frame), sequence(reset_frame)
        body = make_control_body(env, .99, 2, 3, None)
        window = make_window(env, actor, critic, normalizer, horizon, .99, body, encode)
        noise = torch.empty((2, horizon, batch, 93))
        for stream in range(2):
            for t in range(horizon):
                noise[stream, t] = (.1+stream*.4+t*.05)
        action_noise = torch.zeros((horizon, batch, env.nu))
        _, gradient, _, records = window(start, reset, action_noise, noise,
            torch.zeros((horizon, batch, 6)), torch.zeros_like(action_noise, requires_grad=True),
            torch.zeros(2, requires_grad=True))
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        torch.testing.assert_close(records['done'], torch.tensor([[False, False], [True, True]]*3))
        history = records['hist'].float().reshape(horizon, batch, 10, 93)
        # Both two-step episodes use reset.key's first post-reset noise draw.
        torch.testing.assert_close(history[3], history[5], rtol=0, atol=0)
        self.assertFalse(torch.equal(history[1, :, -1], history[3, :, -1]))
        torch.testing.assert_close(history[2], reset['hist'].to(torch.bfloat16).float(), rtol=0, atol=0)
        expected = (env.proprio(reset)+noise[1, 0]*env.noise_limits).to(torch.bfloat16).float()
        torch.testing.assert_close(history[3, :, -1], expected, rtol=0, atol=0)


class DonorMathParityTest(unittest.TestCase):
    """Compare current JAX task formulas on fresh state tensors, without a simulator."""

    def test_quaternion_rotation_and_matrix_values_and_full_jacobians(self):
        import jax
        import jax.numpy as jnp
        from mujoco.mjx._src import math as mjm

        generic = np.array([.8, -.3, .2, .4], np.float32)
        generic /= np.linalg.norm(generic)
        quaternions = (np.array([1., 0., 0., 0.], np.float32),
                       np.array([0., 1., 0., 0.], np.float32), generic, 1.1*generic)
        vector = torch.tensor([.3, -.7, 1.2])
        jvector = jnp.asarray(vector.numpy())
        for value in quaternions:
            with self.subTest(quaternion=value.tolist()):
                quaternion = torch.tensor(value, requires_grad=True)
                jquaternion = jnp.asarray(value)
                for torch_fn, jax_fn in ((lambda q: quat_rotate(vector, q), lambda q: mjm.rotate(jvector, q)),
                                         (quat_to_matrix, mjm.quat_to_mat)):
                    np.testing.assert_allclose(torch_fn(quaternion).detach().numpy(), np.asarray(jax_fn(jquaternion)),
                                               rtol=1e-6, atol=1e-6)
                    actual = torch.autograd.functional.jacobian(torch_fn, quaternion)
                    expected = jax.jacrev(jax_fn)(jquaternion)
                    np.testing.assert_allclose(actual.numpy(), np.asarray(expected), rtol=1e-6, atol=1e-6)
                actual = torch.autograd.functional.jacobian(lambda v: quat_rotate(v, quaternion), vector)
                expected = jax.jacrev(lambda v: mjm.rotate(v, jquaternion))(jvector)
                np.testing.assert_allclose(actual.numpy(), np.asarray(expected), rtol=1e-6, atol=1e-6)

    def test_absolute_value_forward_bits_and_boundary_gradient(self):
        import jax
        import jax.numpy as jnp

        values = np.array([-2., -0., 0., 2.], np.float32)
        x = torch.tensor(values, requires_grad=True)
        result = abs_real(x)
        result.sum().backward()
        expected = np.asarray(jnp.abs(jnp.asarray(values)))
        np.testing.assert_array_equal(result.detach().numpy().view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(x.grad.numpy(), np.asarray(jax.grad(lambda y: jnp.abs(y).sum())(values)))
        self.assertEqual(float(x.grad[1]), 1.)
        self.assertEqual(float(x.grad[2]), 1.)

    def test_zero_tracking_error_and_half_turn_gradient_match_donor(self):
        import jax
        import jax.numpy as jnp
        from nablamimic.env import TrackingEnv, _quat_angle_sq

        env, _ = fixture()
        state = env.reset(torch.tensor([0]))
        tilted = torch.tensor([np.cos(.2), np.sin(.2), 0., 0.], dtype=torch.float32)
        env.ref_qpos[0, 3:7] = tilted
        q = state['qpos'].clone()
        q[0, 3:7] = tilted
        q.requires_grad_(True)
        hazard = env.hazard_features({**state, 'qpos': q})
        (hazard[0, 0]+hazard[0, 2]).backward()
        donor = object.__new__(TrackingEnv)
        donor.cfg = env.cfg
        donor.ref_qpos = jnp.asarray(env.ref_qpos.numpy())
        donor.ref_qvel = jnp.asarray(env.ref_qvel.numpy())
        def margins(qpos):
            dx = SimpleNamespace(qpos=qpos, qvel=jnp.zeros(35))
            value = donor.hazard_features(SimpleNamespace(dx=dx, frame=jnp.int32(0)))
            return value[0]+value[2]
        expected = jax.grad(margins)(jnp.asarray(q.detach().numpy()[0]))
        np.testing.assert_allclose(q.grad.numpy()[0], np.asarray(expected), rtol=1e-6, atol=1e-6)
        self.assertEqual(float(q.grad[0, 2]), -1.)
        half_turn = torch.tensor([[0., 1., 0., 0.]], requires_grad=True)
        identity = torch.tensor([[1., 0., 0., 0.]])
        quat_angle_sq(half_turn, identity).sum().backward()
        expected = jax.grad(_quat_angle_sq, argnums=0)(jnp.asarray(half_turn.detach().numpy()[0]),
                                                     jnp.asarray(identity.numpy()[0]))
        np.testing.assert_allclose(half_turn.grad.numpy()[0], np.asarray(expected), rtol=1e-6, atol=1e-6)
        self.assertLess(float(half_turn.grad[0, 0]), -12.)

    def test_current_donor_observations_reward_and_body_position_gradient(self):
        import jax
        import jax.numpy as jnp
        from nablamimic.env import State, TrackingEnv, Xi

        env, _ = fixture(residual=True)
        state = env.reset(torch.tensor([2]))
        q = state["qpos"].clone()
        q[:, :3] += torch.tensor([[.15, -.1, .08]])
        yaw = torch.tensor([[np.cos(.2), 0., 0., np.sin(.2)]], dtype=torch.float32)
        tilt = torch.tensor([[np.cos(.1), np.sin(.1), 0., 0.]], dtype=torch.float32)
        q[:, 3:7] = quat_mul(yaw, tilt)
        q[:, 7:] = torch.linspace(-.1, .2, 29)
        v = state["qvel"].clone()
        v[:, :6] = torch.tensor([[.2, -.1, .3, .4, .2, -.3]])
        state = env.restore({**env.compact(state), "qpos": q, "qvel": v})
        state["hist"] = torch.arange(930, dtype=torch.float32).reshape(1, 10, 93)*.001
        state["ep"] = torch.tensor([7], dtype=torch.int32)
        state["prev_a"] = torch.linspace(-.2, .1, 29)[None]
        action = torch.linspace(.1, -.1, 29)[None]
        donor = object.__new__(TrackingEnv)
        donor.cfg = env.cfg
        for name in ("nu", "anchor_k", "torso_id", "root_com_id", "hist_dim", "goal_dim", "critic_dim"):
            setattr(donor, name, getattr(env, name))
        for name in (*env.TABLES, "q0", "action_scale", "ctrl_lo", "ctrl_hi", "soft_lo", "soft_hi", "body_ids"):
            setattr(donor, name, jnp.asarray(getattr(env, name).cpu().numpy()))

        def donor_state(torch_state, xpos=None):
            array = lambda key: jnp.asarray(torch_state[key][0].detach().cpu().numpy())
            dx = SimpleNamespace(**{key: array(key) for key in ("qpos", "qvel", "xpos", "xquat", "cvel", "subtree_com")})
            if xpos is not None:
                dx.xpos = xpos
            return State(dx, array("frame"), array("ep"), array("hist"), array("prev_a"),
                         Xi(array("friction"), array("enc_bias"), array("com")), jax.random.PRNGKey(0))

        js = donor_state(state)
        ja = jnp.asarray(action[0].numpy())
        torch_h, torch_g = env.policy_obs(state)
        jax_h, jax_g = donor.policy_obs(js, jax.random.PRNGKey(1))
        for actual, expected in ((torch_h[0], jax_h), (torch_g[0], jax_g),
                                 (env.critic_obs(state)[0], donor.critic_obs(js)),
                                 (env.hazard_features(state)[0], donor.hazard_features(js))):
            np.testing.assert_allclose(actual.detach().numpy(), np.asarray(expected), rtol=3e-6, atol=3e-6)
        tr, tt = env.reward(state, action, state["prev_a"], prev=state)
        jr, jt = donor.reward(js, ja, js.prev_a, prev=js)
        np.testing.assert_allclose(tr.detach().numpy(), np.asarray(jr)[None], rtol=3e-6, atol=3e-6)
        for key in tt:
            np.testing.assert_allclose(tt[key].detach().numpy(), np.asarray(jt[key])[None], rtol=3e-6, atol=3e-6)
        xpos = state["xpos"].detach().clone().requires_grad_(True)
        env.reward({**state, "xpos": xpos}, action, state["prev_a"], prev=state)[0].sum().backward()
        jgrad = jax.grad(lambda pos: donor.reward(donor_state(state, pos), ja, js.prev_a, prev=js)[0])(js.dx.xpos)
        np.testing.assert_allclose(xpos.grad[0].numpy(), np.asarray(jgrad), rtol=1e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
