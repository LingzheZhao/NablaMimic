"""CPU contract checks; real simulator parity is measured separately on the cluster."""

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from nablamimic import sim, warp_physics


class AlgebraicSimulator:
    def __init__(self, model, nworld, foot_pairs, torso, nconmax, njmax, forward_only=False):
        self.nworld, self.nv = nworld, 2
        self.forward_only = forward_only
        self.df = SimpleNamespace(**{name: np.zeros((nworld, 2), np.float32)
                                     for name in ("qpos", "qvel", "qacc_warmstart", "ctrl")})
        self.calls = []

    def forward(self, *args):
        self.calls.append("forward")

    def backward(self, *args):
        self.calls.append("backward")
        if self.forward_only:
            raise AssertionError("Evaluation initialized the unused adjoint")


class ForwardOnlyTest(unittest.TestCase):
    def construct(self, forward_only):
        registered = []

        def callable_graph(callback, **kwargs):
            registered.append(callback.__name__)
            if callback.__name__ == "fwd_cb":
                return lambda q, v, w, u, fr, com, **kw: (q + u, v + 2 * u, w)
            if callback.__name__ == "bwd_cb":
                return lambda q, v, w, u, fr, com, gq, gv, **kw: (gq, gv, gq + 2 * gv)
            return lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected callback"))

        with ExitStack() as stack:
            patches = (
                (warp_physics, "_Sim", AlgebraicSimulator),
                (warp_physics, "mjx_capacity", lambda model: (16, 32)),
                (warp_physics.mjw, "enable_grad", lambda: None),
                (warp_physics.wp, "get_device", lambda name: "fake-device"),
                (warp_physics.wp, "ScopedDevice", lambda dev: nullcontext()),
                (warp_physics.wp, "synchronize_device", lambda dev: None),
                (warp_physics.wp, "zeros_like", lambda value, **kwargs: np.zeros_like(value)),
                (warp_physics.wp, "full", lambda n, value, **kwargs: np.full(n, value)),
                (warp_physics.wp, "zeros", lambda shape, **kwargs: np.zeros(shape)),
                (warp_physics.wp, "jax_callable", callable_graph),
            )
            for module, name, replacement in patches:
                stack.enter_context(patch.object(module, name, replacement))
            physics = warp_physics.WarpPhysics(SimpleNamespace(nv=2), 1, [], 0,
                                               forward_only=forward_only)
        return physics, registered

    def test_forward_only_preserves_forward_results_and_skips_adjoint_warmup(self):
        full, full_callbacks = self.construct(False)
        forward, forward_callbacks = self.construct(True)
        args = (jnp.ones((1, 2)), jnp.ones((1, 2)) * 2, jnp.ones((1, 2)) * 3,
                jnp.ones((1, 2)) * 4, jnp.ones((1,)), jnp.ones((1, 3)))
        expected = jax.jit(full.step)(*args)
        actual = jax.jit(forward.step)(*args)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        self.assertEqual(forward.sim.calls, ["forward"])
        self.assertEqual(full.sim.calls, ["forward", "backward"])
        self.assertEqual(forward_callbacks, ["fwd_cb"])
        self.assertIn("bwd_cb", full_callbacks)
        self.assertEqual((forward.nconmax, forward.njmax), (full.nconmax, full.njmax))

    def test_forward_only_rejects_both_differentiation_modes(self):
        forward, _ = self.construct(True)
        full, _ = self.construct(False)
        q = jnp.ones((1, 2))
        fr, com = jnp.ones((1,)), jnp.ones((1, 3))

        def objective(physics, action):
            q2, v2, _ = physics.step(q, q, q, action, fr, com)
            return (q2 + v2).sum()

        np.testing.assert_array_equal(jax.grad(lambda u: objective(full, u))(q), jnp.full_like(q, 3.0))
        with self.assertRaisesRegex(RuntimeError, "cannot be differentiated"):
            jax.grad(lambda u: objective(forward, u))(q)
        with self.assertRaisesRegex(RuntimeError, "cannot be differentiated"):
            jax.jvp(lambda u: objective(forward, u), (q,), (q,))

    def test_forward_only_allocates_only_the_forward_physics_data(self):
        model = SimpleNamespace(nv=2, body_ipos=np.zeros((1, 3)))
        warp_model = SimpleNamespace(opt=SimpleNamespace(warn_overflow=1))
        data = object()
        with ExitStack() as stack:
            stack.enter_context(patch.object(warp_physics.mujoco, "MjData", return_value=object()))
            stack.enter_context(patch.object(warp_physics.mujoco, "mj_forward"))
            stack.enter_context(patch.object(warp_physics.mjw, "put_model", return_value=warp_model))
            allocate = stack.enter_context(patch.object(warp_physics.mjw, "put_data", side_effect=[data]))
            stack.enter_context(patch.object(warp_physics.mjw, "create_backward_context",
                                             side_effect=AssertionError("Unused adjoint allocation")))
            stack.enter_context(patch.object(warp_physics.wp, "array", side_effect=lambda x, **kw: x))
            instance = warp_physics._Sim(model, 40, [], 0, 64, 288, forward_only=True)
        self.assertIs(instance.df, data)
        self.assertEqual(allocate.call_count, 1)
        self.assertFalse(hasattr(instance, "D"))
        self.assertFalse(hasattr(instance, "bc"))
        self.assertFalse(hasattr(instance, "m_ws"))
        self.assertEqual(warp_model.opt.warn_overflow, 0)

    def test_training_cannot_select_forward_only_physics(self):
        with self.assertRaisesRegex(ValueError, "Training requires"):
            sim.make_warp(None, [1], train=True, forward_only=True)


if __name__ == "__main__":
    unittest.main()
