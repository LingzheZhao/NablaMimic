"""CPU tests for state isolation and compiled-session lifetime, without physics."""

from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from nablamimic import evaluate, evaluation_state, warp_physics


class FakeArray:
    def __init__(self, values):
        self.values = np.array(values)
        self.ptr = id(self)
        self.dtype = self.values.dtype
        self.device = "cpu"

    @property
    def shape(self):
        return self.values.shape

    @property
    def strides(self):
        return self.values.strides

    @property
    def size(self):
        return self.values.size

    def numpy(self):
        return self.values.copy()


@dataclass
class NativeModel:
    iterations: int
    friction: FakeArray
    branches: tuple


@dataclass
class NativeData:
    overflow: FakeArray
    warm: FakeArray
    contact: dict


class PersistentStateTest(unittest.TestCase):
    def fixture(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(evaluation_state.wp, "array", FakeArray))
        stack.enter_context(patch.object(evaluation_state.wp, "clone",
                                         side_effect=lambda value, **kw: FakeArray(value.values)))
        stack.enter_context(patch.object(evaluation_state.wp, "copy",
                                         side_effect=lambda dst, src: np.copyto(dst.values, src.values)))
        stack.enter_context(patch.object(evaluation_state.wp, "synchronize_device"))
        shared = FakeArray([3.0, 4.0])
        model = NativeModel(10, FakeArray([0.75]), (shared, shared))
        data = NativeData(FakeArray([0, 0]), FakeArray([1.0, 2.0]),
                          {"nested_workspace": FakeArray([[5.0, 6.0]])})
        return model, data, evaluation_state.WarpEvaluationState(model, data, "cpu")

    def test_full_restore_preserves_addresses_and_removes_previous_condition_flags(self):
        model, data, saved = self.fixture()
        addresses = [array.ptr for _, array, _ in saved.saved]
        self.assertEqual(len(saved.saved), 5)  # One copy for the shared native array.
        for _, array, _ in saved.saved:
            array.values.fill(99)
        boundary = saved.restore()
        np.testing.assert_array_equal(data.overflow.values, [0, 0])
        np.testing.assert_array_equal(data.warm.values, [1.0, 2.0])
        np.testing.assert_array_equal(data.contact["nested_workspace"].values, [[5.0, 6.0]])
        np.testing.assert_array_equal(model.friction.values, [0.75])
        np.testing.assert_array_equal(model.branches[0].values, [3.0, 4.0])
        self.assertEqual(addresses, [array.ptr for _, array, _ in saved.saved])
        self.assertEqual(boundary["reset_index"], 1)
        self.assertEqual(boundary["overflow_values_at_start"], [0])
        self.assertEqual(saved.restore()["reset_index"], 2)

    def test_array_replacement_is_rejected_before_restore(self):
        _, data, saved = self.fixture()
        data.warm = FakeArray([1.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "layout"):
            saved.restore()

    def test_static_solver_change_is_rejected(self):
        model, _, saved = self.fixture()
        model.iterations = 20
        with self.assertRaisesRegex(RuntimeError, "static values"):
            saved.restore()

    def test_top_level_native_replacement_is_rejected_before_copy(self):
        for replaced in ("model", "data"):
            with self.subTest(replaced=replaced):
                model, data, saved = self.fixture()
                physics = object.__new__(warp_physics.WarpPhysics)
                physics._evaluation_state = saved
                physics.sim = SimpleNamespace(m=model, df=data)
                # Equal fields/array addresses do not make a replacement object
                # the original root retained by captured callback closures.
                if replaced == "model":
                    physics.sim.m = NativeModel(model.iterations, model.friction, model.branches)
                else:
                    physics.sim.df = NativeData(data.overflow, data.warm, data.contact)
                with self.assertRaisesRegex(RuntimeError, "root was replaced"):
                    physics.reset_evaluation_state()
                self.assertEqual(saved.reset_count, 0)


class PersistentSessionTest(unittest.TestCase):
    def session(self, persistent=True):
        session = object.__new__(evaluate.EvaluationSession)
        session._compiled = None
        session._actor_signature = None
        session._run_lock = threading.Lock()
        session.conditions_run = 0
        session.persistent = persistent
        session.initialization_s = 0.0
        session.starts, session.key, session.tables = (jnp.zeros(1),) * 3
        events, state = [], {"native": 0}

        def reset():
            events.append("reset")
            state["native"] = 0
            return {"reset_index": events.count("reset")}

        def execute(actor, *args):
            events.append("execute")
            result = jnp.asarray(actor["weight"] + state["native"])
            state["native"] = 1000
            return result

        def compile_call():
            events.append("compile")
            state["native"] = 100
            return execute

        session.fn = SimpleNamespace(lower=lambda *args: SimpleNamespace(compile=compile_call))
        session.env = SimpleNamespace(warp={1: SimpleNamespace(reset_evaluation_state=reset)})
        return session, events

    def test_a_b_a_uses_one_compile_and_resets_after_compile_before_every_rollout(self):
        session, events = self.session()
        outputs, reuse = [], []
        for value in (1.0, 2.0, 1.0):
            result, boundary, phases = session.run_actor({"weight": jnp.asarray(value)},
                                                        before_rollout=lambda: events.append("observer_reset"))
            outputs.append(float(result))
            reuse.append(phases["compiled_reused"])
            self.assertEqual(boundary["native_resets"][0]["reset_index"], len(outputs))
        self.assertEqual(outputs, [1.0, 2.0, 1.0])
        self.assertEqual(reuse, [False, True, True])
        self.assertEqual(events, ["compile", "reset", "observer_reset", "execute",
                                  "reset", "observer_reset", "execute",
                                  "reset", "observer_reset", "execute"])

    def test_reuse_requires_explicit_persistent_mode_and_fixed_actor_shapes(self):
        session, _ = self.session(False)
        session.run_actor({"weight": jnp.asarray(1.0)})
        with self.assertRaisesRegex(RuntimeError, "explicit persistent"):
            session.run_actor({"weight": jnp.asarray(1.0)})
        session, _ = self.session()
        session.run_actor({"weight": jnp.asarray(1.0)})
        with self.assertRaisesRegex(ValueError, "same actor structure"):
            session.run_actor({"weight": jnp.ones(2)})


if __name__ == "__main__":
    unittest.main()
