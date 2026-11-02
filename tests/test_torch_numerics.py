"""CPU comparisons with current JAX/Optax definitions and fresh shared inputs.

Run in the pinned environment with JAX_PLATFORMS=cpu. These checks establish
functional mathematics, not CUDA-graph support or simulator equivalence.
"""
import pickle
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from nablamimic import nets as jax_nets
from nablamimic import policy_ema as jax_ema
from nablamimic import torch_nets as tn


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def assert_tree_close(left, right, rtol=2e-5, atol=2e-6):
    a, b = tn.tree_leaves(left), tn.tree_leaves(right)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(as_numpy(x), as_numpy(y), rtol=rtol, atol=atol)


def small_actor(anchor=3):
    env = SimpleNamespace(goal_dim=11, hist_dim=7, nu=3)
    return jax_nets.actor_init(jax.random.PRNGKey(17), env, hidden=(9, 5), latent=4,
                              anchor_to_decoder=anchor)


class TorchNumericsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax_nets.set_matmul_precision('highest')

    def test_jax_import_preserves_weights_and_actor_input_gradients(self):
        rng = np.random.default_rng(2)
        history = rng.normal(size=(5, 7)).astype(np.float32)
        goal = rng.normal(size=(5, 11)).astype(np.float32)
        cotangent = rng.normal(size=(5, 3)).astype(np.float32)
        for anchor in (0, 3):
            with self.subTest(anchor=anchor):
                ja = small_actor(anchor)
                ta = tn.tree_from_jax(ja)
                assert_tree_close(ja, ta, rtol=0, atol=0)
                th = torch.tensor(history, requires_grad=True)
                tg = torch.tensor(goal, requires_grad=True)
                actual = tn.actor_apply(ta, th, tg)
                expected = jax_nets.actor_apply(ja, jnp.asarray(history), jnp.asarray(goal))
                assert_tree_close(actual, expected)
                gradients = torch.autograd.grad((actual * torch.tensor(cotangent)).sum(), (th, tg))
                reference = jax.grad(lambda h, g: (jax_nets.actor_apply(ja, h, g) * cotangent).sum(),
                                     argnums=(0, 1))(jnp.asarray(history), jnp.asarray(goal))
                assert_tree_close(gradients, reference)
                if anchor:
                    split = tn.actor_decode(ta, tn.actor_encode(ta, tg[..., :-anchor]), th, tg[..., -anchor:])
                    assert_tree_close(split, actual, rtol=0, atol=0)

    def test_critic_and_running_normalization_match_population_moments(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(11, 4)).astype(np.float32)
        norm = (jnp.arange(4, dtype=jnp.float32), jnp.arange(4, dtype=jnp.float32) + 1)
        expected = jax_nets.normalize(norm, jnp.asarray(x))
        actual = tn.normalize(tn.tree_from_jax(norm), torch.tensor(x))
        assert_tree_close(actual, expected)
        mean, square = jnp.asarray(x).mean(0), (jnp.asarray(x) ** 2).mean(0)
        expected_norm = (.99 * norm[0] + .01 * mean,
                         .99 * norm[1] + .01 * jnp.maximum(square - mean ** 2, 1e-6))
        assert_tree_close(tn.update_normalizer(tn.tree_from_jax(norm), torch.tensor(x)), expected_norm)
        critic = jax_nets.mlp_init(jax.random.PRNGKey(3), [4, 6, 1])
        assert_tree_close(tn.critic_apply(tn.tree_from_jax(critic), actual),
                          jax_nets.critic_apply(critic, expected))

    def test_optional_actor_statistics_are_detached(self):
        actor = tn.tree_from_jax(small_actor())
        mean = torch.zeros(18, requires_grad=True)
        variance = torch.ones(18, requires_grad=True)
        actor['obs_norm'] = (mean, variance)
        h, g = torch.ones((2, 7), requires_grad=True), torch.ones((2, 11), requires_grad=True)
        output = tn.actor_apply(actor, h, g).sum()
        gradients = torch.autograd.grad(output, (h, g, mean, variance), allow_unused=True)
        self.assertIsNone(gradients[2])
        self.assertIsNone(gradients[3])
        self.assertIsNotNone(gradients[0])

    def test_normalization_boundary_derivative_matches_jax(self):
        values = np.array([-11., -10., -9., 0., 9., 10., 11.], np.float32)
        # The float32 sum variance + epsilon is exactly one, placing the two
        # boundary inputs exactly at the clipping thresholds in both backends.
        norm = (jnp.zeros_like(jnp.asarray(values)),
                jnp.full(values.shape, np.float32(1.0 - 1e-5)))
        tx = torch.tensor(values, requires_grad=True)
        actual = tn.normalize(tn.tree_from_jax(norm), tx)
        expected = jax_nets.normalize(norm, jnp.asarray(values))
        gradient, = torch.autograd.grad(actual.sum(), (tx,))
        reference = jax.grad(lambda x: jax_nets.normalize(norm, x).sum())(jnp.asarray(values))
        assert_tree_close(actual, expected, rtol=0, atol=0)
        assert_tree_close(gradient, reference, rtol=0, atol=0)
        np.testing.assert_array_equal(as_numpy(gradient), [0., .5, 1., 1., 1., .5, 0.])

    def test_static_nanmedian_averages_even_sets_and_preserves_empty_nan(self):
        values = np.array([[1, 3, 7, 9, 0, np.nan], [1, 3, 7, 0, 0, np.nan],
                           [0, 0, 0, 0, 0, np.nan], [1, np.inf, 0, 0, 0, np.nan]], np.float32)
        valid = values > 0
        reference = jnp.nanmedian(jnp.where(valid, values, jnp.nan), axis=1)
        actual = tn.masked_nanmedian(torch.tensor(values), torch.tensor(valid), dim=1)
        assert_tree_close(actual, reference, rtol=0, atol=0)
        self.assertEqual(float(actual[0]), 5.0)
        self.assertEqual(float(actual[1]), 3.0)
        self.assertTrue(torch.isnan(actual[2]))

    def test_action_clip_identity_vjp_and_probe_match_jax(self):
        x = np.arange(10, dtype=np.float32).reshape(5, 2)
        cotangent = np.array([[0, 0], [1, 0], [3, 0], [7, 0], [9, 0]], np.float32)
        tx, probe = torch.tensor(x, requires_grad=True), torch.zeros(2, requires_grad=True)
        output = tn.clip_rows_to_median(tx, 1.0, probe)
        self.assertTrue(torch.equal(tx, output))
        actual = torch.autograd.grad(output, (tx, probe), grad_outputs=torch.tensor(cotangent))
        _, backward = jax.vjp(lambda a, p: jax_nets.clip_rows_to_median(a, 1., p),
                             jnp.asarray(x), jnp.zeros(2))
        assert_tree_close(actual, backward(jnp.asarray(cotangent)))
        np.testing.assert_allclose(actual[0][:, 0], [0, 1, 3, 5, 5])
        np.testing.assert_allclose(actual[1], [.5, .3], atol=1e-7)

    def test_action_clip_preserves_closed_loop_state_feedback_gradient(self):
        x = np.arange(10, dtype=np.float32).reshape(5, 2) / 10

        def torch_loss(state, probe):
            for _ in range(3):
                action = torch.tanh(.3 * state)
                state = state + tn.clip_rows_to_median(action, .7, probe)
            return (state ** 2).sum()

        def jax_loss(state, probe):
            for _ in range(3):
                action = jnp.tanh(.3 * state)
                state = state + jax_nets.clip_rows_to_median(action, .7, probe)
            return (state ** 2).sum()

        actual = torch.func.grad(torch_loss, argnums=(0, 1))(torch.tensor(x), torch.zeros(2))
        expected = jax.grad(jax_loss, argnums=(0, 1))(jnp.asarray(x), jnp.zeros(2))
        assert_tree_close(actual, expected)

    def test_targets_keep_live_mask_sanitize_and_use_joint_setpoint_rms(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(8, 3, 2)).astype(np.float32)
        loss_gradient = rng.normal(size=a.shape).astype(np.float32)
        loss_gradient[0] = 0
        loss_gradient[1, 0] = [np.nan, 2.]
        loss_gradient[1, 1] = [np.inf, -np.inf]
        loss_gradient[4, 2] *= 1000
        scales = np.array([.2, .7], np.float32)
        actual = tn.action_targets(torch.tensor(a), torch.tensor(loss_gradient), torch.tensor(scales),
                                   .03, 3., outer_step=torch.tensor(250), decay_steps=16000, delta_floor=.3)
        g = -jnp.asarray(loss_gradient)
        nonfinite = jnp.any(~jnp.isfinite(g), -1)
        g = jnp.where(jnp.isfinite(g), g, 0.)
        norms = jnp.sqrt(jnp.sum(g ** 2, -1))
        live = norms > 0
        median = jnp.nanmedian(jnp.where(live, norms, jnp.nan))
        factor = jnp.minimum(1., 3. * median / (norms + 1e-30))
        clipped = live & (factor < 1.)
        g = g * factor[..., None]
        delta = .03 * (.3 + .7 * .5 * (1. + jnp.cos(jnp.pi * 250 / 16000)))
        count = jnp.maximum(live.sum(), 1)
        eta = delta / jnp.sqrt(jnp.where(live[..., None], g ** 2 * scales ** 2, 0.).sum() / (count * 2) + 1e-30)
        statistics = jnp.stack([clipped.sum()/count, jnp.where(live,1-factor,0.).sum()/count,
                                (live & (norms > 100*median)).sum()/count, nonfinite.sum().astype(jnp.float32)])
        assert_tree_close(actual['targets'], jnp.asarray(a) + eta * g)
        assert_tree_close(actual['at_clip'], statistics)
        assert_tree_close(actual['weights'], live.astype(jnp.float32), rtol=0, atol=0)
        live_t = torch.tensor(np.asarray(live))
        physical_change = (actual['targets'] - torch.tensor(a)) * torch.tensor(scales)
        rms = torch.sqrt(torch.where(live_t[..., None], physical_change ** 2, 0.).sum() / (live_t.sum()*2))
        assert_tree_close(rms, delta)
        assert_tree_close(actual['targets'][0], a[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual['targets']).all())

    def test_zero_action_gradients_do_not_create_regression_weight(self):
        a = torch.ones((8, 2, 3))
        result = tn.action_targets(a, torch.zeros_like(a), torch.ones(3), .03, 3.)
        self.assertTrue(torch.equal(result['targets'], a))
        self.assertTrue(torch.equal(result['weights'], torch.zeros((8, 2))))
        self.assertTrue(torch.equal(result['at_clip'], torch.zeros(4)))
        self.assertTrue(torch.isnan(result['at_prof'][0]).all())

    def test_td_lambda_terminal_truncation_and_pre_reset_values(self):
        reward = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32)
        term = np.array([[False, False], [True, False], [False, False], [False, True]])
        done = np.array([[False, False], [True, True], [False, False], [False, True]])
        value = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], np.float32)
        for lam in (0., .95, 1.):
            actual = tn.td_lambda(torch.tensor(reward), torch.tensor(term), torch.tensor(done),
                                  torch.tensor(value), .99, lam)
            expected = jax_nets.td_lambda(jnp.asarray(reward), jnp.asarray(term), jnp.asarray(done),
                                           jnp.asarray(value), .99, lam)
            assert_tree_close(actual, expected)
            self.assertAlmostEqual(float(actual[1, 0]), 3.)
            self.assertAlmostEqual(float(actual[1, 1]), 4 + .99 * 40, places=5)

    def test_schedules_count_optimizer_steps_and_match_optax(self):
        for per in (1, 8, 20):
            for warmup in (0, 100, 500):
                with self.subTest(per=per, warmup=warmup):
                    ref = (optax.warmup_cosine_decay_schedule(0., 6e-4, warmup*per,
                                                              16000*per, end_value=6e-5)
                           if warmup else optax.cosine_decay_schedule(6e-4, 16000*per, alpha=.1))
                    steps = np.array([0, 1, max(0,warmup*per-1), warmup*per,
                                      warmup*per+1, 8000*per, 16000*per, 18000*per], np.int32)
                    actual = tn.lr_schedule(torch.tensor(steps), 6e-4, warmup_steps=warmup,
                                            decay_steps=16000, decay_to=.1, steps_per_update=per)
                    assert_tree_close(actual, jax.vmap(ref)(jnp.asarray(steps)), rtol=1e-6, atol=1e-10)

    def test_functional_adam_matches_actual_optax_actor_and_critic_chains(self):
        rng = np.random.default_rng(8)
        initial = {'w': rng.normal(size=(4, 3)).astype(np.float32),
                   'b': rng.normal(size=(3,)).astype(np.float32)}
        for b2, element_clip, per in ((.95, .1, 8), (.999, None, 20)):
            with self.subTest(b2=b2):
                schedule = optax.warmup_cosine_decay_schedule(0., 6e-4, 2*per, 6*per, end_value=6e-5)
                optimizer = optax.chain(optax.clip_by_global_norm(1.),
                                        optax.identity() if element_clip is None else optax.clip(element_clip),
                                        optax.adam(schedule, b1=.9, b2=b2, eps=1e-8))
                jp = jax.tree.map(jnp.asarray, initial)
                js = optimizer.init(jp)
                update = jax.jit(lambda p, g, s: (lambda u, state: (optax.apply_updates(p,u),state))(
                    *optimizer.update(g,s,p)))
                tp = tn.tree_from_jax(jp)
                ts = tn.adam_init(tp)
                for step in range(45):
                    scale = (0., 1e-6, .1, 100.)[step % 4]
                    gradient = {k: (rng.normal(size=v.shape)*scale).astype(np.float32) for k,v in initial.items()}
                    lr = tn.lr_schedule(ts['count'], 6e-4, warmup_steps=2, decay_steps=6,
                                        decay_to=.1, steps_per_update=per)
                    tp, ts, _ = tn.adam_step(tp, tn.tree_from_jax(gradient), ts, lr,
                                             b2=b2, element_clip=element_clip)
                    jp, js = update(jp, jax.tree.map(jnp.asarray,gradient), js)
                    assert_tree_close(tp, jp, rtol=3e-5, atol=3e-6)
                    assert_tree_close(ts['mu'], js[2][0].mu)
                    assert_tree_close(ts['nu'], js[2][0].nu)
                    self.assertEqual(int(ts['count']), int(js[2][0].count))
                    self.assertEqual(int(ts['count']), int(js[2][1].count))

    def test_ema_serialization_continuation_and_nonweight_metadata(self):
        ja = small_actor()
        ja['obs_norm'] = (jnp.zeros(18), jnp.ones(18))
        ta = tn.tree_from_jax(ja)
        je, jn = jax_ema.init_policy_ema(ja)
        te, tn_count = tn.init_policy_ema(ta)
        for step in range(4):
            ja = jax.tree.map(lambda value: value + .1, ja)
            ta = tn.tree_from_jax(ja)
            je, jn = jax_ema.update_policy_ema(je, ja, jn)
            te, tn_count = tn.update_policy_ema(te, ta, tn_count)
            if step == 1:
                te, tn_count = pickle.loads(pickle.dumps((te, tn_count)))
            assert_tree_close(te, je)
            assert_tree_close(te['anchor_dec'], ta['anchor_dec'], rtol=0, atol=0)
            assert_tree_close(te['obs_norm'], ta['obs_norm'], rtol=0, atol=0)
            self.assertEqual(int(tn_count), int(jn))
        self.assertEqual(int(tn_count), 4)

    def test_functional_regression_and_optimizer_trace_without_graph_breaks(self):
        actor = tn.tree_from_jax(small_actor())
        trainable = tn.actor_trainable(actor)
        history = torch.arange(35, dtype=torch.float32).reshape(5,7)/20
        goal = torch.arange(55, dtype=torch.float32).reshape(5,11)/20
        target, weights, scale = torch.ones((5,3)), torch.ones(5), torch.tensor([.2,.5,.8])

        def loss(parameters, hist, goal_values):
            prediction = tn.actor_apply(tn.with_actor_parameters(actor, parameters), hist, goal_values)
            return tn.weighted_action_mse(prediction, target, weights, scale)

        def step(parameters, state, hist, goal_values):
            gradients = torch.func.grad(loss)(parameters, hist, goal_values)
            lr = tn.lr_schedule(state['count'], 6e-4, warmup_steps=100,
                                decay_steps=16000, steps_per_update=8)
            return tn.adam_step(parameters, gradients, state, lr)

        state = tn.adam_init(trainable)
        eager = step(trainable, state, history, goal)
        compiled = torch.compile(step, backend='eager', fullgraph=True)
        actual = compiled(trainable, state, history, goal)
        assert_tree_close(actual, eager, rtol=0, atol=0)
        changed = compiled(actual[0], actual[1], history + .1, goal - .2)
        expected = step(eager[0], eager[1], history + .1, goal - .2)
        assert_tree_close(changed, expected, rtol=0, atol=0)

    def test_clipping_and_targets_trace_without_host_scalar_reads(self):
        x = torch.arange(48, dtype=torch.float32).reshape(8, 3, 2) / 20
        gradient = torch.flip(x, dims=(0,)) - .4
        scale = torch.tensor([.2, .7])

        def targets(a, grad, count):
            return tn.action_targets(a, grad, scale, .03, 3., outer_step=count,
                                     decay_steps=16000, delta_floor=.3)

        compiled = torch.compile(targets, backend='eager', fullgraph=True)
        for count in (0, 500, 16000):
            step = torch.tensor(count, dtype=torch.int32)
            assert_tree_close(compiled(x, gradient, step), targets(x, gradient, step), rtol=0, atol=0)

        def objective(action, probe):
            clipped = tn.clip_rows_to_median(action, 1., probe)
            return (clipped ** 2).sum()

        derivative = torch.func.grad(objective, argnums=(0, 1))
        compiled_derivative = torch.compile(derivative, backend='eager', fullgraph=True)
        action, probe = x[0], torch.zeros(2)
        assert_tree_close(compiled_derivative(action, probe), derivative(action, probe), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
