"""Whole encoder-decoder policy averaging, once per outer actor update.

The counter is the number of *previously completed* EMA updates: the first update
uses n=0 and beta=0.1. Only encoder/decoder trainable parameters are averaged.
Structure markers and observation statistics come from the current live actor.
The functions accept the replicated actor trees used by the training loop too.
"""

import jax
import jax.numpy as jnp


def _validate_actor(actor):
    required, optional = {"enc", "dec"}, {"anchor_dec", "obs_norm"}
    if not isinstance(actor, dict) or not required <= actor.keys() or actor.keys() - required - optional:
        raise ValueError("Policy EMA requires enc/dec and only optional anchor_dec/obs_norm fields")
    for name in required:
        leaves = jax.tree_util.tree_leaves(actor[name])
        if not leaves or any(not hasattr(x, "dtype") or not jnp.issubdtype(x.dtype, jnp.floating) for x in leaves):
            raise ValueError(f"Policy EMA requires nonempty floating-point {name} parameters")


def init_policy_ema(actor):
    """Copy the initial live actor and return its zero completed-update counter."""
    _validate_actor(actor)
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), actor), jnp.int32(0)


def update_policy_ema(ema, actor, updates):
    """Average one completed outer update, returning the policy and next count.

    Call after all regression minibatches, never once per minibatch. ``updates``
    must be a nonnegative scalar integer; persisted counts are checked by the
    trainer against completed outer updates before every save.
    """
    _validate_actor(actor)
    _validate_actor(ema)
    if jax.tree_util.tree_structure(ema) != jax.tree_util.tree_structure(actor):
        raise ValueError("EMA and live actor structures differ")
    if isinstance(updates, int) and updates < 0:
        raise ValueError("EMA update count must be nonnegative")
    count = jnp.asarray(updates)
    if count.shape != () or not jnp.issubdtype(count.dtype, jnp.integer):
        raise ValueError("EMA update count must be a scalar integer")
    n = count.astype(jnp.float32)
    beta = jnp.minimum(jnp.float32(0.999), (1.0 + n) / (10.0 + n))

    def average(old, live):
        if old.shape != live.shape or old.dtype != live.dtype:
            raise ValueError("EMA and live parameter shapes/dtypes differ")
        b = beta.astype(live.dtype)
        return b * old + (1.0 - b) * live

    return {
        **actor,
        **{name: jax.tree_util.tree_map(average, ema[name], actor[name]) for name in ("enc", "dec")},
    }, count + jnp.asarray(1, dtype=count.dtype)
