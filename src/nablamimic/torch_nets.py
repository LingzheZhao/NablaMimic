"""Functional PyTorch mathematics for the fixed NablaMimic action-target recipe.

Weights keep the donor's [input, output] layout. Runtime functions use tensor
counters and fixed-size reductions, with no host scalar reads or masked gathers
of data-dependent length. Initialization/import helpers run outside captured work.
The caller owns precision settings, random draws, optimizer commits and the one
policy-EMA update after each complete outer actor-regression group.
"""

import math

import torch
from torch.nn import functional as F


ACTOR_HIDDEN = (2048, 1024, 512, 256)
ACTOR_LATENT = 32
CRITIC_HIDDEN = (256, 128)


def tree_map(function, *trees):
    first = trees[0]
    if isinstance(first, dict):
        return {key: tree_map(function, *(tree[key] for tree in trees)) for key in sorted(first)}
    if isinstance(first, list):
        return [tree_map(function, *(tree[i] for tree in trees)) for i in range(len(first))]
    if isinstance(first, tuple):
        return tuple(tree_map(function, *(tree[i] for tree in trees)) for i in range(len(first)))
    return function(*trees)


def tree_leaves(tree):
    if isinstance(tree, dict):
        return tuple(leaf for key in sorted(tree) for leaf in tree_leaves(tree[key]))
    if isinstance(tree, (list, tuple)):
        return tuple(leaf for child in tree for leaf in tree_leaves(child))
    return (tree,)


def tree_from_jax(tree, device='cpu', requires_grad=False):
    """Copy explicitly supplied current JAX arrays; never generate different weights."""
    import numpy as np

    def convert(value):
        tensor = torch.from_numpy(np.array(value, copy=True)).to(device=device)
        return tensor.requires_grad_(requires_grad and tensor.is_floating_point())

    return tree_map(convert, tree)


def actor_trainable(actor):
    return {'enc': actor['enc'], 'dec': actor['dec']}


def with_actor_parameters(actor, trainable):
    return {**actor, 'enc': trainable['enc'], 'dec': trainable['dec']}


def copy_tree_(destination, source):
    """Commit functional outputs to persistent storage, including inside capture."""
    with torch.no_grad():
        tree_map(lambda dst, src: dst.copy_(src), destination, source)
    return destination


def mlp_apply(parameters, x):
    for weight, bias in parameters[:-1]:
        x = F.elu(x @ weight + bias)
    weight, bias = parameters[-1]
    return x @ weight + bias


critic_apply = mlp_apply


def normalize(norm, x):
    mean, variance = norm
    normalized = (x - mean) / torch.sqrt(variance + 1e-5)
    # minimum/maximum split equal-input cotangents, as JAX clip does. Torch
    # clamp instead passes the full cotangent at either clipping boundary.
    lower, upper = normalized.new_full((), -10.0), normalized.new_full((), 10.0)
    return torch.minimum(torch.maximum(normalized, lower), upper)


def update_normalizer(norm, x, decay=0.99):
    """Population moment update, matching the donor's mean-of-squares formula."""
    old_mean, old_variance = norm
    mean = x.mean(dim=0)
    square_mean = (x ** 2).mean(dim=0)
    variance = torch.clamp_min(square_mean - mean ** 2, 1e-6)
    return (decay * old_mean + (1.0 - decay) * mean,
            decay * old_variance + (1.0 - decay) * variance)


def actor_encode(actor, goal_ref):
    return torch.tanh(mlp_apply(actor['enc'], goal_ref))


def actor_decode(actor, latent, history, anchor):
    return mlp_apply(actor['dec'], torch.cat((latent, history, anchor), dim=-1))


def actor_apply(actor, history, goal):
    if 'source_perceptive' in actor:
        from .perceptive_task_inputs import source_actor_apply
        return source_actor_apply(actor, history, goal)
    if 'obs_norm' in actor:
        norm = tuple(value.detach() for value in actor['obs_norm'])
        normalized = normalize(norm, torch.cat((history, goal), dim=-1))
        history, goal = normalized[..., :history.shape[-1]], normalized[..., history.shape[-1]:]
    if 'anchor_dec' in actor:
        anchor_size = actor['anchor_dec'].shape[-1]
        latent = actor_encode(actor, goal[..., :-anchor_size])
        return actor_decode(actor, latent, history, goal[..., -anchor_size:])
    latent = actor_encode(actor, goal)
    return mlp_apply(actor['dec'], torch.cat((latent, history), dim=-1))


def discounted_return_bound(maximum_reward, gamma, maximum_controls):
    """The native trainer's finite-horizon positive-return magnitude for --v-bound."""
    if not math.isfinite(maximum_reward) or maximum_reward < 0 or not 0 <= gamma < 1 or maximum_controls < 1:
        raise ValueError('Value bound requires finite nonnegative reward, gamma in [0,1) and positive horizon')
    return maximum_reward*(1-gamma**maximum_controls)/(1-gamma)


def bootstrap_value(value, bound):
    """Symmetric native algorithm clip. The lower bound is an algorithm choice."""
    if bound is None:
        return value
    lower,upper=value.new_full((),-bound),value.new_full((),bound)
    return torch.minimum(torch.maximum(value,lower),upper)


def td_lambda(reward, terminated, done, next_value, gamma, lam):
    """[H,N] lambda returns; next_value belongs to the pre-reset next state."""
    following = next_value[-1]
    reversed_returns = []
    for index in range(reward.shape[0] - 1, -1, -1):
        bootstrap = (1.0 - lam) * next_value[index] + lam * torch.where(
            done[index], next_value[index], following)
        following = reward[index] + gamma * (1.0 - terminated[index].to(reward.dtype)) * bootstrap
        reversed_returns.append(following)
    return torch.stack(reversed_returns[::-1], dim=0)


def masked_nanmedian(values, valid, dim=-1):
    """JAX/NumPy median, including the average of the two middle valid values.

    Sorting uses the original fixed shape. Invalid/NaN entries sort to +infinity;
    integer tensor counts determine two scalar gather indices. Valid infinities
    retain nanmedian semantics. An empty set returns NaN, as in the donor.
    """
    valid = valid & ~torch.isnan(values)
    ordered = torch.sort(torch.where(valid, values, float('inf')), dim=dim).values
    count = valid.sum(dim=dim)
    safe_count = torch.clamp_min(count, 1)
    lower = torch.div(safe_count - 1, 2, rounding_mode='floor')
    upper = torch.div(safe_count, 2, rounding_mode='floor')
    a = torch.gather(ordered, dim, lower.unsqueeze(dim)).squeeze(dim)
    b = torch.gather(ordered, dim, upper.unsqueeze(dim)).squeeze(dim)
    median = (a + b) * 0.5
    return torch.where(count > 0, median, float('nan'))


def clip_cotangent(gradient, kappa):
    """Per-row clipped cotangent and the donor's two backward probe readings."""
    norm = torch.sqrt(torch.sum(gradient ** 2, dim=-1))
    live = norm > 0
    threshold = kappa * masked_nanmedian(norm, live)
    threshold = torch.where(torch.isfinite(threshold), threshold, float('inf'))
    scale = torch.minimum(torch.ones_like(norm), threshold / (norm + 1e-30))
    live_count = torch.clamp_min(live.sum(), 1)
    fraction = (live & (scale < 1.0)).sum().to(gradient.dtype) / live_count
    removed = 1.0 - (norm * scale).sum() / (norm.sum() + 1e-30)
    return gradient * scale.unsqueeze(-1), torch.stack((fraction, removed))


class _ActionCotangentClip(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(action, kappa, probe):
        return action.clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, kappa, _ = inputs
        ctx.tensor_kappa = isinstance(kappa, torch.Tensor)
        if ctx.tensor_kappa:
            ctx.save_for_backward(kappa)
        else:
            ctx.kappa = kappa

    @staticmethod
    def backward(ctx, gradient):
        kappa = ctx.saved_tensors[0] if ctx.tensor_kappa else ctx.kappa
        clipped, statistics = clip_cotangent(gradient, kappa)
        return clipped, None, statistics


# Keep the custom VJP intact through Dynamo's torch.func tracing. Without this
# boundary, Torch 2.9 can inline only forward and silently lose backward
# (pytorch/pytorch#193279). AOTAutograd/Inductor still trace the whole operation;
# every tensor is an explicit argument, with no captured tensors or mutation.
@torch.compiler.allow_in_graph
def clip_rows_to_median(action, kappa, probe=None):
    """Identity in the forward; clip the action cotangent only during backward."""
    if probe is None:
        probe = action.new_zeros((2,))
    return _ActionCotangentClip.apply(action, kappa, probe)


def action_delta(outer_step, delta, decay_steps=0, delta_floor=1.0):
    if decay_steps <= 0 or delta_floor >= 1.0:
        return torch.ones_like(outer_step, dtype=torch.float32) * delta
    fraction = torch.clamp_max(outer_step.to(torch.float32) / decay_steps, 1.0)
    return delta * (delta_floor + (1.0 - delta_floor) * 0.5 * (1.0 + torch.cos(math.pi * fraction)))


def action_targets(a_pi, loss_action_grad, action_scale, delta, kappa=None, *,
                   outer_step=None, decay_steps=0, delta_floor=1.0):
    """Targets from d(-J)/da, with the same sanitization, clipping and RMS units.

    Inputs and outputs have [H,N,nu] action shape. ``a_pi`` is the noiseless live
    actor output, not the noisy action used by physics. The caller stores the
    returned targets/weights detached from the original rollout.
    """
    if outer_step is None:
        outer_step = torch.zeros((), device=a_pi.device, dtype=torch.int32)
    delta = action_delta(outer_step, delta, decay_steps, delta_floor)
    gradient = -loss_action_grad
    nonfinite = (~torch.isfinite(gradient)).any(dim=-1)
    gradient = torch.where(torch.isfinite(gradient), gradient, 0.0)
    norm = torch.sqrt((gradient ** 2).sum(dim=-1))
    live = norm > 0
    median = masked_nanmedian(norm.reshape(-1), live.reshape(-1))
    scale = torch.ones_like(norm)
    if kappa is not None:
        threshold = kappa * median
        threshold = torch.where(torch.isfinite(threshold), threshold, float('inf'))
        scale = torch.minimum(scale, threshold / (norm + 1e-30))
        gradient = gradient * scale.unsqueeze(-1)
    clipped = live & (scale < 1.0)
    live_count = torch.clamp_min(live.sum(), 1)
    statistics = torch.stack((
        clipped.sum().to(gradient.dtype) / live_count,
        torch.where(live, 1.0 - scale, 0.0).sum() / live_count,
        (live & (norm > 100.0 * median)).sum().to(gradient.dtype) / live_count,
        nonfinite.sum().to(gradient.dtype)))
    squared = torch.where(live.unsqueeze(-1), gradient ** 2 * action_scale ** 2, 0.0).sum()
    count = live_count.to(gradient.dtype) * a_pi.shape[-1]
    eta = delta / torch.sqrt(squared / count + 1e-30)
    result = dict(targets=a_pi + eta * gradient, weights=live.to(gradient.dtype), eta=eta,
                  step_delta=delta, gradients=gradient, scales=scale, at_clip=statistics)
    if a_pi.shape[0] % 8 == 0:
        live_blocks = live.reshape(8, -1)
        medians = masked_nanmedian(norm.reshape(8, -1), live_blocks, dim=1)
        result['at_prof'] = torch.stack((
            torch.where(torch.isfinite(medians), medians, 0.0) / (median + 1e-30),
            clipped.reshape(8, -1).sum(dim=1).to(gradient.dtype) /
            torch.clamp_min(live_blocks.sum(dim=1), 1)))
    return result


def weighted_action_mse(prediction, target, weights, action_scale):
    error = (prediction - target) * action_scale
    return (weights * (error ** 2).mean(dim=-1)).sum() / torch.clamp_min(weights.sum(), 1.0)


def weighted_mse(prediction, target, weights):
    return (weights * (prediction - target) ** 2).sum() / torch.clamp_min(weights.sum(), 1.0)


def lr_schedule(optimizer_count, learning_rate, *, warmup_steps=0, decay_steps=0,
                decay_to=0.1, steps_per_update=1):
    """Optax warmup+cosine evaluated at the pre-update optimizer counter."""
    count = optimizer_count.to(torch.float32)
    if decay_steps <= 0:
        return torch.ones_like(count) * learning_rate
    warmup = warmup_steps * steps_per_update
    decay = decay_steps * steps_per_update
    if warmup < 0 or warmup >= decay:
        raise ValueError('Warmup must be nonnegative and smaller than the cosine horizon')
    cosine_count = torch.clamp_max(count - warmup, decay - warmup)
    cosine = 0.5 * (1.0 + torch.cos(math.pi * cosine_count / (decay - warmup)))
    decayed = learning_rate * ((1.0 - decay_to) * cosine + decay_to)
    if warmup == 0:
        return decayed
    # Optax linear_schedule is polynomial_schedule(power=1), in this order.
    warm = -learning_rate * (1.0 - torch.clamp(count, 0, warmup) / warmup) + learning_rate
    return torch.where(count < warmup, warm, decayed)


def adam_init(parameters):
    leaves = tree_leaves(parameters)
    if not leaves:
        raise ValueError('Adam requires at least one parameter')
    return dict(count=torch.zeros((), dtype=torch.int32, device=leaves[0].device),
                mu=tree_map(torch.zeros_like, parameters), nu=tree_map(torch.zeros_like, parameters))


def adam_step(parameters, gradients, state, learning_rate, *, b1=0.9, b2=0.95,
              eps=1e-8, max_grad_norm=1.0, element_clip=0.1):
    """Optax chain: global-norm clip, optional element clip, Adam, negative LR.

    Returns functional parameters/state and the pre-clip global norm. Actor uses
    b2=.95 and element_clip=.1; critic uses b2=.999 and element_clip=None.
    The caller commits outputs in place to its persistent captured storage.
    """
    leaves = tree_leaves(gradients)
    norm = torch.sqrt(torch.stack([(gradient ** 2).sum() for gradient in leaves]).sum())
    if max_grad_norm is not None:
        gradients = tree_map(lambda gradient: torch.where(
            norm < max_grad_norm, gradient, (gradient / norm.to(gradient.dtype)) * max_grad_norm), gradients)
    if element_clip is not None:
        gradients = tree_map(lambda gradient: torch.clamp(gradient, -element_clip, element_clip), gradients)
    mu = tree_map(lambda gradient, old: (1.0 - b1) * gradient + b1 * old, gradients, state['mu'])
    nu = tree_map(lambda gradient, old: (1.0 - b2) * (gradient ** 2) + b2 * old, gradients, state['nu'])
    # Avoid integer wraparound at the terminal representable Optax counter.
    count = torch.clamp_max(state['count'].to(torch.int64) + 1, 2**31 - 1).to(torch.int32)
    exponent = count.to(torch.float32)
    correction1 = 1.0 - b1 ** exponent
    correction2 = 1.0 - b2 ** exponent
    direction = tree_map(lambda m, v: (m / correction1.to(m.dtype)) /
                         (torch.sqrt(v / correction2.to(v.dtype)) + eps), mu, nu)
    updated = tree_map(lambda parameter, update: parameter + (-learning_rate) * update,
                       parameters, direction)
    return updated, dict(count=count, mu=mu, nu=nu), norm


def init_policy_ema(actor):
    _validate_ema_actor(actor)
    average = tree_map(lambda value: value.detach().clone(), actor)
    device = tree_leaves(actor_trainable(actor))[0].device
    return average, torch.zeros((), device=device, dtype=torch.int32)


def update_policy_ema(average, actor, updates):
    """One completed outer update; n counts prior updates, so first beta=.1."""
    _validate_ema_actor(average)
    _validate_ema_actor(actor)
    n = updates.to(torch.float32)
    beta = torch.clamp_max((1.0 + n) / (10.0 + n), 0.999)
    weights = tree_map(lambda old, live: beta.to(live.dtype) * old + (1.0 - beta.to(live.dtype)) * live,
                       actor_trainable(average), actor_trainable(actor))
    return with_actor_parameters(actor, weights), updates + 1


def _validate_ema_actor(actor):
    # Plain membership keeps this static schema check traceable by Dynamo;
    # set subset/difference operations are unsupported in the pinned compiler.
    allowed = ('enc', 'dec', 'anchor_dec', 'obs_norm', 'source_perceptive', 'source_norm')
    if (not isinstance(actor, dict) or 'enc' not in actor or 'dec' not in actor or
            any(name not in allowed for name in actor)):
        raise ValueError('Policy EMA requires enc/dec and only optional anchor_dec/obs_norm')
    for name in ('enc', 'dec'):
        leaves = tree_leaves(actor[name])
        if not leaves or any(not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point
                             for value in leaves):
            raise ValueError('Policy EMA requires nonempty floating-point encoder/decoder parameters')
