"""Optional fixed reset probability for one named clip, preserving native ratios."""
import jax.numpy as jnp


def fixed_clip_mass(probabilities, clip_mask, mass):
    """Condition the existing valid-frame distribution within each partition."""
    clip_total = jnp.sum(jnp.where(clip_mask, probabilities, 0.0))
    other_total = jnp.sum(jnp.where(clip_mask, 0.0, probabilities))
    return jnp.where(clip_mask, mass * probabilities / clip_total,
                     (1.0 - mass) * probabilities / other_total)


def clip_frame_counts(frames, frame_owner, clip_count, weights=None):
    """Count named state frames, optionally selecting only applied reset proposals."""
    return jnp.bincount(frame_owner[frames.reshape(-1)],
                        weights=None if weights is None else weights.reshape(-1),
                        length=clip_count)
