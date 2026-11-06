"""Initialize a new training process from selected policy and critic parameters."""
import jax
import jax.numpy as jnp
import numpy as np


def initialize_parameters(checkpoint, actor_template, critic_template, normalizer_template):
    """Select EMA for both policy copies, retain critic/normalizer, discard training state."""
    selected = (checkpoint['actor_ema'], checkpoint['critic'], tuple(checkpoint['cnorm']))
    templates = (actor_template, critic_template, normalizer_template)
    for name, value, template in zip(('actor_ema', 'critic', 'cnorm'), selected, templates):
        leaves, structure = jax.tree_util.tree_flatten(value)
        expected, expected_structure = jax.tree_util.tree_flatten(template)
        if structure != expected_structure:
            raise ValueError(f'Initialization structure differs for {name}')
        for source, target in zip(leaves, expected):
            if np.shape(source) != np.shape(target) or np.asarray(source).dtype != np.asarray(target).dtype:
                raise ValueError(f'Initialization shape or dtype differs for {name}')
            if not np.isfinite(np.asarray(source)).all():
                raise ValueError(f'Nonfinite initialization parameter in {name}')
    return tuple(jax.tree_util.tree_map(jnp.asarray, value) for value in selected)
