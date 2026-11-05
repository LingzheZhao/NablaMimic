"""Frozen actor for the official Instinct supermimic release checkpoint."""
import torch
from torch import nn


class OfficialSourcePolicy(nn.Module):
    """Read the published state dict exactly, including its stored standard deviation."""

    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint['model_state_dict']
        normalization = checkpoint['policy_normalizer_state_dict']
        self.register_buffer('mean', normalization['_mean'].clone())
        self.register_buffer('std', normalization['_std'].clone())
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(18432, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(1446, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 29))
        self.actor.load_state_dict({key.removeprefix('actor.'): value
                                   for key, value in state.items() if key.startswith('actor.')}, strict=True)
        for destination, source in ((0, 'conv.conv.0'), (2, 'conv.conv.2'),
                                    (5, 'head.model.0'), (7, 'head.model.2')):
            prefix = 'encoders._parallel_blocks.depth_image.' + source
            self.encoder[destination].load_state_dict(
                {name: state[prefix + '.' + name] for name in ('weight', 'bias')}, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, history_and_depth, goal):
        raw = torch.cat((goal, history_and_depth[:, 744:], history_and_depth[:, :744]), -1)
        normalized = (raw - self.mean) / (self.std + .01)
        depth = normalized[:, 670:1246].reshape(-1, 1, 18, 32)
        return self.actor(torch.cat((normalized[:, :670], normalized[:, 1246:], self.encoder(depth)), -1))


def native_actor_from_official(checkpoint, device):
    """Initialize the current trainable native actor from the explicit release schema.

    Native normalization owns variance. It intentionally derives its standard
    deviation from that variance, while the frozen reader uses stored _std.
    """
    from .perceptive_task_inputs import source_actor_init

    actor = source_actor_init(device)
    official = OfficialSourcePolicy(checkpoint).to(device)
    for destination, source in ((0, 0), (2, 2)):
        for name in ('weight', 'bias'):
            actor['enc'][f'convolutions.{destination}.{name}'].copy_(
                getattr(official.encoder[source], name))
    for destination, source in ((0, 5), (2, 7)):
        for name in ('weight', 'bias'):
            actor['enc'][f'head.{destination}.{name}'].copy_(
                getattr(official.encoder[source], name))
    for (weight, bias), index in zip(actor['dec'], (0, 2, 4, 6)):
        weight.copy_(official.actor[index].weight.T)
        bias.copy_(official.actor[index].bias)
    normalization = checkpoint['policy_normalizer_state_dict']
    actor['source_norm']['mean'].copy_(normalization['_mean'].reshape(1990))
    actor['source_norm']['variance'].copy_(normalization['_var'].reshape(1990))
    actor['source_norm']['count'].copy_(normalization['count'])
    return actor
