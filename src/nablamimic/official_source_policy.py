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
