"""Source-policy vision features with reconstruction-owned encoder gradients."""

import torch
from torch import nn
from torch.nn import functional as F

from .perceptive_task_inputs import SOURCE_DEPTH_SHAPE, SourceDepthEncoder
from .torch_nets import mlp_apply


class SourceWorldModelVision(nn.Module):
    """Own copies of a current functional source actor and its normalizer.

    Images supplied to ``encode`` are the processed source depth values in [0, 1].
    The actor's empirical normalization precedes the encoder. Reconstruction
    targets are clean processed images before that empirical normalization.
    """

    def __init__(self, actor):
        super().__init__()
        weight = actor['dec'][0][0]
        self.encoder = SourceDepthEncoder().to(device=weight.device, dtype=weight.dtype)
        self.encoder.load_state_dict(actor['enc'])
        self.actor_layers = nn.ModuleList([
            nn.ParameterList([nn.Parameter(value.detach().clone()) for value in layer])
            for layer in actor['dec']])
        for name in ('mean', 'variance', 'count'):
            self.register_buffer('normalizer_' + name, actor['source_norm'][name].detach().clone())
        self.register_buffer('source_perceptive', actor['source_perceptive'].detach().clone())
        # Layer specification from InstinctMJ networks.LatentDecoder at
        # 03563ee02665f327425106372264070315673748, expressed with PyTorch layers.
        # https://github.com/project-instinct/InstinctMJ/blob/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic/networks.py
        self.depth_decoder = nn.Sequential(
            nn.Linear(32, 32 * 5 * 8), nn.ELU(), nn.Unflatten(-1, (32, 5, 8)),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=(0, 1)), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 1)), nn.ELU(),
            nn.Conv2d(16, 1, 3, stride=1, padding=1), nn.Sigmoid())
        self.depth_decoder.to(device=weight.device, dtype=weight.dtype)

    def encode_depth(self, depth):
        """Encode processed [batch,1,18,32] depth with its source normalization."""
        if depth.ndim != 4 or depth.shape[1:] != SOURCE_DEPTH_SHAPE:
            raise ValueError('Source processed depth requires [batch,1,18,32]')
        mean = self.normalizer_mean[670:1246].reshape(1, *SOURCE_DEPTH_SHAPE)
        variance = self.normalizer_variance[670:1246].reshape(1, *SOURCE_DEPTH_SHAPE)
        return self.encoder((depth - mean) / (torch.sqrt(variance) + .01))

    def encode(self, history_and_depth, goal):
        """Return feature32, normalized goal670 and normalized history744."""
        if (history_and_depth.ndim != 2 or history_and_depth.shape[1] != 1320
                or goal.shape != (history_and_depth.shape[0], 670)):
            raise ValueError('Source inputs require [batch,1320] history/depth and [batch,670] goal')
        normalized_goal = (goal - self.normalizer_mean[:670]) / (torch.sqrt(self.normalizer_variance[:670]) + .01)
        normalized_history = (history_and_depth[:, :744] - self.normalizer_mean[1246:]) / (
            torch.sqrt(self.normalizer_variance[1246:]) + .01)
        depth = history_and_depth[:, 744:].reshape(-1, *SOURCE_DEPTH_SHAPE)
        return self.encode_depth(depth), normalized_goal, normalized_history

    def actor_from_features(self, feature, normalized_goal, normalized_history):
        """Train the actor head with the image feature detached."""
        value = torch.cat((normalized_goal, normalized_history, feature.detach()), -1)
        return mlp_apply(self.actor_layers, value)

    def forward(self, history_and_depth, goal):
        return self.actor_from_features(*self.encode(history_and_depth, goal))

    def reconstruct(self, feature):
        """Predict a clean processed 18-by-32 image from the source feature."""
        return self.depth_decoder(feature)

    def reconstruction_loss(self, feature, clean_depth):
        """Train encoder and decoder with source L1 against a detached clean image."""
        reconstructed = self.reconstruct(feature)
        if reconstructed.shape != clean_depth.shape:
            raise ValueError('Clean source depth must match [batch,1,18,32] reconstruction')
        return F.l1_loss(reconstructed, clean_depth.detach())

    def prediction_target(self, depth):
        """Return the detached image feature used as a latent-prediction target."""
        return self.encode_depth(depth).detach()

    def export_actor(self):
        """Copy the current functional source actor, including its normalization."""
        return dict(
            enc={name: value.detach().clone() for name, value in self.encoder.named_parameters()},
            dec=[tuple(value.detach().clone() for value in layer) for layer in self.actor_layers],
            source_perceptive=self.source_perceptive.detach().clone(),
            source_norm={name: getattr(self, 'normalizer_' + name).detach().clone()
                         for name in ('mean', 'variance', 'count')})
