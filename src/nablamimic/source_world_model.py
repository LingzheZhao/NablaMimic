"""Learned source-state increments for visual world-model feasibility studies.

State and integration conventions follow InstinctMJ supermimic/networks.py
WorldModel (03563ee), implemented independently without its runtime dependencies.
Positions translate in the current root yaw frame, rotation increments multiply
on the right, and world velocity increments rotate from the current root frame.
Predicted execution is distinct from measured simulator execution.
"""
import torch
from torch import nn


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(vector):
    """Rodrigues exponential with finite derivatives at zero rotation."""
    angle = torch.linalg.vector_norm(vector, dim=-1)
    matrix = skew(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    first = torch.sinc(angle / torch.pi)[..., None, None]
    second = (.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    return identity + first * matrix + second * (matrix @ matrix)


def yaw_rotation(rotation):
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    c, s = yaw.cos(), yaw.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(*yaw.shape, 3, 3)


def rotate(rotation, vector):
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


def invariant_features(state):
    p, rotation, velocity, angular_velocity, joints, joint_velocity = state
    inverse = rotation.transpose(-1, -2)
    # Height and root-frame gravity retain tilt while removing world XY and yaw.
    return torch.cat((p[..., 2:3], rotation[..., 2, :], rotate(inverse, velocity),
                      rotate(inverse, angular_velocity), joints, joint_velocity), -1)


def integrate_increments(state, increments):
    p, rotation, velocity, angular_velocity, joints, joint_velocity = state
    count = joints.shape[-1]
    dp, dr, dv, dw, dq, dqd = increments.split((3, 3, 3, 3, count, count), -1)
    return (p + rotate(yaw_rotation(rotation), dp), rotation @ so3_exp(dr),
            velocity + rotate(rotation, dv), angular_velocity + rotate(rotation, dw),
            joints + dq, joint_velocity + dqd)


def state_loss_components(prediction, target):
    """World position/velocity, geodesic chord rotation and joint errors per sample.

    The rotation chord is smooth at identity, avoiding a singular acos derivative.
    Readers additionally report angular error in radians when comparing forecasts.
    """
    p, rotation, velocity, angular_velocity, joints, joint_velocity = prediction
    tp, tr, tv, tw, tq, tqd = target
    return dict(position=(p-tp).square().mean(-1),
                rotation=(rotation-tr).square().mean((-1, -2)),
                velocity=(velocity-tv).square().mean(-1),
                angular_velocity=(angular_velocity-tw).square().mean(-1),
                joint_position=(joints-tq).square().mean(-1),
                joint_velocity=(joint_velocity-tqd).square().mean(-1))


class SourceWorldModel(nn.Module):
    """State/latent predictor shared by independently trained visual conditions.

    Physical normalization belongs to the training split. Visual statistics
    follow the current detached encoder with the source-style EMA update.
    Control is the actual 29-dimensional PD target, never an unbased residual.
    This module accepts physical parameters explicitly, without assigning hidden
    simulator defaults. A zero-dimensional latent supports rendered forecasts.
    """
    def __init__(self, *, scan_dim, physics_dim, latent_dim, contact_dim,
                 joint_count=29, hidden=(512, 512, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.contact_dim = contact_dim
        self.increment_dim = 12 + 2 * joint_count
        physical_input_dim = 10 + 3 * joint_count + scan_dim + physics_dim
        input_dim = physical_input_dim + latent_dim
        layers = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, self.increment_dim + contact_dim + latent_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('input_mean', torch.zeros(physical_input_dim))
        self.register_buffer('input_std', torch.ones(physical_input_dim))
        self.register_buffer('increment_scale', torch.ones(self.increment_dim))
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))
        self.register_buffer('dz_std', torch.ones(latent_dim))

    @torch.no_grad()
    def update_latent_statistics(self, z_current, valid, *, rate=.01):
        """Update source visual EMA from [batch,time+1,latent] current encodings.

        Target latent mean/sample std use the next-frame encodings. The
        one-step difference RMS uses [batch,time] validity weights, following
        InstinctMJ WorldModel.update_z_stats. Singleton targets use zero std.
        """
        if self.latent_dim == 0:
            return
        targets = z_current[:, 1:].reshape(-1, self.latent_dim)
        difference = (z_current[:, 1:] - z_current[:, :-1]).reshape(-1, self.latent_dim)
        weights = valid.reshape(-1, 1)
        correction = min(1, targets.shape[0] - 1)
        std = targets.std(0, correction=correction).clamp_min(1e-3)
        difference_rms = ((difference.square() * weights).sum(0) / weights.sum().clamp_min(1)).sqrt()
        self.z_mean.lerp_(targets.mean(0), rate)
        self.z_std.lerp_(std, rate)
        self.dz_std.lerp_(difference_rms.clamp_min(1e-3), rate)

    def forward(self, state, pd_target, scan, physics, latent):
        physical = torch.cat((invariant_features(state), pd_target, scan, physics), -1)
        features = torch.cat(((physical-self.input_mean)/self.input_std,
                              (latent-self.z_mean)/self.z_std), -1)
        output = self.net(features)
        increment, contacts, dz = output.split((self.increment_dim, self.contact_dim, self.latent_dim), -1)
        return integrate_increments(state, increment*self.increment_scale), contacts, latent+dz*self.dz_std


@torch.no_grad()
def increment_targets(state, next_state):
    """Frame-local one-control targets for training-only increment scales."""
    p, rotation, velocity, angular_velocity, joints, joint_velocity = state
    np_, nr, nv, nw, nq, nqd = next_state
    relative = rotation.transpose(-1, -2) @ nr
    sine_axis = .5 * torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                                  relative[..., 0, 2]-relative[..., 2, 0],
                                  relative[..., 1, 0]-relative[..., 0, 1]), -1)
    sine = sine_axis.norm(dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*.5).clamp(-1, 1)
    if bool(((cosine < -.99999) & (sine < 1e-5)).any()):
        raise ValueError('Training rotation increments at pi require an unambiguous axis')
    angle = torch.atan2(sine, cosine)
    factor = torch.where(sine > 1e-5, angle/sine.clamp_min(1e-8), 1+angle.square()/6)
    inverse = rotation.transpose(-1, -2)
    return torch.cat((rotate(yaw_rotation(rotation).transpose(-1, -2), np_-p),
                      sine_axis*factor[..., None], rotate(inverse, nv-velocity),
                      rotate(inverse, nw-angular_velocity), nq-joints, nqd-joint_velocity), -1)
