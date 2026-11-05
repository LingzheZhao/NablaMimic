"""Instinct perceptive-task tensor layouts and the deterministic camera intrinsics.

The camera's raw image is 27x48. Sensor measurements are optical-axis distances
in metres, normalized over [0,2], cropped two pixels per edge and resized 18x32.
Robot self-occlusion and the configured stochastic camera pipeline belong to the
scene renderer and remain explicit requirements of a complete camera binding.
"""
import math

import torch
from torch.nn import functional as F


SOURCE_HISTORY_FRAMES = 8
SOURCE_GOAL_DIM = 670
SOURCE_DEPTH_SHAPE = (1, 18, 32)
SOURCE_ACTOR_HIDDEN = (512, 256, 128)
SOURCE_CAMERA_POSITION = (
    .04764571478+.0039635-.0042*math.cos(math.radians(48)),
    .015, .46268178553-.044+.0042*math.sin(math.radians(48))+.016)
SOURCE_CAMERA_QUATERNION_WXYZ = (
    math.cos(math.radians(.5)/2)*math.cos(math.radians(48)/2),
    math.sin(math.radians(.5)/2), math.sin(math.radians(48)/2), 0.)


def source_policy_observation(goal, depth, history):
    """Pack source goal, image and term-major eight-frame proprioception."""
    if goal.shape[-1] != SOURCE_GOAL_DIM or history.shape[-2:] != (8, 93):
        raise ValueError('Source task requires goal670 and eight 93-wide history frames')
    if depth.shape[-3:] != SOURCE_DEPTH_SHAPE:
        raise ValueError('Source processed depth requires shape [batch,1,18,32]')
    terms = [history[..., start:end].flatten(1) for start, end in
             ((0, 3), (3, 6), (6, 35), (35, 64), (64, 93))]
    return torch.cat((goal, depth.flatten(1), *terms), -1)


def source_raw_camera_rays(device, dtype=torch.float32):
    """Unit rays in camera world-convention local axes, forward X, left Y, up Z.

    Pixel centers use the configured horizontal/vertical apertures for 87/58
    degree fields of view. Pose installation transforms are applied by caller.
    """
    height, width = 27, 48
    focal_x = width/(2*math.tan(math.radians(87)/2))
    focal_y = height/(2*math.tan(math.radians(58)/2))
    row, column = torch.meshgrid(torch.arange(height, device=device, dtype=dtype)+.5,
                                 torch.arange(width, device=device, dtype=dtype)+.5,
                                 indexing='ij')
    rays = torch.stack((torch.ones_like(row), -(column-width/2)/focal_x,
                        -(row-height/2)/focal_y), -1)
    return F.normalize(rays, dim=-1)


class SourceDepthEncoder(torch.nn.Module):
    """Conv2dHeadModel shape from the pinned Instinct policy configuration."""
    def __init__(self):
        super().__init__()
        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(32*18*32, 32),
                                        torch.nn.ReLU(), torch.nn.Linear(32, 32))

    def forward(self, depth):
        return self.head(self.convolutions(depth).flatten(1))


def source_depth_image(ray_range, local_unit_directions):
    """Convert raw ranges into the deterministic processed source image.

    Caller supplies [B,27,48] ranges after the source minimum-distance filtering.
    The image-plane depth uses each unit ray's local forward component.
    """
    optical_depth = ray_range*local_unit_directions[..., 0]
    normalized = optical_depth.clamp(0., 2.)/2.
    return F.interpolate(normalized[:, None, 2:-2, 2:-2], size=(18, 32),
                         mode='bilinear', align_corners=False)


def source_actor_init(device):
    """Create functional CNN and actor weights for the native-gradient condition."""
    encoder = SourceDepthEncoder().to(device)
    layers = []
    for input_width, output_width in zip((1446,512,256,128), (512,256,128,29)):
        layer = torch.nn.Linear(input_width, output_width, device=device)
        layers.append((layer.weight.detach().T.contiguous(), layer.bias.detach().clone()))
    return dict(enc={name: value.detach().clone() for name, value in encoder.named_parameters()},
                dec=layers, source_perceptive=torch.empty(0, device=device),
                source_norm=dict(mean=torch.zeros(1990,device=device), variance=torch.ones(1990,device=device),
                                 count=torch.zeros((),dtype=torch.int64,device=device)))


def source_actor_apply(actor, history_and_depth, goal):
    history, depth = history_and_depth[:, :744], history_and_depth[:, 744:].reshape(-1,1,18,32)
    packed = torch.cat((goal, depth.flatten(1), history), -1)
    norm = actor['source_norm']
    packed = (packed-norm['mean'])/(torch.sqrt(norm['variance'])+.01)
    goal, depth, history = packed[:,:670], packed[:,670:1246].reshape(-1,1,18,32), packed[:,1246:]
    # Concatenating physical observations can give this image slice a grad_fn.
    # The camera has no pose VJP, while CNN weights retain regression gradients.
    depth = depth.detach()
    parameters = actor['enc']
    for index in (0,2):
        depth = F.relu(F.conv2d(depth, parameters[f'convolutions.{index}.weight'],
                                parameters[f'convolutions.{index}.bias'], padding=1))
    feature = F.relu(F.linear(depth.flatten(1), parameters['head.0.weight'], parameters['head.0.bias']))
    feature = F.linear(feature, parameters['head.2.weight'], parameters['head.2.bias'])
    value = torch.cat((goal, feature, history), -1)
    for weight, bias in actor['dec'][:-1]:
        value = F.elu(value@weight+bias)
    weight, bias = actor['dec'][-1]
    return value@weight+bias


def update_source_normalizer(actor, history_and_depth, goal):
    history, depth = history_and_depth[:,:744], history_and_depth[:,744:]
    values = torch.cat((goal, depth, history), -1)
    norm = actor['source_norm']
    count = norm['count']+values.shape[0]
    rate = values.shape[0]/count.to(values.dtype)
    batch_mean = values.mean(0)
    delta = batch_mean-norm['mean']
    mean = norm['mean']+rate*delta
    variance = norm['variance']+rate*(values.var(0,unbiased=False)-norm['variance']+delta*(batch_mean-mean))
    norm['mean'].copy_(mean)
    norm['variance'].copy_(variance)
    norm['count'].copy_(count)
