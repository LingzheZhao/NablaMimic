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
