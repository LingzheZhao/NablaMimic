"""Physical convex-terrain ray observations in the robot's yaw frame.

Hull halfspaces use outward unit normals and n.x + offset <= 0 inside.
Terrain and robot positions share world metres. Geometry is static. The scan
is a baseline sensor with gradients through pose away from ray/face changes.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConvexTerrain:
    normals: torch.Tensor
    offsets: torch.Tensor
    valid_faces: torch.Tensor

    def ray_distance(self, origins, directions, maximum_distance):
        """First hit on the union of hull solids, with capped no-hit distances.

        origins/directions are [batch, rays, 3]. Directions must be unit vectors.
        The result is [batch, rays] in metres. Inside-solid origins return zero.
        """
        numerator = -(torch.einsum('brd,hfd->brhf', origins, self.normals) + self.offsets)
        denominator = torch.einsum('brd,hfd->brhf', directions, self.normals)
        parallel = denominator.abs() < 1.e-8
        safe = torch.where(parallel, torch.ones_like(denominator), denominator)
        crossing = numerator / safe
        near = torch.where((denominator < -1.e-8) & self.valid_faces, crossing, -torch.inf).amax(-1)
        far = torch.where((denominator > 1.e-8) & self.valid_faces, crossing, torch.inf).amin(-1)
        outside_parallel = (parallel & (numerator < 0) & self.valid_faces).any(-1)
        near = torch.clamp_min(near, 0)
        hit = (~outside_parallel) & (near <= far) & (far >= 0)
        distances = torch.where(hit, near, maximum_distance)
        return distances.amin(-1).clamp(max=maximum_distance)


def height_scan(terrain, root_position, root_quaternion_wxyz, xy_offsets, sensor_height=1.0,
                maximum_distance=3.0):
    """Downward distances from a yaw-aligned physical grid above the pelvis.

    xy_offsets is [rays,2] in metres in the robot heading frame. Sensor origin
    is pelvis height plus sensor_height. A distance cap explicitly represents
    missing ground. Reference frames and motion identities are absent.
    """
    w, x, y, z = root_quaternion_wxyz.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
    offset_x, offset_y = xy_offsets.unbind(-1)
    origins = root_position[:, None, :].expand(-1, len(xy_offsets), -1).clone()
    origins[:, :, 0] += cosine * offset_x - sine * offset_y
    origins[:, :, 1] += sine * offset_x + cosine * offset_y
    origins[:, :, 2] += sensor_height
    directions = torch.zeros_like(origins)
    directions[:, :, 2] = -1
    return terrain.ray_distance(origins, directions, maximum_distance)
