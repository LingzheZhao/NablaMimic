"""Source depth observations from predicted native physical coordinates.

The bound environment owns the physical model, camera geometry, calibration and
noise pipeline. This callable only reconstructs full-body poses with its native
kinematics and asks the existing camera for an observation. It performs no
physics steps and retains no predicted state or image cache.
"""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from nablaengine.depth_geometry import DepthGeometry

from .source_perceptive_env import SourcePerceptiveEnv
from .source_world_model import yaw_rotation


def state_to_qpos_qvel(state):
    """Convert world-frame learned state to native free-joint coordinates.

    Select the best-conditioned W/X/Y/Z quaternion expression. Clamping unused
    square-root denominators keeps both values and derivatives finite at pi.
    """
    p, rotation, velocity, angular_velocity, joints, joint_velocity = state
    m00, m01, m02 = rotation[..., 0, :].unbind(-1)
    m10, m11, m12 = rotation[..., 1, :].unbind(-1)
    m20, m21, m22 = rotation[..., 2, :].unbind(-1)
    squared = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                           1-m00+m11-m22, 1-m00-m11+m22), -1)
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21-m12, m02-m20, m10-m01), -1),
        torch.stack((m21-m12, squared[..., 1], m10+m01, m02+m20), -1),
        torch.stack((m02-m20, m10+m01, squared[..., 2], m12+m21), -1),
        torch.stack((m10-m01, m02+m20, m12+m21, squared[..., 3]), -1)), -2)
    candidates = candidates/(2*squared.clamp_min(1.e-12).sqrt()[..., :, None])
    index = squared.argmax(-1)[..., None, None].expand(*squared.shape[:-1], 1, 4)
    quaternion = F.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)
    body_angular_velocity = (rotation.transpose(-1, -2)@angular_velocity[..., None]).squeeze(-1)
    return (torch.cat((p, quaternion, joints), -1),
            torch.cat((velocity, body_angular_velocity, joint_velocity), -1))


class SourceWorldModelRender:
    """Render predicted qpos/qvel/com with the environment's clean/noisy camera.

    Inputs use native physical coordinates, including torso-frame free-root
    angular velocity. The result is detached processed depth [B,1,18,32].
    A camera with ``camera_noise=None`` produces the clean source observation.
    Otherwise each call samples the environment's normal source noise stages,
    advancing the caller's Torch RNG normally. No noise configuration is changed.
    The actor consumer separately owns detaching its encoded visual feature.
    """

    def __init__(self, env):
        if (type(env) is not SourcePerceptiveEnv or not isinstance(env.depth_geometry, DepthGeometry)
                or env.camera_state_sync is not None):
            raise ValueError('Predicted-state rendering requires the native SourcePerceptiveEnv and camera geometry')
        self.env = env

    @torch.no_grad()
    def __call__(self, qpos, qvel, com):
        env = self.env
        if (qpos.ndim != 2 or qpos.shape[1] != env.m.nq
                or qvel.shape != (qpos.shape[0], env.m.nv) or com.shape != (qpos.shape[0], 3)):
            raise ValueError('Predicted rendering requires model-bound batched qpos, qvel and COM offsets')
        if any(value.device != env.camera_rays.device or value.dtype != env.camera_rays.dtype
               for value in (qpos, qvel, com)):
            raise ValueError('Predicted coordinates must use the native camera device and float32 dtype')
        _, xpos, xquat, _, _ = env.engine_kinematics(
            qpos.detach().contiguous(), qvel.detach().contiguous(), com.detach().contiguous())
        return env.depth_image(dict(xpos=xpos, xquat=xquat)).detach()


class SourceWorldModelContext:
    """Known nominal physics and a differentiable yaw-aligned terrain scan.

    Build a static top-down height field once using native rays. This records
    the top surface and cannot represent an overhang's interior. Missing rays
    receive ground height -100m. Border lookup follows the source scanner.
    """
    scan_dim = 187
    physics_dim = 0

    def __init__(self, env, heightmap_resolution=.1, scan_max_distance=3., margin=2., z_top=10.):
        self.renderer = SourceWorldModelRender(env)
        if env.cfg.dr:
            raise ValueError('Known fixed-model context requires nominal physics with domain randomization disabled')
        if (not all(math.isfinite(value) for value in (heightmap_resolution, scan_max_distance, margin, z_top))
                or heightmap_resolution <= 0 or scan_max_distance <= 0 or margin <= 0):
            raise ValueError('Height-field resolution, scan range and margin must be finite and positive')
        self.resolution, self.scan_max_distance = heightmap_resolution, scan_max_distance
        geometry = env.depth_geometry
        static_geoms = tuple(geom for geom in geometry.geoms if env.m.body_weldid[geom.body_id] == 0)
        if not static_geoms:
            raise ValueError('Terrain conditioning requires visible static geometry')
        backend = None
        if geometry.cuda_backend is not None:
            from nablaengine.depth_cuda import CudaDepthGeometry
            backend = CudaDepthGeometry(static_geoms, geometry.body_count, env.camera_rays.device)
        static_geometry = replace(geometry, geoms=static_geoms, cuda_backend=backend)
        device, dtype = env.camera_rays.device, env.camera_rays.dtype
        with torch.no_grad():
            positions = env.ref_body_pos[..., :2].detach()
            lower = torch.floor((positions.amin((0, 1))-margin)/heightmap_resolution)*heightmap_resolution
            upper = torch.ceil((positions.amax((0, 1))+margin)/heightmap_resolution)*heightmap_resolution
            self.x0, self.y0 = lower.cpu().tolist()
            nx, ny = (torch.round((upper-lower)/heightmap_resolution).long()+1).cpu().tolist()
            com = env.ref_qpos.new_zeros((1, 3))
            _, body_positions, body_quaternions, _, _ = env.engine_kinematics(
                env.ref_qpos[:1].detach().contiguous(), env.ref_qvel[:1].detach().contiguous(), com)
            heights = torch.empty(nx*ny, device=device, dtype=dtype)
            misses = 0
            for first in range(0, nx*ny, 8192):
                indices = torch.arange(first, min(first+8192, nx*ny), device=device)
                x = self.x0+(indices % nx)*heightmap_resolution
                y = self.y0+torch.div(indices, nx, rounding_mode='floor')*heightmap_resolution
                origins = torch.stack((x, y, torch.full_like(x, z_top)), -1)[None]
                directions = torch.zeros_like(origins)
                directions[..., 2] = -1
                ranges, geom_ids = static_geometry.raycast(body_positions, body_quaternions, origins, directions,
                                                          max_distance=1.e6, min_distance=0., max_hops=0)
                valid = geom_ids[0] >= 0
                heights[first:first+len(indices)] = torch.where(valid, z_top-ranges[0], -100.)
                misses += int((~valid).sum().item())
            self.heights = heights.reshape(1, 1, ny, nx)
            x = torch.arange(-.8, .8+.05, .1, device=device, dtype=dtype)
            y = torch.arange(-.5, .5+.05, .1, device=device, dtype=dtype)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
            self.offsets = torch.stack((grid_x.flatten(), grid_y.flatten()), -1)
        self.metadata = dict(heightmap_resolution_m=heightmap_resolution,
            heightmap_bounds_m=dict(x=[self.x0, self.x0+(nx-1)*heightmap_resolution],
                                    y=[self.y0, self.y0+(ny-1)*heightmap_resolution]),
            heightmap_shape=[ny, nx], bounds_definition='All reference body XY extrema plus margin, rounded outward to grid',
            margin_m=margin, z_top_m=z_top, miss_height_m=-100., miss_fraction=misses/(nx*ny),
            static_geom_ids=[geom.geom_id for geom in static_geoms], static_selection='body_weldid == 0',
            scan_dim=self.scan_dim, scan_offsets_m=dict(x=[-.8, .8, 17], y=[-.5, .5, 11], order='y-major'),
            scan_max_distance_m=scan_max_distance, lookup='bilinear, border padding, align_corners=True',
            physics_dim=self.physics_dim, physics='Known fixed prepared model; no per-episode randomized physical parameters',
            render_noise='clean' if env.camera_noise is None else 'bound environment source noise pipeline',
            physics_steps=0)

    def __call__(self, state):
        p, rotation = state[:2]
        yaw = yaw_rotation(rotation)[..., :2, :2]
        xy = p[:, None, :2]+self.offsets@yaw.transpose(-1, -2)
        ny, nx = self.heights.shape[-2:]
        gx = (xy[..., 0]-self.x0)/(self.resolution*(nx-1))*2-1
        gy = (xy[..., 1]-self.y0)/(self.resolution*(ny-1))*2-1
        grid = torch.stack((gx, gy), -1)[None]
        height = F.grid_sample(self.heights, grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
        return (torch.clamp(p[:, 2:3]-height, max=self.scan_max_distance),
                p.new_empty((p.shape[0], 0)))

    @torch.no_grad()
    def render(self, state):
        qpos, qvel = state_to_qpos_qvel(state)
        return self.renderer(qpos, qvel, qpos.new_zeros((qpos.shape[0], 3)))
