"""Official MuJoCo rays for the source camera against synchronized physics data.

The physics adapter owns the model and MjData batch. Before camera access it
synchronizes qpos and refreshes kinematics. This consumer reads those geometries
through mj_multiRay and mj_ray. SourcePerceptiveEnv retains ownership of the
27x48 torso camera rays, fixed far optical misses, noise and 18x32 processing.
"""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class MujocoVisibleGeom:
    geom_id: int
    body_id: int


class MuJoCoRayCamera:
    """Read the physics adapter's actual geometry, including robot self occlusion."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.model = adapter.model
        self.body_count = self.model.nbody
        self.geom_groups = np.array([1,0,1,0,0,0],dtype=np.uint8)
        self.geoms = tuple(MujocoVisibleGeom(i,int(self.model.geom_bodyid[i]))
            for i in range(self.model.ngeom)
            if self.model.geom_group[i] in (0,2) and self.model.geom_rgba[i,3] != 0)

    def raycast_numpy(self, body_positions, body_quaternions, ray_origins, ray_directions,
                      *, max_distance=1e6, min_distance=.05, max_hops=24, surface_epsilon=1e-5):
        """Cast each source image from one point using the synchronized MjData batch.

        Body poses verify the physical binding. They never replace MuJoCo geom
        transforms. Float32 state conversion permits 2e-5 m / 2e-6 quaternion
        absolute differences from the physics adapter's float64 MjData values.
        Near hits are skipped from their surface with the source absolute-range
        threshold and hop budget, preserving original model geometry IDs.
        """
        positions = np.asarray(body_positions)
        quaternions = np.asarray(body_quaternions)
        origins = np.asarray(ray_origins,dtype=np.float64)
        directions = np.asarray(ray_directions,dtype=np.float64)
        data_batch = tuple(self.adapter.data_batch)
        batch = len(data_batch)
        if (positions.shape != (batch,self.body_count,3)
                or quaternions.shape != (batch,self.body_count,4)
                or origins.ndim != 3 or origins.shape[0] != batch or origins.shape[-1] != 3
                or directions.shape != origins.shape or origins.shape[1] < 1):
            raise ValueError('Camera rays and body poses must bind the synchronized MuJoCo batch')
        if (not all(np.isfinite(value).all() for value in (positions,quaternions,origins,directions))
                or not np.allclose(np.linalg.norm(directions,axis=-1),1.,rtol=0,atol=2e-6)):
            raise ValueError('Camera requires finite poses and unit ray directions')
        if (not np.isfinite(max_distance) or not np.isfinite(min_distance)
                or not 0 <= min_distance <= max_distance or max_distance <= 0
                or not isinstance(max_hops,(int,np.integer)) or max_hops < 0
                or not np.isfinite(surface_epsilon) or surface_epsilon <= 0):
            raise ValueError('Camera range limits, hop count or surface epsilon are invalid')
        if not np.all(origins == origins[:,:1]):
            raise ValueError('Source camera rays must share one origin per image')
        ranges = np.full(origins.shape[:2],max_distance,dtype=np.float64)
        geometry_ids = np.full(origins.shape[:2],-1,dtype=np.int64)
        for world,data in enumerate(data_batch):
            if (not np.allclose(positions[world],data.xpos,rtol=0,atol=2e-5)
                    or not np.allclose(quaternions[world],data.xquat,rtol=0,atol=2e-6)):
                raise ValueError('Camera body poses differ from synchronized MuJoCo data')
            point = np.ascontiguousarray(origins[world,0])
            vectors = np.ascontiguousarray(directions[world])
            first_ids = np.empty(len(vectors),dtype=np.int32)
            first_ranges = np.empty(len(vectors),dtype=np.float64)
            mujoco.mj_multiRay(self.model,data,point,vectors.reshape(-1),self.geom_groups,
                               True,-1,first_ids,first_ranges,None,len(vectors),max_distance)
            for pixel,vector in enumerate(vectors):
                current = point.copy()
                traveled = 0.
                hit = float(first_ranges[pixel])
                geom_id = np.array([first_ids[pixel]],dtype=np.int32)
                for hop in range(max_hops+1):
                    if hit < 0 or not np.isfinite(hit) or traveled+hit > max_distance:
                        break
                    if traveled+hit >= min_distance:
                        ranges[world,pixel] = traveled+hit
                        geometry_ids[world,pixel] = geom_id[0]
                        break
                    if hop == max_hops:
                        break
                    advance = hit+surface_epsilon
                    current += vector*advance
                    traveled += advance
                    hit = mujoco.mj_ray(self.model,data,current,vector,self.geom_groups,True,-1,geom_id)
        return ranges,geometry_ids

    def raycast(self, body_positions, body_quaternions, ray_origins, ray_directions, **limits):
        """Stop-gradient Torch interface consumed by the existing source image pipeline."""
        import torch

        values = [value.detach().cpu().numpy() for value in
                  (body_positions,body_quaternions,ray_origins,ray_directions)]
        ranges,geometry_ids = self.raycast_numpy(*values,**limits)
        return (torch.as_tensor(ranges,dtype=ray_origins.dtype,device=ray_origins.device),
                torch.as_tensor(geometry_ids,dtype=torch.int64,device=ray_origins.device))
