"""Generated MuJoCo geometry/FK camera checks without physical stepping."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from nablamimic.mujoco_source_camera import MuJoCoRayCamera


def fixture(near_surface=False, batch=1):
    near = '<geom name="near" type="sphere" pos=".01 0 0" size=".001" group="0"/>' if near_surface else ''
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1" pos="0 0 -2" group="0"/>
      <geom name="terrain" type="sphere" size=".2" pos="3 0 0" group="0"/>
      <geom name="collision_only" type="sphere" size=".05" pos=".4 0 0" group="3"/>
      <geom name="transparent" type="sphere" size=".05" pos=".2 0 0" group="2" rgba="1 1 1 0"/>
      '''+near+'''<body name="robot" pos="1 0 0"><freejoint/>
        <geom name="robot_visual" type="sphere" size=".2" group="2"/>
      </body></worldbody></mujoco>''')
    data = tuple(mujoco.MjData(model) for _ in range(batch))
    for i,world in enumerate(data):
        world.qpos[0] += i
        mujoco.mj_kinematics(model,world)
    return SimpleNamespace(model=model,data_batch=data)


def inputs(adapter, directions):
    batch=len(adapter.data_batch)
    vectors=np.broadcast_to(np.asarray(directions,dtype=np.float64),(batch,len(directions),3)).copy()
    return (np.stack([world.xpos for world in adapter.data_batch]),
            np.stack([world.xquat for world in adapter.data_batch]),np.zeros_like(vectors),vectors)


class MuJoCoSourceCameraTest(unittest.TestCase):
    def test_self_only_official_static_flag_keeps_robot_and_original_physics(self):
        adapter=fixture();measured=MuJoCoRayCamera(adapter)
        self_only=MuJoCoRayCamera(adapter,include_static=False)
        groups=adapter.model.geom_group.copy();qpos=adapter.data_batch[0].qpos.copy()
        values=inputs(adapter,[[1,0,0],[0,0,-1]])
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('No physical steps permitted')):
            with patch.object(mujoco,'mj_multiRay',wraps=mujoco.mj_multiRay) as multi:
                distance,ids=self_only.raycast_numpy(*values)
                self.assertIs(multi.call_args.args[5],False)
        np.testing.assert_allclose(distance,[[.8,1e6]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[adapter.model.geom('robot_visual').id,-1]])
        self.assertEqual([geom.geom_id for geom in self_only.geoms],[adapter.model.geom('robot_visual').id])
        self.assertTrue(measured.include_static)
        self.assertIs(measured.adapter,self_only.adapter)
        np.testing.assert_array_equal(adapter.model.geom_group,groups)
        np.testing.assert_array_equal(adapter.data_batch[0].qpos,qpos)
        # Near-surface continuation must use the same static exclusion flag.
        near_values=list(inputs(adapter,[[1,0,0]]));near_values[2][:,:,0]=.79
        with patch.object(mujoco,'mj_ray',wraps=mujoco.mj_ray) as single:
            distance,ids=self_only.raycast_numpy(*near_values)
            self.assertIs(single.call_args.args[5],False)
        np.testing.assert_allclose(distance,[[.41]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[adapter.model.geom('robot_visual').id]])

    def test_static_flag_excludes_nonworld_bodies_welded_to_world(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="fixed" pos=".3 0 0"><geom name="fixed_visual" type="sphere" size=".05" group="2"/></body>
          <body name="robot" pos="1 0 0"><freejoint/><geom name="robot_visual" type="sphere" size=".2" group="2"/></body>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        adapter=SimpleNamespace(model=model,data_batch=(data,))
        self.assertGreater(model.body('fixed').id,0)
        self.assertEqual(model.body_weldid[model.body('fixed').id],0)
        full=MuJoCoRayCamera(adapter);self_only=MuJoCoRayCamera(adapter,include_static=False)
        full_distance,_=full.raycast_numpy(*inputs(adapter,[[1,0,0]]))
        distance,ids=self_only.raycast_numpy(*inputs(adapter,[[1,0,0]]))
        np.testing.assert_allclose(full_distance,[[.25]],rtol=0,atol=1e-12)
        np.testing.assert_allclose(distance,[[.8]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[model.geom('robot_visual').id]])

    def test_official_rays_include_static_terrain_and_robot_self_occlusion(self):
        adapter=fixture();camera=MuJoCoRayCamera(adapter)
        values=inputs(adapter,[[1,0,0],[0,1,0],[0,0,-1]])
        qpos=adapter.data_batch[0].qpos.copy()
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('No physical steps permitted')):
            with patch.object(mujoco,'mj_multiRay',wraps=mujoco.mj_multiRay) as multi:
                distance,ids=camera.raycast_numpy(*values)
                self.assertEqual(multi.call_count,1)
        np.testing.assert_allclose(distance,[[.8,1e6,2.]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[adapter.model.geom('robot_visual').id,-1,adapter.model.geom('floor').id]])
        np.testing.assert_array_equal(adapter.data_batch[0].qpos,qpos)
        self.assertEqual([geom.geom_id for geom in camera.geoms],
                         [adapter.model.geom(name).id for name in ('floor','terrain','robot_visual')])
        self.assertEqual(camera.geoms[-1].body_id,adapter.model.body('robot').id)

    def test_camera_reads_current_mujoco_data_and_preserves_original_geom_ids(self):
        adapter=fixture();camera=MuJoCoRayCamera(adapter)
        values=inputs(adapter,[[1,0,0]])
        adapter.data_batch[0].qpos[0]=2.
        mujoco.mj_kinematics(adapter.model,adapter.data_batch[0])
        with self.assertRaisesRegex(ValueError,'differ from synchronized'):
            camera.raycast_numpy(*values)
        distance,ids=camera.raycast_numpy(*inputs(adapter,[[1,0,0]]))
        np.testing.assert_allclose(distance,[[1.8]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[adapter.model.geom('robot_visual').id]])

    def test_near_surface_hops_use_absolute_original_range_and_hop_limit(self):
        adapter=fixture(near_surface=True);camera=MuJoCoRayCamera(adapter)
        values=inputs(adapter,[[1,0,0]])
        with patch.object(mujoco,'mj_ray',wraps=mujoco.mj_ray) as single:
            distance,ids=camera.raycast_numpy(*values,max_hops=2)
            self.assertEqual(single.call_count,2)
        np.testing.assert_allclose(distance,[[.8]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[adapter.model.geom('robot_visual').id]])
        for hops in (0,1):
            distance,ids=camera.raycast_numpy(*values,max_hops=hops)
            np.testing.assert_array_equal(distance,[[1e6]])
            np.testing.assert_array_equal(ids,[[-1]])

    def test_maximum_distance_and_two_real_worlds(self):
        adapter=fixture(batch=2);camera=MuJoCoRayCamera(adapter)
        values=inputs(adapter,[[1,0,0]])
        distance,ids=camera.raycast_numpy(*values,max_distance=.9)
        np.testing.assert_allclose(distance,[[.8],[.9]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ids,[[adapter.model.geom('robot_visual').id],[-1]])

    def test_contract_rejects_wrong_batch_nonunit_rays_and_distinct_camera_origins(self):
        adapter=fixture();camera=MuJoCoRayCamera(adapter)
        values=list(inputs(adapter,[[1,0,0],[0,1,0]]))
        with self.assertRaisesRegex(ValueError,'synchronized MuJoCo batch'):
            camera.raycast_numpy(values[0][:,:1],*values[1:])
        invalid=values[3].copy();invalid[0,0]*=2
        with self.assertRaisesRegex(ValueError,'unit ray'):
            camera.raycast_numpy(*values[:3],invalid)
        invalid=values[2].copy();invalid[0,0,0]=.1
        with self.assertRaisesRegex(ValueError,'share one origin'):
            camera.raycast_numpy(*values[:2],invalid,values[3])
        for limits in ({'min_distance':-.1},{'max_distance':np.inf},{'max_hops':-1},{'surface_epsilon':0}):
            with self.assertRaisesRegex(ValueError,'range limits'):
                camera.raycast_numpy(*values,**limits)


if __name__=='__main__':
    unittest.main()
