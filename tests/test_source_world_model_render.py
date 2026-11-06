"""Generated CPU FK/ray checks for predicted source-camera observations."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from nablaengine.depth_geometry import DepthGeometry
from nablamimic.perceptive_task_inputs import SOURCE_CAMERA_POSITION, SOURCE_CAMERA_QUATERNION_WXYZ, source_raw_camera_rays
from nablamimic.source_depth_noise import SourceDepthNoise
from nablamimic.source_perceptive_env import SourcePerceptiveEnv
from nablamimic.source_world_model import so3_exp
from nablamimic.source_world_model_render import SourceWorldModelContext, SourceWorldModelRender, state_to_qpos_qvel
from nablamimic.torch_env import quat_to_matrix


def camera_fixture(noise=False):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="4 4 .1" group="0"/>
      <geom type="box" pos="2 0 1" size=".1 2 1" group="0"/>
      <body name="torso_link" pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1" group="3"/>
        <body name="arm" pos=".3 0 .1"><joint axis="0 0 1"/>
          <geom type="sphere" pos=".25 0 0" size=".12" group="2"/>
        </body>
      </body>
    </worldbody></mujoco>''')
    env = SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
    env.m, env.device = model, 'cpu'
    env.cfg = SimpleNamespace(dr=False, source_camera_input="measured", source_depth_noise=noise, source_depth_gradient=True)
    env.torso_id = model.body('torso_link').id
    env.depth_geometry = DepthGeometry.from_model(model, device='cpu', backend='torch')
    env.camera_state_sync = None
    env.camera_rays = source_raw_camera_rays('cpu')
    env.camera_position = torch.tensor(SOURCE_CAMERA_POSITION)
    env.camera_rotation = torch.nn.functional.normalize(torch.tensor(SOURCE_CAMERA_QUATERNION_WXYZ), dim=-1)
    env.camera_noise = SourceDepthNoise('cpu') if noise else None
    calls = []

    def kinematics(qpos, qvel, com):
        calls.append(dict(qpos=qpos.clone(), qvel=qvel.clone(), com=com.clone(),
                          grad_enabled=torch.is_grad_enabled()))
        position, quaternion = [], []
        for q, v in zip(qpos, qvel):
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:] = q.numpy(), v.numpy()
            mujoco.mj_kinematics(model, data)
            position.append(data.xpos.copy())
            quaternion.append(data.xquat.copy())
        xpos = torch.tensor(np.asarray(position), dtype=qpos.dtype)
        xquat = torch.tensor(np.asarray(quaternion), dtype=qpos.dtype)
        return qpos, xpos, xquat, qpos.new_zeros((len(qpos), model.nbody, 6)), torch.zeros_like(xpos)

    env.engine_kinematics = kinematics

    def forbidden_step(*args, **kwargs):
        raise AssertionError('Predicted-state rendering must never step physics')

    env.engine_step = forbidden_step
    qpos = torch.tensor(model.qpos0.copy(), dtype=torch.float32)[None]
    qvel = torch.zeros(1, model.nv)
    com = torch.zeros(1, 3)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    env.ref_qpos, env.ref_qvel = qpos, qvel
    env.ref_body_pos = torch.tensor(data.xpos[None].copy(), dtype=torch.float32)
    return env, qpos, qvel, com, calls


class SourceWorldModelRenderTest(unittest.TestCase):
    def test_rotation_conversion_is_finite_at_identity_and_pi(self):
        vectors = torch.tensor([[0., 0., 0.], [np.pi, 0., 0.], [0., np.pi, 0.],
                                [0., 0., np.pi], [.3, -.5, .7]], dtype=torch.float64, requires_grad=True)
        rotation = so3_exp(vectors)
        p = torch.zeros(5, 3, dtype=torch.float64)
        velocity = p+torch.tensor([1., 2., 3.])
        angular = p+torch.tensor([.2, -.4, .7])
        joints = torch.zeros(5, 2, dtype=torch.float64)
        qpos, qvel = state_to_qpos_qvel((p, rotation, velocity, angular, joints, joints))
        torch.testing.assert_close(quat_to_matrix(qpos[:, 3:7]), rotation, rtol=0, atol=1.e-12)
        torch.testing.assert_close(qvel[:, :3], velocity)
        torch.testing.assert_close((rotation@qvel[:, 3:6, None]).squeeze(-1), angular)
        gradient = torch.autograd.grad(qpos.square().sum()+qvel.square().sum(), vectors)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_context_static_field_scan_order_and_gradients(self):
        env, qpos, qvel, com, calls = camera_fixture()
        context = SourceWorldModelContext(env, heightmap_resolution=.2)
        self.assertEqual(context.metadata['static_geom_ids'], [0, 1])
        self.assertEqual(context.metadata['miss_fraction'], 0.)
        ny, nx = context.heights.shape[-2:]
        x = context.x0+torch.arange(nx)*context.resolution
        y = context.y0+torch.arange(ny)*context.resolution
        # A manufactured plane isolates the bilinear lookup and its derivatives.
        context.heights = (.2*x[None]+.1*y[:, None]+.3)[None, None]
        p = torch.tensor([[.2, .1, 1.]], requires_grad=True)
        angle = torch.tensor(.3, requires_grad=True)
        vector = torch.stack((angle*0, angle*0, angle))[None]
        rotation = so3_exp(vector)
        state = (p, rotation, torch.zeros(1, 3), torch.zeros(1, 3), qpos[:, 7:], qvel[:, 6:])
        scan, physics = context(state)
        self.assertEqual(scan.shape, (1, 187))
        self.assertEqual(physics.shape, (1, 0))
        torch.testing.assert_close(context.offsets[0], torch.tensor([-.8, -.5]))
        torch.testing.assert_close(context.offsets[16], torch.tensor([.8, -.5]))
        torch.testing.assert_close(context.offsets[17], torch.tensor([-.8, -.4]))
        first_x = p[0, 0]-.8*angle.cos()+.5*angle.sin()
        first_y = p[0, 1]-.8*angle.sin()-.5*angle.cos()
        torch.testing.assert_close(scan[0, 0], p[0, 2]-(.2*first_x+.1*first_y+.3))
        p_gradient, yaw_gradient = torch.autograd.grad(scan[0, 0], (p, angle))
        torch.testing.assert_close(p_gradient, torch.tensor([[-.2, -.1, 1.]]))
        self.assertGreater(float(yaw_gradient.abs()), .01)
        far = (torch.tensor([[1.e4, 1.e4, 100.]]), *state[1:])
        torch.testing.assert_close(context(far)[0], torch.full((1, 187), 3.))
        image = context.render(state)
        self.assertEqual(image.shape, (1, 1, 18, 32))
        self.assertFalse(image.requires_grad)
        torch.testing.assert_close(calls[-1]['com'], torch.zeros(1, 3))

    def test_context_miss_height_and_nominal_physics_are_explicit(self):
        env, qpos, qvel, com, calls = camera_fixture()
        env.cfg.dr = True
        with self.assertRaises(ValueError):
            SourceWorldModelContext(env)
        env.cfg.dr = False
        env.depth_geometry = replace(env.depth_geometry,
            geoms=tuple(geom for geom in env.depth_geometry.geoms if geom.geom_id != 0))
        context = SourceWorldModelContext(env, heightmap_resolution=.2, margin=.2)
        self.assertEqual(context.metadata['miss_fraction'], 1.)
        torch.testing.assert_close(context.heights, torch.full_like(context.heights, -100.))

    def test_clean_camera_matches_existing_pipeline_and_preserves_inputs(self):
        env, qpos, qvel, com, calls = camera_fixture()
        inputs = [value.clone() for value in (qpos, qvel, com)]
        _, xpos, xquat, _, _ = env.engine_kinematics(qpos, qvel, com)
        with torch.no_grad():
            expected = env.depth_image(dict(xpos=xpos, xquat=xquat))
        image = SourceWorldModelRender(env)(qpos, qvel, com)
        self.assertEqual(image.shape, (1, 1, 18, 32))
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(image).all())
        for value, saved in zip((qpos, qvel, com), inputs):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        self.assertFalse(calls[-1]['grad_enabled'])

    def test_articulated_robot_occlusion_uses_predicted_full_body_kinematics(self):
        env, qpos, qvel, com, calls = camera_fixture()
        render = SourceWorldModelRender(env)
        first = render(qpos, qvel, com)
        moved = qpos.clone()
        moved[:, -1] = 1.3
        second = render(moved, qvel, com)
        torch.testing.assert_close(calls[-1]['qpos'], moved)
        self.assertGreater(float((first-second).abs().max()), .01)

    def test_noised_camera_matches_source_rng_progression(self):
        env, qpos, qvel, com, calls = camera_fixture(noise=True)
        _, xpos, xquat, _, _ = env.engine_kinematics(qpos, qvel, com)
        torch.manual_seed(1729)
        before = torch.random.get_rng_state()
        with torch.no_grad():
            expected = env.depth_image(dict(xpos=xpos, xquat=xquat))
        expected_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(before)
        image = SourceWorldModelRender(env)(qpos, qvel, com)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        self.assertFalse(torch.equal(before, expected_rng))

    def test_predicted_coordinates_and_depth_stop_gradients(self):
        env, qpos, qvel, com, calls = camera_fixture()
        for value in (qpos, qvel, com):
            value.requires_grad_()
        image = SourceWorldModelRender(env)(qpos, qvel, com)
        self.assertFalse(image.requires_grad)
        self.assertIsNone(image.grad_fn)
        for name in ('qpos', 'qvel', 'com'):
            self.assertFalse(calls[-1][name].requires_grad)

    def test_camera_owner_and_coordinate_shapes_are_explicit(self):
        env, qpos, qvel, com, calls = camera_fixture()
        render = SourceWorldModelRender(env)
        with self.assertRaises(ValueError):
            render(qpos[:, :-1], qvel, com)
        with self.assertRaises(ValueError):
            render(qpos.double(), qvel.double(), com.double())
        env.camera_state_sync = lambda state: None
        with self.assertRaises(ValueError):
            SourceWorldModelRender(env)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
