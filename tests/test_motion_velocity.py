"""Reference body velocities are measured at xpos, including offset inertias."""

import mujoco
import numpy as np

from nablamimic import motion


def test_reference_velocity_at_body_origin(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody><body name="base"><freejoint/>
      <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
      <body name="rotor" pos="1 0 0"><joint axis="0 0 1"/>
        <inertial mass="1" pos=".4 0 0" diaginertia=".1 .1 .1"/>
        <body name="tip" pos="1 0 0">
          <inertial mass="1" pos=".2 0 0" diaginertia=".1 .1 .1"/>
        </body>
      </body>
    </body></worldbody></mujoco>''')
    monkeypatch.setattr(motion, '_foot_min_z', lambda model, data: 0.)
    times = np.arange(10) / 100
    raw = np.zeros((len(times), 8))
    raw[:, 6] = 1  # source quaternion XYZW
    raw[:, 7] = .4 + 2*times
    bodies = ('rotor', 'tip')
    qpos, qvel, _, _, linear, angular = motion.process_clip(
        model, raw, fps_out=100, fps_in=100, bodies=bodies)
    data = mujoco.MjData(model)
    plus, minus = mujoco.MjData(model), mujoco.MjData(model)
    ids = [model.body(name).id for name in bodies]
    epsilon = 1e-6
    for i, q in enumerate(qpos):
        plus.qpos[:], minus.qpos[:] = q, q
        mujoco.mj_integratePos(model, plus.qpos, qvel[i], epsilon)
        mujoco.mj_integratePos(model, minus.qpos, qvel[i], -epsilon)
        mujoco.mj_kinematics(model, plus)
        mujoco.mj_kinematics(model, minus)
        derivative = (plus.xpos[ids] - minus.xpos[ids]) / (2*epsilon)
        np.testing.assert_allclose(linear[i], derivative, atol=2e-10)
        data.qpos[:], data.qvel[:] = q, qvel[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        relative = data.xpos[ids] - data.subtree_com[model.body_rootid[ids]]
        # Same world-origin conversion as TorchTrackingEnv.body_velocity.
        from_cvel = data.cvel[ids, 3:] - np.cross(relative, data.cvel[ids, :3])
        np.testing.assert_allclose(linear[i], from_cvel, atol=1e-14)
        np.testing.assert_allclose(angular[i], data.cvel[ids, :3], atol=1e-14)
    np.testing.assert_allclose(linear[:, 0], 0, atol=1e-14)
    assert np.linalg.norm(linear[:, 1], axis=-1).min() > 1.9
