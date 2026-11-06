"""Torque-limit layout compatibility without time integration or GPU simulation."""
import json
import os
from pathlib import Path
import unittest

import mujoco
import numpy as np

from nablamimic.robot import normalize_effort_limits


XML = '''<mujoco><worldbody><body><joint name="hinge" type="hinge"/><geom type="sphere" size=".1" mass="1"/></body></worldbody>
<actuator><position name="motor" joint="hinge" kp="10" kv="1" forcerange="-5 5"/></actuator></mujoco>'''


class EffortCompatibility(unittest.TestCase):
    def test_single_direct_gear_torque_cap_migrates_once(self):
        spec=mujoco.MjSpec.from_string(XML)
        self.assertEqual(normalize_effort_limits(spec),1)
        model=spec.compile()
        np.testing.assert_array_equal(model.jnt_actfrcrange[0],[-5,5])
        self.assertTrue(model.jnt_actfrclimited[0])
        self.assertFalse(model.actuator_forcelimited[0])
        self.assertEqual(normalize_effort_limits(spec),0)

    def test_nonunit_gear_is_rejected(self):
        spec=mujoco.MjSpec.from_string(XML.replace('name="motor"','name="motor" gear="2"'))
        with self.assertRaises(ValueError):normalize_effort_limits(spec)

    def test_multiple_actuators_are_rejected(self):
        extra='<position name="second" joint="hinge" kp="10" forcerange="-5 5"/>'
        spec=mujoco.MjSpec.from_string(XML.replace('</actuator>',extra+'</actuator>'))
        with self.assertRaises(ValueError):normalize_effort_limits(spec)

    def test_unequal_joint_and_actuator_caps_are_rejected(self):
        spec=mujoco.MjSpec.from_string(XML.replace('type="hinge"','type="hinge" actuatorfrcrange="-4 4"'))
        with self.assertRaises(ValueError):normalize_effort_limits(spec)

    def test_asymmetric_caps_are_rejected(self):
        spec=mujoco.MjSpec.from_string(XML.replace('forcerange="-5 5"','forcerange="-4 5"'))
        with self.assertRaises(ValueError):normalize_effort_limits(spec)

    @unittest.skipUnless(os.environ.get("MENAGERIE_DIR"), "requires a declared Menagerie checkout")
    def test_latest_g1_all29_joint_torques_are_equal(self):
        root=Path(os.environ['MENAGERIE_DIR'])
        path=root/'unitree_g1/scene_mjx.xml'
        original=mujoco.MjModel.from_xml_path(str(path))
        spec=mujoco.MjSpec.from_file(str(path))
        self.assertEqual(normalize_effort_limits(spec),29)
        migrated=spec.compile()
        targets=original.actuator_trnid[:,0]
        self.assertEqual(original.nu,29)
        self.assertTrue(np.all(original.actuator_forcelimited))
        self.assertTrue(np.all(migrated.jnt_actfrclimited[targets]))
        self.assertTrue(np.all(~migrated.actuator_forcelimited.astype(bool)))
        np.testing.assert_array_equal(migrated.jnt_actfrcrange[targets],original.actuator_forcerange)
        error=0.;positive=np.zeros(29,bool);negative=np.zeros(29,bool)
        rng=np.random.default_rng(42)
        for velocity in (-1000.,-10.,0.,10.,1000.):
            for command in (-1.,0.,1.):
                a,b=mujoco.MjData(original),mujoco.MjData(migrated)
                a.qpos[:]=b.qpos[:]=original.qpos0
                a.qvel[6:]=b.qvel[6:]=velocity
                a.ctrl[:]=b.ctrl[:]=command+rng.normal(0,.1,29)
                mujoco.mj_forward(original,a);mujoco.mj_forward(migrated,b)
                error=max(error,float(np.abs(a.qfrc_actuator-b.qfrc_actuator).max()))
                force=a.qfrc_actuator[original.jnt_dofadr[targets]]
                positive|=np.isclose(force,original.actuator_forcerange[:,1],atol=1e-10,rtol=0)
                negative|=np.isclose(force,original.actuator_forcerange[:,0],atol=1e-10,rtol=0)
        self.assertLessEqual(error,1e-10)
        self.assertTrue(positive.all() and negative.all())
        from nablamimic.env import build_model,EnvConfig
        model,kp=build_model(EnvConfig())
        effort=model.jnt_actfrcrange[model.actuator_trnid[:,0],1]
        scale=.25*effort/kp
        self.assertTrue(np.isfinite(scale).all() and np.all(scale>0))
        np.testing.assert_array_equal(effort,original.actuator_forcerange[:,1])
        report=dict(time_integration_steps=0,physics_backend_step_calls=0,joint_torque_max_error=error,
            migrated_actuators=29,positive_saturation_verified=int(positive.sum()),negative_saturation_verified=int(negative.sum()),
            effort=effort.tolist(),action_scale=scale.tolist(),native_model_dimensions=[model.nq,model.nv,model.nu,model.nbody],
            meaning='joint torque law only; actuator-force sensor arrays are not asserted equivalent')
        output=os.environ.get('EFFORT_COMPAT_REPORT')
        if output:
            with Path(output).open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':unittest.main()
