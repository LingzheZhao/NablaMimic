"""Generated MuJoCo model tests use FK and wrench mathematics without stepping."""
import unittest

import numpy as np
import torch

from nablamimic.mujoco_source_adapter import MuJoCoSourceAdapter


class MuJoCoAdapterTest(unittest.TestCase):
    def model(self):
        import mujoco
        return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".005" integrator="implicitfast"/>
        <worldbody><body name="torso_link"><freejoint/><geom type="sphere" size=".1" mass="1"/>
        <body pos="0 0 .2"><joint name="hinge"/><geom type="capsule" size=".03 .1" mass=".2"/></body>
        </body></worldbody><actuator><position joint="hinge" kp="10"/></actuator></mujoco>''')

    def test_fk_keeps_double_state_and_does_not_change_warmstart(self):
        model=self.model()
        adapter=MuJoCoSourceAdapter(model,iterations=7)
        q=torch.tensor(model.qpos0.copy(),dtype=torch.float64)[None]
        q[0,0]=1.+1.e-10
        velocity=torch.arange(model.nv,dtype=torch.float64)[None]*.01
        warm=torch.arange(model.nv,dtype=torch.float64)[None]*.02
        rows=adapter.synchronized_data(q,velocity,warm)
        np.testing.assert_array_equal(rows[0].qpos,q[0].numpy())
        self.assertNotEqual(rows[0].qpos[0],float(q[0,0].float()))
        np.testing.assert_array_equal(rows[0].qacc_warmstart,warm[0].numpy())
        before=rows[0].qacc_warmstart.copy()
        normalized,positions,rotations,velocities,centers=adapter.kinematics(q,velocity,torch.zeros(1,3))
        self.assertEqual(normalized.dtype,torch.float64)
        np.testing.assert_array_equal(rows[0].qacc_warmstart,before)
        np.testing.assert_allclose(positions[0].numpy(),rows[0].xpos,rtol=0,atol=0)
        self.assertEqual(rows[0].time,0.)
        self.assertEqual(model.opt.iterations,7)

    def test_contact_wrench_world_transform_and_reactions(self):
        from types import SimpleNamespace
        adapter=MuJoCoSourceAdapter.__new__(MuJoCoSourceAdapter)
        adapter.model=SimpleNamespace(nbody=3,geom_bodyid=np.array([0,1,2]))
        rotation=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        contact=SimpleNamespace(efc_address=0,frame=rotation.reshape(-1),pos=np.array([1.,0.,0.]),geom1=0,geom2=1)
        def force(model,data,index,out):
            out[:]=[2.,0.,0.,0.,0.,.5]
        adapter.mujoco=SimpleNamespace(mj_contactForce=force)
        data=SimpleNamespace(ncon=1,contact=[contact])
        terrain,count,all_wrench,all_count=adapter._wrenches(data)
        np.testing.assert_allclose(all_wrench[1],[0.,2.,0.,0.,0.,2.5])
        np.testing.assert_allclose(all_wrench[0],-all_wrench[1])
        np.testing.assert_allclose(all_wrench.sum(0),0.)
        np.testing.assert_array_equal(terrain,all_wrench)
        np.testing.assert_array_equal(count,[1,1,0])


if __name__=='__main__':
    unittest.main()
