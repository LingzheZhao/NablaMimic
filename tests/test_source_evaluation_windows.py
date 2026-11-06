"""Generated local-window supports, terminal decisions and reference activity."""
from pathlib import Path
from dataclasses import dataclass
import os
import subprocess
import sys
import unittest

import numpy as np
import mujoco


SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from evaluate_source_perceptive import CAMERA_INPUT_DEFINITIONS, reference_window_activity, select_motion_indices, self_only_camera_geometry, set_camera_input, set_evaluation_conditions, source_window, summarize_window
from render_source_trajectory import reference_frame_indices
sys.path.pop(0)


class SourceEvaluationWindowTest(unittest.TestCase):
    def test_self_only_camera_excludes_world_and_preserves_dynamic_geoms_and_ids(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="terrain" type="plane" size="1 1 .1" group="0"/>
          <body name="robot"><freejoint/>
            <geom name="robot_visual" type="sphere" size=".1" group="2"/>
            <geom name="robot_collision" type="sphere" size=".1" group="3"/>
            <body name="hand" pos=".3 0 0"><geom name="hand_visual" type="sphere" size=".05" group="2"/></body>
          </body>
          <geom name="static_visual" type="box" size=".1 .1 .1" group="2"/>
        </worldbody></mujoco>''')
        @dataclass(frozen=True)
        class VisibleGeom:
            geom_id: int
            body_id: int
        @dataclass(frozen=True)
        class Geometry:
            geoms: tuple
            body_count: int
            cuda_backend: object = None
        visible=tuple(VisibleGeom(i,int(model.geom_bodyid[i])) for i in range(model.ngeom)
                      if model.geom_group[i] in (0,2) and model.geom_rgba[i,3] != 0)
        original=Geometry(visible,model.nbody)
        selected=self_only_camera_geometry(original,'cpu')
        self.assertIsNot(selected,original)
        self.assertEqual(original.geoms,visible)
        self.assertEqual(selected.body_count,model.nbody)
        self.assertEqual([geom.geom_id for geom in selected.geoms],
                         [model.geom('robot_visual').id,model.geom('hand_visual').id])
        self.assertTrue(all(geom.body_id != 0 for geom in selected.geoms))
        self.assertTrue(all(any(geom is old for old in visible) for geom in selected.geoms))
        self.assertEqual(model.ngeom,5)
        with self.assertRaises(ValueError):
            self_only_camera_geometry(Geometry((visible[0],),model.nbody),'cpu')

    def test_self_only_depth_is_not_replaced_after_camera_generation(self):
        observation=np.arange(1320,dtype=np.float32)[None]
        original=observation.copy()
        set_camera_input(observation,'self-only')
        np.testing.assert_array_equal(observation,original)
        self.assertIn('Static world camera occluders are excluded',CAMERA_INPUT_DEFINITIONS['self-only'])
        self.assertIn('noise and image processing run normally',CAMERA_INPUT_DEFINITIONS['self-only'])

    def test_self_only_eval_helper_constructs_mujoco_sensor_without_dataclass_replacement(self):
        from types import SimpleNamespace
        from nablamimic.mujoco_source_camera import MuJoCoRayCamera
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="terrain" type="plane" size="1 1 .1" group="0"/>
          <body><freejoint/><geom name="robot" type="sphere" size=".1" group="2"/></body>
        </worldbody></mujoco>''')
        adapter=SimpleNamespace(model=model,data_batch=())
        original=MuJoCoRayCamera(adapter)
        selected=self_only_camera_geometry(original,'cpu')
        self.assertIsNot(selected,original)
        self.assertIs(selected.adapter,adapter)
        self.assertTrue(original.include_static)
        self.assertFalse(selected.include_static)
        self.assertEqual([geom.geom_id for geom in selected.geoms],[model.geom('robot').id])

    def test_evaluation_owns_no_gradient_independent_of_training_gradient(self):
        for gradient in ({},{'source_depth_gradient':True},{'source_depth_gradient':False}):
            for reset in ('source-reset','exact-reference'):
                conditions=dict(source_motion_start_height_offset=.1,
                    source_ensure_link_below_zero_ground=True,
                    source_depth_noise=True,residual_action=True,term_penalty=[2.,.02,.1],**gradient)
                set_evaluation_conditions(conditions,reset)
                self.assertIs(conditions['source_depth_gradient'],False)
                self.assertEqual(conditions['source_motion_start_height_offset'],.1 if reset=='source-reset' else 0.)
                self.assertEqual(conditions['source_ensure_link_below_zero_ground'],reset=='source-reset')
                self.assertTrue(conditions['source_depth_noise'])
                self.assertTrue(conditions['residual_action'])
                self.assertEqual(conditions['term_penalty'],[2.,.02,.1])

    def test_camera_input_cli_accepts_only_declared_choices(self):
        command=[sys.executable,'-B',str(SCRIPTS/'evaluate_source_perceptive.py'),'--camera-input']
        environment={**os.environ,'PYTHONPATH':str(SCRIPTS.parent/'src')}
        for mode in ('measured','constant-far','self-only'):
            result=subprocess.run([*command,mode,'--help'],env=environment,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([*command,'invalid','--help'],env=environment,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)

    def test_default_measured_camera_preserves_observation_values(self):
        observation=np.arange(2*1320,dtype=np.float32).reshape(2,1320)
        original=observation.copy()
        set_camera_input(observation)
        np.testing.assert_array_equal(observation,original)
        set_camera_input(observation,'measured')
        np.testing.assert_array_equal(observation,original)

    def test_constant_far_replaces_only_depth_in_the_shared_observation(self):
        observation=np.arange(2*1320,dtype=np.float32).reshape(2,1320)
        observation[:,744:]=np.tile([0.,1.],288)
        original=observation.copy()
        goal=np.arange(2*670,dtype=np.float32).reshape(2,670)
        original_goal=goal.copy()
        set_camera_input(observation,'constant-far')
        np.testing.assert_array_equal(observation[:,:744],original[:,:744])
        np.testing.assert_array_equal(observation[:,744:],np.ones((2,576)))
        np.testing.assert_array_equal(goal,original_goal)
        # Both the actor's depth view and the trajectory's saved depth use this owner.
        actor_depth=observation[:,744:].reshape(2,1,18,32)
        saved_depth=observation[0,744:]
        np.testing.assert_array_equal(actor_depth[0].ravel(),saved_depth)

    def test_camera_input_zero_one_boundaries_and_current_schema(self):
        for value in (0.,1.):
            observation=np.full((1,1320),value,dtype=np.float32)
            set_camera_input(observation,'constant-far')
            np.testing.assert_array_equal(observation[:,:744],value)
            np.testing.assert_array_equal(observation[:,744:],1.)
        for shape in ((1320,),(1,1319),(1,1321),(1,1,1320)):
            with self.assertRaises(ValueError):
                set_camera_input(np.zeros(shape),'constant-far')
        with self.assertRaises(ValueError):
            set_camera_input(np.zeros((1,1320)),'invalid')

    def test_motion_selection_defaults_to_all_original_indices(self):
        self.assertEqual(select_motion_indices(None,10),list(range(10)))

    def test_motion_selection_retains_library_order_and_original_clip_ids(self):
        self.assertEqual(select_motion_indices([9,0,4],10),[0,4,9])
        self.assertEqual(select_motion_indices([9],10),[9])

    def test_motion_selection_rejects_duplicates_and_out_of_range(self):
        for indices in ([0,0],[-1],[10],[]):
            with self.assertRaises(ValueError):
                select_motion_indices(indices,10)

    def test_fraction_floor_and_remaining_cap(self):
        self.assertEqual(source_window(10,.25,100),
            dict(start_offset=2,remaining_controls=7,requested_controls=7))
        self.assertEqual(source_window(10,.5,2),
            dict(start_offset=4,remaining_controls=5,requested_controls=2))
        self.assertEqual(source_window(10,np.nextafter(1.,0.),None),
            dict(start_offset=8,remaining_controls=1,requested_controls=1))
        self.assertEqual(source_window(2,.75,100)['requested_controls'],1)

    def test_invalid_window_boundaries(self):
        for fraction in (-.01,1.,np.nan,np.inf):
            with self.assertRaises(ValueError):
                source_window(10,fraction,100)
        for length, maximum in ((1,None),(10,0)):
            with self.assertRaises(ValueError):
                source_window(length,0.,maximum)

    def test_local_window_and_local_clip_end_keep_full_reference_false(self):
        window=source_window(9,.5,2)
        reading=summarize_window(np.full((3,14),.05),7,[False]*3,False,2,window)
        self.assertTrue(reading['requested_window_complete'])
        self.assertTrue(reading['window_fidelity_pass'])
        self.assertFalse(reading['reference_complete'])
        self.assertFalse(reading['strict_pass'])
        window=source_window(9,.5,None)
        reading=summarize_window(np.zeros((5,14)),7,[False]*5,True,4,window)
        self.assertTrue(reading['clip_end_reached'])
        self.assertTrue(reading['window_fidelity_pass'])
        self.assertFalse(reading['reference_complete'])

    def test_complete_full_reference_and_short_window_are_distinct(self):
        window=source_window(4,0.,None)
        reading=summarize_window(np.zeros((4,14)),7,[False]*4,True,3,window)
        self.assertTrue(reading['reference_complete'])
        self.assertTrue(reading['strict_pass'])
        reading=summarize_window(np.zeros((3,14)),7,[False]*3,False,2,window)
        self.assertFalse(reading['requested_window_complete'])
        self.assertFalse(reading['window_fidelity_pass'])

    def test_initial_and_terminal_termination_fail_window_fidelity(self):
        window=source_window(4,0.,None)
        for terminated in ([True,False,False,False],[False,False,False,True]):
            reading=summarize_window(np.zeros((4,14)),7,terminated,True,3,window)
            self.assertTrue(reading['requested_window_complete'])
            self.assertFalse(reading['window_fidelity_pass'])
            self.assertFalse(reading['strict_pass'])
        reading=summarize_window(np.zeros((2,14)),7,[False,True],False,1,window)
        self.assertFalse(reading['requested_window_complete'])
        self.assertFalse(reading['window_fidelity_pass'])

    def test_terminal_error_and_torso_bound_participate_in_fidelity(self):
        window=source_window(4,.5,2)
        errors=np.zeros((3,14))
        errors[-1]=.33
        reading=summarize_window(errors,7,[False]*3,True,2,window)
        self.assertFalse(reading['window_fidelity_pass'])
        errors[:]=0.
        errors[:,7]=.11
        reading=summarize_window(errors,7,[False]*3,True,2,window)
        self.assertFalse(reading['window_fidelity_pass'])

    def test_reference_activity_uses_every_requested_interval(self):
        positions=np.zeros((4,14,3))
        positions[:,:,0]=np.arange(4)[:,None]*.04
        activity=reference_window_activity(positions,7)
        self.assertAlmostEqual(activity['reference_world_body_speed_mean_m_s'],2.)
        self.assertAlmostEqual(activity['reference_torso_displacement_m'],.12)

    def test_renderer_preserves_actual_start_and_continuous_terminal_support(self):
        reading=dict(source_window(10,.5,100),start_fraction=.5,controls=5,
                     clip_start_frame=20,start_frame=24)
        frames=np.arange(24,29)
        np.testing.assert_array_equal(reference_frame_indices(reading,20,10,frames,np.array([29])),np.arange(4,10))
        with self.assertRaises(ValueError):
            reference_frame_indices(reading,20,10,np.array([24,25,27,28,29]),np.array([30]))
        with self.assertRaises(ValueError):
            reference_frame_indices(dict(reading,start_frame=20),20,10,frames,np.array([29]))
        with self.assertRaises(ValueError):
            reference_frame_indices(dict(reading,requested_controls=6),20,10,frames,np.array([29]))


if __name__=='__main__':
    unittest.main()
