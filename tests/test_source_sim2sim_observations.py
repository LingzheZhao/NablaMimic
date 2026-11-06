"""Generated same-state comparison quantities and ray-selection discontinuities."""
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_source_sim2sim_observations import compare_observations, delta_statistics
sys.path.pop(0)


def snapshot():
    shapes=dict(qpos=(1,36),qvel=(1,35),fk_position=(1,31,3),fk_quaternion=(1,31,4),
        subtree_com=(1,31,3),cvel=(1,31,6),proprio=(1,93),goal=(1,670),policy_goal=(1,670),
        policy_observation=(1,1320),processed_depth=(1,576),policy_action=(1,29),
        bounded_action=(1,29),control_target=(1,29),ray_origins=(1,3,3),ray_directions=(1,3,3))
    arrays={key:np.zeros(shape) for key,shape in shapes.items()}
    arrays.update(ray_ranges=np.array([[1.,2.,1e6]]),ray_geometry_ids=np.array([[7,8,-1]]))
    return arrays


class SourceSim2simObservationTest(unittest.TestCase):
    def test_component_delta_reports_units_maximum_and_rms(self):
        result=delta_statistics([0.,0.,0.],[3.,4.,0.],'m')
        self.assertEqual(result['units'],'m')
        self.assertEqual(result['max_absolute'],4.)
        self.assertAlmostEqual(result['rms'],np.sqrt(25/3))
        with self.assertRaises(ValueError):
            delta_statistics([0.],[0.,1.],'m')

    def test_empty_and_nonfinite_delta_remain_explicit(self):
        empty=delta_statistics([1.],[2.],'m',np.array([False]))
        self.assertEqual(empty['elements'],0)
        self.assertIsNone(empty['max_absolute'])
        self.assertIsNone(empty['rms'])
        nonfinite=delta_statistics([0.],[np.nan],'m')
        self.assertFalse(nonfinite['finite'])
        self.assertIsNone(nonfinite['rms'])

    def test_ray_discontinuities_separate_hit_and_geom_changes_from_range_error(self):
        left,right=snapshot(),snapshot()
        right['ray_geometry_ids'][:]=[7,9,6]
        right['ray_ranges'][:]=[1.002,4.,.5]
        result=compare_observations(left,right)['ray_selection']
        self.assertEqual(result['hit_mismatches'],1)
        self.assertEqual(result['geometry_id_mismatches'],2)
        self.assertEqual(result['different_geom_both_hit'],1)
        self.assertEqual(result['same_geom_both_hit'],1)
        self.assertAlmostEqual(result['same_geom_range']['max_absolute'],.002)
        self.assertEqual(result['different_geom_range']['max_absolute'],2.)
        self.assertEqual(result['both_hit_range']['elements'],2)

    def test_raw_and_actor_boundary_goal_and_history_are_measured_separately(self):
        left,right=snapshot(),snapshot()
        right['goal'][0,580]=.2
        right['policy_goal'][0,580]=.1
        right['proprio'][0,6]=.3
        right['policy_observation'][0,48]=.4
        result=compare_observations(left,right)
        self.assertEqual(result['goal_position']['max_absolute'],.2)
        self.assertEqual(result['actor_goal_position']['max_absolute'],.1)
        self.assertEqual(result['proprio_joint_position']['max_absolute'],.3)
        self.assertEqual(result['actor_history_joint_position']['max_absolute'],.4)
        self.assertEqual(result['goal_position']['units'],'m current reference torso')
        self.assertEqual(result['actor_history_joint_position']['units'],'rad relative to defaults')


if __name__=='__main__':
    unittest.main()
