"""Generated terminal-inclusive source evaluation supports and trial decisions."""
import unittest

import numpy as np
from nablamimic.source_evaluation_metrics import summarize_source_chain


class SourceEvaluationMetricsTest(unittest.TestCase):
    def test_terminal_error_participates_in_mean_and_decision(self):
        errors=np.zeros((3,14))
        errors[-1]=.33
        result=summarize_source_chain(errors,7,[False]*3,True,2,2)
        self.assertAlmostEqual(result['mean_world_body_error_m'],.11)
        self.assertFalse(result['strict_pass'])

    def test_complete_low_error_chain_passes_and_termination_prevents_pass(self):
        errors=np.full((4,14),.05)
        result=summarize_source_chain(errors,7,[False]*4,True,3,3)
        self.assertTrue(result['strict_pass'])
        result=summarize_source_chain(errors,7,[False,False,False,True],True,3,3)
        self.assertFalse(result['strict_pass'])
        self.assertAlmostEqual(result['first_tracking_termination_s'],.06)

    def test_short_requested_window_is_not_reference_completion(self):
        result=summarize_source_chain(np.zeros((3,14)),7,[False]*3,False,2,2)
        self.assertFalse(result['strict_pass'])
        self.assertEqual(result['states'],3)

    def test_torso_bound_is_separate_from_fourteen_body_mean(self):
        errors=np.zeros((3,14))
        errors[:,7]=.11
        result=summarize_source_chain(errors,7,[False]*3,True,2,2)
        self.assertLess(result['mean_world_body_error_m'],.1)
        self.assertFalse(result['strict_pass'])


if __name__=='__main__':
    unittest.main()
