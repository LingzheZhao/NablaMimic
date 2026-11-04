"""Fresh arithmetic fixtures for the small offline three-child sum study."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location('subtree_sum_orders',
    Path(__file__).resolve().parents[1]/'studies/evaluation/subtree_sum_orders.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_round_each_add_instead_of_one_float64_sum():
    before = np.zeros((1, 4, 3), dtype=np.float32)
    before[0, :, 0] = [2**24, 1, -(2**24), 1]
    after = before.copy()
    after[0, 0, 0] = 1
    original = before.copy()
    result = study.analyze_pair(before, after, 0, [1, 2, 3])
    assert result['all_coordinates_match']
    assert set(result['coordinates'][0]['candidate_values']) == {0., 1., 2.}
    assert float(before[0, :, 0].astype(np.float64).sum()) == 2.
    assert study.equal_bits(before, original)


def test_coordinates_may_require_different_orders_and_invalid_sum_fails():
    before = np.zeros((1, 4, 3), dtype=np.float32)
    before[0, :, 0] = [2**24, 1, -(2**24), 1]
    before[0, :, 1] = [2**24, -(2**24), 1, 1]
    after = before.copy()
    after[0, 0] = 0
    result = study.analyze_pair(before, after, 0, [1, 2, 3])
    assert result['all_coordinates_match']
    assert result['worlds_without_one_common_xyz_order'] == 1
    after[0, 0, 0] = 3
    assert not study.analyze_pair(before, after, 0, [1, 2, 3])['all_coordinates_match']
