"""CPU checks for explicit capacity declarations and the validator boundary."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


PATH = Path(__file__).resolve().parents[1] / 'studies/pytorch/engine_contract.py'
SPEC = importlib.util.spec_from_file_location('study_engine_contract', PATH)
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


class EngineContractTests(unittest.TestCase):
    def arguments(self, **changes):
        values = dict(expected_max_contacts=96, expected_constraint_rows=416,
                      expected_tape_words=1504, expected_zone_words=13)
        return SimpleNamespace(**{**values, **changes})

    def test_declared_capacity_and_header(self):
        parser = argparse.ArgumentParser()
        CONTRACT.add_engine_capacity_arguments(parser)
        args = parser.parse_args(['--expected-max-contacts', '96',
                                  '--expected-constraint-rows', '416',
                                  '--expected-tape-words', '1504',
                                  '--expected-zone-words', '13'])
        self.assertEqual(CONTRACT.expected_engine_capacity(args), dict(
            max_contacts=96, constraint_rows=416, record_words=1504,
            zone_words=13, header_bits=[7, 9, 8, 8]))

    def test_count_fields_cannot_overflow(self):
        for changes in ({'expected_max_contacts': 128}, {'expected_constraint_rows': 512}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CONTRACT.expected_engine_capacity(self.arguments(**changes))

    def test_tape_alignment_and_all_row_bits(self):
        for changes in ({'expected_tape_words': 1503}, {'expected_zone_words': 12}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CONTRACT.expected_engine_capacity(self.arguments(**changes))

    def test_validator_is_cpu_only_and_uses_fresh_child_output(self):
        args = SimpleNamespace(capacity_validator=Path('/declared/validator.py'),
                               engine_capacity_revision='current-engine-revision')
        command = CONTRACT.capacity_validator_command(args, Path('/fresh/run'))
        self.assertEqual(command[1:], ['/declared/validator.py', '--cpu', '--out',
                                      '/fresh/run/capacity-check', '--engine-capacity-revision',
                                      'current-engine-revision'])
        environment = CONTRACT.capacity_validator_environment()
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(environment['JAX_PLATFORMS'], 'cpu')


if __name__ == '__main__':
    unittest.main()
