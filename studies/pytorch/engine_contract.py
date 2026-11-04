"""Explicit native-capacity expectations for provisioned validation studies.

These are declared run inputs, never inferred from an artifact or used to change
Engine kernels. The current G1 tape header remains contact/row/limit/iteration
widths 7/9/8/8 bits.
"""
import os
import sys


def add_engine_capacity_arguments(parser):
    parser.add_argument('--expected-max-contacts', type=int, required=True)
    parser.add_argument('--expected-constraint-rows', type=int, required=True)
    parser.add_argument('--expected-tape-words', type=int, required=True)
    parser.add_argument('--expected-zone-words', type=int, required=True)


def expected_engine_capacity(args):
    contacts = args.expected_max_contacts
    rows = args.expected_constraint_rows
    words = args.expected_tape_words
    zones = args.expected_zone_words
    if not 1 <= contacts <= 127 or not 1 <= rows <= 511:
        raise ValueError('Declared capacity exceeds the current 7/9-bit tape count fields')
    if words <= 0 or words % 32 or zones != (rows + 31) // 32:
        raise ValueError('Expected tape alignment or row-zone word count is inconsistent')
    return dict(max_contacts=contacts, constraint_rows=rows, record_words=words,
                zone_words=zones, header_bits=[7, 9, 8, 8])


def capacity_validator_command(args, output_root):
    return [sys.executable, str(args.capacity_validator), '--cpu',
            '--out', str(output_root / 'capacity-check'),
            '--engine-capacity-revision', args.engine_capacity_revision]


def capacity_validator_environment():
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}
