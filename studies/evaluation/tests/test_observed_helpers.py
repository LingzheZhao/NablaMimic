"""Fresh current-schema observations retain full supports and invalid lineage."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'instrumentation'))
import observed_helpers
from observed_helpers import FIELDS, evaluate_condition, prepare_trace_root, write_observation
from nablamimic import evaluate as ev


class Capacity:
    def __init__(self, invalid):
        self.invalid = invalid
        self.resets = 0

    def reset(self):
        self.resets += 1

    def read(self, active, nconmax, njmax):
        assert nconmax == 40 and njmax == 5
        steps, worlds = active.shape
        return (dict(validity='invalid' if self.invalid.any() else 'no_observed_capacity_loss'),
                self.invalid, np.zeros((steps, 4), np.int32),
                *[np.zeros((steps, 4, worlds), np.int32) for _ in range(3)])


def example():
    steps, worlds = 3, 40
    names = [f'clip_{i:02d}' for i in range(worlds)]
    env = SimpleNamespace(lengths=np.full(worlds, steps + 1), max_reward=.1,
                          warp={40: SimpleNamespace(nconmax=1, njmax=5)})
    tsum = np.zeros((worlds, len(ev.TERMS)))
    tsum[:, 0] = .3
    res = dict(fall_t=np.full(worlds, -1), cause=np.zeros(worlds, np.int32),
               tsum=tsum, tbin=tsum[:, None, :], cbin=np.full((worlds, 1), steps),
               jerr=np.zeros((worlds, 29)), jabs=np.zeros((worlds, 29)))
    trace = dict(frame=np.broadcast_to(np.arange(steps)[:, None], (steps, worlds)),
                 active=np.ones((steps, worlds), bool), finite=np.ones((steps, worlds), bool),
                 reward_finite=np.ones((steps, worlds), bool),
                 native_reward_terms=np.zeros((steps, worlds, len(ev.TERMS))),
                 terminated=np.zeros((steps, worlds), bool),
                 pelvis_height=np.ones((steps, worlds)), reference_pelvis_height=np.ones((steps, worlds)),
                 torso_gravity_z=-np.ones((steps, worlds)), reference_torso_gravity_z=-np.ones((steps, worlds)),
                 action=np.zeros((steps, worlds, 29)), setpoint=np.zeros((steps, worlds, 29)),
                 body_world_lin_vel=np.zeros((steps, worlds, 14, 3)),
                 body_world_ang_vel=np.zeros((steps, worlds, 14, 3)),
                 root_world_velocity=np.zeros((steps, worlds, 3)), root_body_velocity=np.zeros((steps, worlds, 3)),
                 source_frame=np.broadcast_to(np.arange(steps)[:, None], (steps, worlds)),
                 qpos=np.zeros((steps, worlds, 36)), qvel=np.zeros((steps, worlds, 35)),
                 body_pos=np.zeros((steps, worlds, 14, 3)), body_quat=np.zeros((steps, worlds, 14, 4)),
                 metrics=np.zeros((steps, worlds, len(FIELDS))))
    return env, names, res, trace


class ObservedHelpersTest(unittest.TestCase):
    def test_two_writers_preserve_every_array_scalar_and_clip_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots, results = [], []
            for workers in (1, 2):
                env, names, res, trace = example()
                res['fall_t'][0], res['cause'][0] = 2, 1
                res['tsum'][0, 0], res['cbin'][0, 0] = .2, 2
                trace['active'][2, 0] = False
                trace['terminated'][1, 0] = True
                trace['qpos'][:] = np.arange(trace['qpos'].size).reshape(trace['qpos'].shape) / 10
                invalid = np.zeros((3, 40), bool)
                invalid[1, 1] = True
                root = Path(temporary) / f'workers-{workers}'
                prepare_trace_root(root, names)
                results.append(write_observation(env, names, res, trace, Capacity(invalid), root, 3, 'ema',
                                                  writer_workers=workers))
                roots.append(root)
            self.assertEqual(results[0], results[1])
            self.assertEqual([r['clip'] for r in results[1][1]['per_clip']], names)
            files = sorted(path.name for path in roots[0].glob('*.npz'))
            self.assertEqual(len(files), 42)
            self.assertEqual(files, sorted(path.name for path in roots[1].glob('*.npz')))
            for name in files:
                with np.load(roots[0] / name) as a, np.load(roots[1] / name) as b:
                    self.assertEqual(a.files, b.files)
                    for field in a.files:
                        np.testing.assert_array_equal(a[field], b[field])
            for name in ('it3_metrics.json', 'it3_same_execution_scalars.json'):
                self.assertEqual(json.loads((roots[0] / name).read_text()),
                                 json.loads((roots[1] / name).read_text()))
            self.assertFalse(list(roots[1].glob('*.partial')))

    def test_writer_slot_is_reserved_before_constructing_the_third_payload(self):
        release = threading.Event()
        two_started = threading.Event()
        constructing_third = threading.Event()
        lock = threading.Lock()
        started, created, errors = [], [], []

        def save(path, payload):
            self.assertFalse(payload['values'].flags.writeable)
            with lock:
                started.append(path)
                if len(started) == 2:
                    two_started.set()
            if not release.wait(5):
                raise RuntimeError('Test writer was not released')

        def make_payload(index):
            created.append(index)
            if index == 2:
                constructing_third.set()
            return {'values': np.full((8,), index)}

        def produce():
            try:
                with observed_helpers._NPZWrites(2) as writes:
                    for index in range(3):
                        writes.submit(str(index), lambda index=index: make_payload(index))
            except BaseException as error:
                errors.append(error)

        with patch.object(observed_helpers, '_atomic_npz', side_effect=save):
            producer = threading.Thread(target=produce)
            producer.start()
            try:
                self.assertTrue(two_started.wait(5))
                self.assertFalse(constructing_third.wait(.1))
                self.assertEqual(created, [0, 1])
            finally:
                release.set()
                producer.join(5)
            self.assertFalse(producer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(created, [0, 1, 2])

    def test_writer_error_drains_threads_and_preserves_complete_and_partial_evidence(self):
        env, names, res, trace = example()
        before = {t.ident for t in threading.enumerate() if t.name.startswith('nabla-observed-npz')}
        native_save = np.savez_compressed

        def save(stream, **payload):
            if str(stream.name).endswith('it3_clip_00.npz.partial'):
                stream.write(b'incomplete-npz-evidence')
                raise OSError('forced compression failure')
            if str(stream.name).endswith('it3_clip_01.npz.partial'):
                time.sleep(.03)
            return native_save(stream, **payload)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'traces'
            prepare_trace_root(root, names)
            with patch.object(np, 'savez_compressed', side_effect=save):
                with self.assertRaisesRegex(OSError, 'forced compression failure'):
                    write_observation(env, names, res, trace, Capacity(np.zeros((3, 40), bool)),
                                      root, 3, 'ema', writer_workers=2)
            self.assertFalse((root / 'it3_clip_00.npz').exists())
            self.assertTrue((root / 'it3_clip_00.npz.partial').exists())
            self.assertTrue((root / 'it3_clip_01.npz').is_file())
            self.assertTrue((root / 'it3_native_aggregates.npz').is_file())
            self.assertTrue((root / 'it3_capacity.npz').is_file())
            self.assertFalse((root / 'it3_metrics.json').exists())
        after = {t.ident for t in threading.enumerate() if t.name.startswith('nabla-observed-npz')}
        self.assertEqual(after, before)

    def test_fallen_tail_retains_denominator_and_all_forty_outputs(self):
        env, names, res, trace = example()
        res['fall_t'][0], res['cause'][0] = 2, 1
        res['tsum'][0, 0], res['cbin'][0, 0] = .2, 2
        trace['active'][2, 0] = False
        trace['terminated'][1, 0] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'traces'
            prepare_trace_root(root, names)
            scalar, report = write_observation(env, names, res, trace,
                Capacity(np.zeros((3, 40), bool)), root, 3, 'ema')
            self.assertAlmostEqual(scalar['full_norm_return'], (39 + 2 / 3) / 40)
            self.assertEqual(report['complete_clip_count'], 39)
            self.assertFalse(report['accepted'])
            self.assertEqual(len(list(root.glob('*.npz'))), 42)
            with np.load(root / 'it3_clip_00.npz') as raw:
                self.assertEqual(raw['qpos'].shape[0], 2)
            with np.load(root / 'it3_clip_39.npz') as raw:
                self.assertEqual(raw['qpos'].shape[0], 3)

    def test_invalid_physics_keeps_raw_rows_and_invalid_descendants(self):
        env, names, res, trace = example()
        invalid = np.zeros((3, 40), bool)
        invalid[1, 0] = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'traces'
            prepare_trace_root(root, names)
            _, report = write_observation(env, names, res, trace, Capacity(invalid), root, 3, 'ema')
            first = report['per_clip'][0]
            self.assertEqual((first['executed_steps'], first['valid_metric_steps'], first['invalid_metric_steps']), (3, 1, 2))
            self.assertEqual(report['physics_validity'], 'invalid')
            self.assertFalse(report['accepted'])
            with np.load(root / 'it3_clip_00.npz') as raw:
                np.testing.assert_array_equal(raw['invalid_lineage'], [False, True, True])
                self.assertEqual(raw['qpos'].shape[0], 3)

    def test_condition_boundary_and_timings_stay_outside_scientific_record(self):
        env, names, res, trace = example()
        capacity = Capacity(np.zeros((3, 40), bool))
        checkpoint = dict(iter=3, wall_s=10., completed_updates=3, ema_updates=3)

        def run_actor(actor, before_rollout):
            before_rollout()
            return (res, trace), dict(condition_index=1, persistent=True, native_resets=[]), dict(
                initialization_s=.1, compilation_s=0., compiled_reused=True, condition_reset_s=.01,
                execution_s=.02, readback_s=.01)

        session = SimpleNamespace(env=env, names=names, run=Path('/fixed/run'), solver_iterations=10,
                                  forward_only=True, load_checkpoint=lambda it, policy: (checkpoint, policy, {}),
                                  run_actor=run_actor)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_trace_root(root / 'traces', names)
            observed, report = evaluate_condition(session, capacity, 3, 'ema', root / 'observed.json', root / 'traces')
            self.assertEqual(capacity.resets, 1)
            self.assertTrue(report['accepted'])
            rec = observed['it3/nominal/ema']
            self.assertNotIn('reset_boundary', rec)
            self.assertNotIn('phase_timings', rec)
            self.assertTrue(json.loads((root / 'observed.boundary.json').read_text())['persistent'])
            self.assertTrue(json.loads((root / 'observed.timings.json').read_text())['compiled_reused'])


if __name__ == '__main__':
    unittest.main()
