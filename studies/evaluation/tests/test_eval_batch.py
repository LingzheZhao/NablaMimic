"""Actual short subprocesses exercise scheduling and process ownership; no simulator."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_batch import BatchRunner, Condition, worker_environment


FAKE = r'''
import json,os,sys,time
from pathlib import Path
folder,stage,delay,code=sys.argv[1:]
start=time.monotonic()
time.sleep(float(delay))
record=dict(start=start,end=time.monotonic(),pid=os.getpid(),device=os.environ['CUDA_VISIBLE_DEVICES'],
            memory_fraction=float(os.environ['XLA_CLIENT_MEM_FRACTION']),preallocate=os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'])
with (Path(folder)/(stage+'.fake.json')).open('x') as stream: json.dump(record,stream)
raise SystemExit(int(code))
'''


def still_running(pid):
    path = Path(f'/proc/{pid}/status')
    if not path.exists():
        return False
    try:
        state = next(line for line in path.read_text().splitlines() if line.startswith('State:'))
    except FileNotFoundError:
        return False
    return '\tZ' not in state


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nabla-eval-runner-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, *, workers=2, seconds=10, factory=None, conditions=None, workers_per_gpu=1, memory_fraction=None):
        conditions = conditions or [Condition(1, 'ema'), Condition(2, 'ema')]
        if factory is None:
            factory = lambda c, stage, folder, success: [sys.executable, '-c', FAKE, str(folder), stage, '.20', '0']
        return BatchRunner(conditions=conditions,
                           worker_envs=[{**os.environ, 'CUDA_VISIBLE_DEVICES': f'test-device-{i//workers_per_gpu}',
                                         'XLA_CLIENT_MEM_FRACTION': str(.9/workers_per_gpu if memory_fraction is None else memory_fraction)}
                                        for i in range(workers)],
                           out=self.root/'out', seconds=seconds, command_factory=factory, grace_s=.1, poll_s=.01,
                           workers_per_gpu=workers_per_gpu)

    def test_two_workers_overlap_and_run_each_stage_once_on_fixed_devices(self):
        runner = self.runner()
        result = runner.run()
        self.assertTrue(result['all_processes_completed'])
        records = []
        for i, condition in enumerate(result['conditions']):
            folder = runner.out / condition['name']
            self.assertEqual(condition['worker'], i)
            self.assertEqual(len(list(folder.glob('*.fake.json'))), 2)
            original = json.loads((folder/'original.fake.json').read_text())
            observed = json.loads((folder/'observed.fake.json').read_text())
            self.assertEqual(original['device'], f'test-device-{i}')
            self.assertEqual(observed['device'], original['device'])
            self.assertLessEqual(original['end'], observed['start'])
            records.append(original)
        self.assertLess(max(r['start'] for r in records), min(r['end'] for r in records))
        self.assertEqual(json.loads((runner.out/'manifest.json').read_text()), result)

    def test_explicit_single_gpu_workers_overlap_with_split_memory_limits(self):
        runner = self.runner(workers=2, workers_per_gpu=2, memory_fraction=.4)
        result = runner.run()
        self.assertTrue(result['all_processes_completed'])
        self.assertEqual(result['provenance']['workers_per_gpu'], 2)
        self.assertEqual(result['provenance']['gpu_count'], 1)
        self.assertEqual(result['provenance']['jax_memory_fraction_limits_per_gpu'], {'test-device-0': .8})
        records = []
        for condition in result['conditions']:
            record = json.loads((runner.out/condition['name']/'original.fake.json').read_text())
            self.assertEqual(record['device'], 'test-device-0')
            self.assertEqual(record['memory_fraction'], .4)
            self.assertEqual(record['preallocate'], 'false')
            records.append(record)
        self.assertLess(max(r['start'] for r in records), min(r['end'] for r in records))

    def test_explicit_sharing_rejects_overcommitted_allocator_limits_and_mismatched_count(self):
        with self.assertRaisesRegex(ValueError, 'Summed worker JAX memory fractions'):
            self.runner(workers=2, workers_per_gpu=2, memory_fraction=.6)
        self.assertFalse((self.root/'out').exists())
        with self.assertRaisesRegex(ValueError, 'Worker sharing must match'):
            self.runner(workers=3, workers_per_gpu=2, memory_fraction=.4)
        self.assertFalse((self.root/'out').exists())
        with self.assertRaisesRegex(ValueError, 'JAX memory fraction must be finite'):
            self.runner(workers=2, workers_per_gpu=2, memory_fraction=float('nan'))

    def test_physical_device_aliases_cannot_bypass_explicit_sharing(self):
        from eval_batch import resolve_devices
        response = subprocess.CompletedProcess([], 0, stdout='GPU-same,NVIDIA CMP 170HX,65536\n', stderr='')
        with mock.patch('eval_batch.subprocess.run', return_value=response):
            with self.assertRaisesRegex(ValueError, 'multiple aliases'):
                resolve_devices(['0', 'GPU-same'])

    def test_deadline_cleans_both_explicit_shared_gpu_processes(self):
        script = 'import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)'
        factory = lambda c, stage, folder, success: [sys.executable, '-c', script, str(folder/'owned-pid')]
        runner = self.runner(workers=2, workers_per_gpu=2, memory_fraction=.4, seconds=.5, factory=factory)
        result = runner.run()
        self.assertEqual(result['cancellation_reason'], 'deadline')
        for condition in result['conditions']:
            self.assertEqual(condition['stages']['original']['status'], 'interrupted')
            self.assertEqual(condition['stages']['observed']['status'], 'skipped')
            pid = int((runner.out/condition['name']/'owned-pid').read_text())
            self.assertFalse(still_running(pid))

    def test_one_worker_keeps_conditions_sequential(self):
        runner = self.runner(workers=1)
        result = runner.run()
        first = runner.out / result['conditions'][0]['name']
        second = runner.out / result['conditions'][1]['name']
        observed = json.loads((first/'observed.fake.json').read_text())
        original = json.loads((second/'original.fake.json').read_text())
        self.assertLessEqual(observed['end'], original['start'])

    def test_failed_original_preserves_output_and_does_not_skip_observed_or_peer(self):
        def factory(c, stage, folder, success):
            if stage == 'observed':
                self.assertEqual(success, c.iteration != 1)
            code = '7' if c.iteration == 1 and stage == 'original' else '0'
            return [sys.executable, '-c', FAKE, str(folder), stage, '.02', code]
        runner = self.runner(factory=factory)
        result = runner.run()
        failed, healthy = result['conditions']
        self.assertFalse(result['all_processes_completed'])
        self.assertEqual(failed['stages']['original']['returncode'], 7)
        self.assertEqual(failed['stages']['observed']['status'], 'completed')
        self.assertEqual(healthy['status'], 'completed')
        self.assertTrue((runner.out/failed['name']/'original.fake.json').exists())
        self.assertIsNone(result['cancellation_reason'])

    def test_deadline_stops_owned_descendants_and_marks_unstarted_work(self):
        script = r'''
import json,os,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'])
Path(sys.argv[1]).write_text(json.dumps(dict(parent=os.getpid(),child=child.pid)))
time.sleep(30)
'''
        factory = lambda c, stage, folder, success: [sys.executable, '-c', script, str(folder/'pids.json')]
        runner = self.runner(workers=1, seconds=.5, factory=factory)
        result = runner.run()
        self.assertEqual(result['cancellation_reason'], 'deadline')
        self.assertLess(result['elapsed_s'], 3)
        first, second = result['conditions']
        self.assertEqual(first['stages']['original']['status'], 'interrupted')
        self.assertEqual(first['stages']['observed']['status'], 'skipped')
        self.assertEqual(second['stages']['original']['status'], 'skipped')
        pids = json.loads((runner.out/first['name']/'pids.json').read_text())
        until = time.monotonic()+1
        while any(still_running(pid) for pid in pids.values()) and time.monotonic() < until:
            time.sleep(.01)
        self.assertFalse(any(still_running(pid) for pid in pids.values()))

    def test_sigterm_cleans_owned_work_and_leaves_unrelated_process_alive(self):
        harness = Path(__file__).resolve().parents[1]
        script = r'''
import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from eval_batch import BatchRunner,Condition
command=lambda c,stage,folder,success:[sys.executable,'-c','import time; time.sleep(30)']
r=BatchRunner(conditions=[Condition(1,'ema')],worker_envs=[{**os.environ,'CUDA_VISIBLE_DEVICES':'test-device'}],out=Path(sys.argv[2]),seconds=20,command_factory=command,grace_s=.1,poll_s=.01)
result=r.run()
raise SystemExit(143 if result['cancellation_reason']=='SIGTERM' else 1)
'''
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        process = subprocess.Popen([sys.executable, '-c', script, str(harness), str(self.root/'out')],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            manifest = self.root/'out/manifest.json'
            until = time.monotonic()+5
            pid = None
            while time.monotonic() < until:
                if manifest.exists():
                    stage = json.loads(manifest.read_text())['conditions'][0]['stages']['original']
                    if stage['status'] == 'running':
                        pid = stage['pid']
                        break
                time.sleep(.01)
            self.assertIsNotNone(pid)
            process.send_signal(signal.SIGTERM)
            output, _ = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 143, output)
            self.assertFalse(still_running(pid))
            self.assertIsNone(unrelated.poll())
            final = json.loads(manifest.read_text())
            self.assertEqual(final['cancellation_reason'], 'SIGTERM')
            self.assertEqual(final['conditions'][0]['stages']['observed']['status'], 'skipped')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            unrelated.terminate()
            unrelated.wait()

    def test_internal_bookkeeping_error_cleans_descendants_and_continues_observed(self):
        script = r"""
import json,os,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'])
Path(sys.argv[1]).write_text(json.dumps(dict(parent=os.getpid(),child=child.pid)))
time.sleep(30)
"""
        def factory(c, stage, folder, success):
            if stage == 'original':
                return [sys.executable, '-c', script, str(folder/'pids.json')]
            return [sys.executable, '-c', FAKE, str(folder), stage, '.01', '0']
        runner = self.runner(workers=1, conditions=[Condition(1,'ema')], factory=factory)
        native_stage = runner.manifest.stage
        def fail_after_start(name, stage, **changes):
            native_stage(name, stage, **changes)
            if stage == 'original' and changes.get('status') == 'running':
                until = time.monotonic()+2
                while not (runner.out/name/'pids.json').exists() and time.monotonic()<until:
                    time.sleep(.01)
                raise OSError('injected bookkeeping failure after launch')
        runner.manifest.stage = fail_after_start
        result = runner.run()
        condition = result['conditions'][0]
        self.assertEqual(condition['stages']['original']['status'], 'failed')
        self.assertEqual(condition['stages']['observed']['status'], 'completed')
        pids = json.loads((runner.out/condition['name']/'pids.json').read_text())
        until = time.monotonic()+1
        while any(still_running(pid) for pid in pids.values()) and time.monotonic()<until:
            time.sleep(.01)
        self.assertFalse(any(still_running(pid) for pid in pids.values()))

    def test_worker_status_write_failure_does_not_leave_running_manifest(self):
        import eval_batch
        runner = self.runner(workers=1, conditions=[Condition(1,'ema')])
        native_write = eval_batch.atomic_json
        def fail_stage_status(path, value):
            if Path(path).name == 'original.status.json':
                raise OSError('injected status write failure')
            native_write(path, value)
        with mock.patch('eval_batch.atomic_json', side_effect=fail_stage_status):
            result = runner.run()
        self.assertEqual(result['cancellation_reason'], 'worker_error')
        statuses = result['conditions'][0]['stages']
        self.assertEqual(statuses['original']['status'], 'interrupted')
        self.assertEqual(statuses['observed']['status'], 'skipped')
        self.assertNotIn('running', [stage['status'] for stage in statuses.values()])

    def test_rejects_reused_output_and_duplicate_devices(self):
        runner = self.runner()
        marker = runner.out/'user-file'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.runner()
        self.assertEqual(marker.read_text(), 'keep')
        with self.assertRaises(ValueError):
            BatchRunner(conditions=[Condition(1,'ema')],
                        worker_envs=[{'CUDA_VISIBLE_DEVICES':'same'}]*2,
                        out=self.root/'duplicate', seconds=1, command_factory=lambda *a: [])
        self.assertFalse((self.root/'duplicate').exists())

    def test_long_output_roots_use_short_distinct_real_compiler_directories(self):
        output = self.root.joinpath(*(['long-output-' + 'x'*100] * 8))
        self.assertGreater(len(str(output)), 800)
        owned = []
        try:
            for index in range(2):
                env, persistent = worker_environment(index, f'device-{index}', output,
                                                     self.root/'source', 'declared-revision', self.root/'reference')
                compiler = Path(env['TMPDIR'])
                owned.append(compiler)
                self.assertEqual(compiler.parent, Path('/tmp'))
                self.assertTrue(compiler.name.startswith('nabla-eval-'))
                self.assertLess(len(str(compiler)), 80)
                self.assertTrue(compiler.is_dir())
                self.assertNotIn(compiler, persistent)
                self.assertTrue(all(path.is_relative_to(output) for path in persistent))
                actual = subprocess.check_output([sys.executable, '-c',
                    'import tempfile; f=tempfile.NamedTemporaryFile(); print(f.name)'], env=env, text=True).strip()
                self.assertEqual(Path(actual).parent, compiler)
            self.assertEqual(len(set(owned)), 2)
            self.assertFalse(output.exists())
        finally:
            for path in owned:
                shutil.rmtree(path)

    def test_main_removes_only_created_local_temporary_dirs_after_setup_failure(self):
        import eval_batch
        run, source, reference = [self.root/name for name in ('run', 'source', 'reference')]
        (run/'ckpt').mkdir(parents=True)
        (run/'config.json').write_text(json.dumps({'clips': ['all']}))
        (run/'ckpt/it_000001.pkl').touch()
        (source/'src/nablamimic').mkdir(parents=True)
        (source/'src/nablamimic/evaluate.py').touch()
        reference.mkdir()
        for i in range(40):
            (reference/f'clip_{i}.npz').touch()
        untouched = self.root/'unrelated-temporary'; untouched.mkdir()
        created = []
        native = tempfile.mkdtemp
        def create(*args, **kwargs):
            result = native(*args, **kwargs)
            created.append(Path(result))
            return result
        arguments = ['--run', str(run), '--source', str(source), '--source-revision', 'declared',
                     '--reference-cache', str(reference), '--iters', '1', '--workers', '1',
                     '--devices', 'assigned', '--seconds', '1', '--out', str(self.root/'fresh-output')]
        with mock.patch('eval_batch.resolve_devices', return_value=[{'device':'assigned','model':'test','memory_bytes':1}]), \
             mock.patch('eval_batch.tempfile.mkdtemp', side_effect=create), \
             mock.patch('eval_batch.BatchRunner', side_effect=RuntimeError('injected setup failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected setup failure'):
                eval_batch.main(arguments)
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())
        self.assertTrue(untouched.exists())

    def test_rejects_nonfinite_budget(self):
        with self.assertRaises(ValueError):
            self.runner(seconds=float('nan'))
        self.assertFalse((self.root/'out').exists())


if __name__ == '__main__':
    unittest.main()
