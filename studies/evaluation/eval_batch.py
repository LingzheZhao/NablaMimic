"""Bounded independent nominal evaluations with explicit workers per assigned GPU.

Each condition owns fresh original and observed processes, in that order. Source,
checkpoints and reference data are read-only inputs. Persistent writable state is
private under the new output root; compiler temporaries use short owned /tmp paths. This module itself never imports the simulator or JAX.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import tempfile
import time


ENV_RECORD = (
    'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONNOUSERSITE', 'NABLA_SOURCE_DIR',
    'NABLA_SOURCE_REVISION', 'NABLA_VERIFIED_IMAGE_REFERENCE', 'MENAGERIE_DIR',
    'LAFAN1_G1_DIR', 'MOTION_CACHE_DIR', 'JAX_COMPILATION_CACHE_DIR',
    'WARP_CACHE_PATH', 'CUDA_CACHE_PATH', 'DK_CACHE_DIR', 'G1_FAST_GEN_DIR',
    'XDG_CACHE_HOME', 'TMPDIR', 'PYTHONPYCACHEPREFIX', 'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_CLIENT_MEM_FRACTION',
    'XLA_PYTHON_CLIENT_PREALLOCATE',
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


@dataclass(frozen=True)
class Condition:
    iteration: int
    policy: str

    @property
    def name(self):
        return f'it{self.iteration:06d}-{self.policy}'


class Manifest:
    """All concurrent mutations and atomic replacements use this one lock."""
    def __init__(self, root, conditions, workers, provenance):
        self.path = root / 'manifest.json'
        self.lock = threading.Lock()
        self.indices = {condition.name: i for i, condition in enumerate(conditions)}
        self.data = dict(status='running', started_utc=utc_now(), provenance=provenance,
                         workers=workers, conditions=[dict(
                             name=c.name, iteration=c.iteration, policy=c.policy,
                             worker=None, device=None, status='pending',
                             stages={stage: dict(status='pending') for stage in ('original', 'observed')}
                         ) for c in conditions])
        atomic_json(self.path, self.data)

    def condition(self, name, **changes):
        with self.lock:
            self.data['conditions'][self.indices[name]].update(changes)
            atomic_json(self.path, self.data)

    def stage(self, name, stage, **changes):
        with self.lock:
            self.data['conditions'][self.indices[name]]['stages'][stage].update(changes)
            atomic_json(self.path, self.data)

    def finish(self, reason, elapsed_s):
        with self.lock:
            for condition in self.data['conditions']:
                for stage in condition['stages'].values():
                    if stage['status'] == 'pending':
                        stage.update(status='skipped', reason=reason or 'worker_failed')
                    elif stage['status'] == 'running':
                        stage.update(status='interrupted', reason=reason or 'worker_failed_before_status_write')
                stages = [stage['status'] for stage in condition['stages'].values()]
                condition['status'] = 'completed' if stages == ['completed', 'completed'] else 'partial_or_failed'
            complete = all(c['status'] == 'completed' for c in self.data['conditions'])
            self.data.update(status='completed' if complete else 'partial_or_failed',
                             cancellation_reason=reason, elapsed_s=elapsed_s,
                             completed_utc=utc_now(), all_processes_completed=complete)
            atomic_json(self.path, self.data)
        return self.data


class Processes:
    """Own only groups created here with start_new_session; never touch peers."""
    def __init__(self, grace_s):
        self.lock = threading.Lock()
        self.active = {}
        self.cancelled = threading.Event()
        self.reason = None
        self.grace_s = grace_s

    def request_cancel(self, reason):
        # Signal handlers call only this: no lock, waiting, filesystem or kill.
        if not self.cancelled.is_set():
            self.reason = reason
            self.cancelled.set()

    def launch(self, command, log, env):
        with self.lock:
            if self.cancelled.is_set():
                return None
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            self.active[process.pid] = dict(process=process, term_at=None)
            return process

    @staticmethod
    def _signal(pid, sig):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass

    def stop_owned(self):
        if not self.cancelled.is_set():
            return
        now = time.monotonic()
        with self.lock:
            for pid, owned in self.active.items():
                if owned['term_at'] is None:
                    self._signal(pid, signal.SIGTERM)
                    owned['term_at'] = now
                elif now - owned['term_at'] >= self.grace_s:
                    self._signal(pid, signal.SIGKILL)

    def terminate_one(self, process):
        # Internal errors need the same descendant cleanup as external cancellation.
        # Mark this owned group before TERM, even if its leader has already exited.
        with self.lock:
            self.active[process.pid]['term_at'] = time.monotonic()
            self._signal(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=self.grace_s)
        except subprocess.TimeoutExpired:
            self._signal(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            self._signal(process.pid, signal.SIGKILL)

    def release(self, process):
        with self.lock:
            owned = self.active.pop(process.pid)
            # The leader may exit before a child. A cancelled group remains ours;
            # finish its cleanup even after the leader has been reaped.
            if owned['term_at'] is not None or self.cancelled.is_set():
                self._signal(process.pid, signal.SIGKILL)


class BatchRunner:
    """CPU-testable scheduler; command_factory supplies real or test workloads."""
    def __init__(self, *, conditions, worker_envs, out, seconds, command_factory,
                 provenance=None, grace_s=5.0, poll_s=0.1, started=None, workers_per_gpu=1):
        if not conditions or len({c.name for c in conditions}) != len(conditions):
            raise ValueError('Conditions must be nonempty and unique')
        if not isinstance(workers_per_gpu, int) or isinstance(workers_per_gpu, bool) or workers_per_gpu < 1:
            raise ValueError('workers_per_gpu must be a positive integer')
        worker_envs = [dict(env) for env in worker_envs]
        devices = [env.get('CUDA_VISIBLE_DEVICES') for env in worker_envs]
        if not devices or any(not d or ',' in d for d in devices):
            raise ValueError('Each worker requires exactly one explicitly assigned GPU')
        device_counts = Counter(devices)
        if any(count != workers_per_gpu for count in device_counts.values()):
            raise ValueError('Worker sharing must match explicit workers_per_gpu exactly; default1 requires distinct GPUs')
        memory_limits = {device: 0.0 for device in device_counts}
        for env, device in zip(worker_envs, devices):
            fraction = float(env.get('XLA_CLIENT_MEM_FRACTION', .9/workers_per_gpu))
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError('Each worker JAX memory fraction must be finite and in (0,1]')
            env['XLA_CLIENT_MEM_FRACTION'] = str(fraction)
            env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            memory_limits[device] += fraction
        if any(total > 1.0 + 1e-12 for total in memory_limits.values()):
            raise ValueError('Summed worker JAX memory fractions may not exceed1 per GPU')
        if not all(math.isfinite(value) and value > 0 for value in (seconds, grace_s, poll_s)):
            raise ValueError('Budgets and polling/termination intervals must be positive')
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic() if started is None else started
        self.deadline = self.started + seconds
        self.conditions = conditions
        self.worker_envs = worker_envs
        self.command_factory = command_factory
        self.poll_s = poll_s
        self.processes = Processes(grace_s)
        workers = [dict(worker=i, device=devices[i], environment={k: env.get(k) for k in ENV_RECORD})
                   for i, env in enumerate(worker_envs)]
        self.manifest = Manifest(self.out, conditions, workers, {
            **(provenance or {}), 'seconds_budget': seconds,
            'termination_grace_s': grace_s, 'worker_count': len(workers),
            'workers_per_gpu': workers_per_gpu, 'gpu_count': len(device_counts),
            'jax_memory_fraction_limits_per_gpu': memory_limits,
            'memory_limit_scope': 'JAX allocator limits only; Warp and driver allocations are additional',
            'condition_order': [c.name for c in conditions],
            'assignment': 'Conditions round-robin over workers; each GPU has exactly the explicitly declared worker count',
            'physics_instances': 'Fresh process and physics for each stage',
        })

    def _check_deadline(self):
        if time.monotonic() >= self.deadline:
            self.processes.request_cancel('deadline')

    def _stage(self, condition, stage, folder, env, original_success):
        self._check_deadline()
        if self.processes.cancelled.is_set():
            self.manifest.stage(condition.name, stage, status='skipped', reason=self.processes.reason)
            return False
        start = time.monotonic()
        process = None
        command = None
        result = {}
        try:
            command = list(self.command_factory(condition, stage, folder, original_success))
            stage_env = {**env, 'NABLA_ENGINE_CACHE_AUDIT': str(folder / f'{stage}.engine-cache.json')}
            with (folder / f'{stage}.log').open('x') as log:
                process = self.processes.launch(command, log, stage_env)
                if process is None:
                    result = dict(status='skipped', reason=self.processes.reason)
                else:
                    self.manifest.stage(condition.name, stage, status='running', command=command,
                                        pid=process.pid, started_utc=utc_now(),
                                        start_elapsed_s=start-self.started)
                    while process.poll() is None:
                        self._check_deadline()
                        self.processes.stop_owned()
                        time.sleep(self.poll_s)
                    code = process.wait()
                    result = dict(status='completed' if code == 0 else (
                        'interrupted' if self.processes.cancelled.is_set() else 'failed'),
                        returncode=code, pid=process.pid,
                        cancellation_reason=self.processes.reason if self.processes.cancelled.is_set() else None)
        except Exception as exc:
            result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
            # Failures of a worker's bookkeeping must not leave its subprocess.
            if process is not None:
                self.processes.terminate_one(process)
        finally:
            if process is not None:
                self.processes.release(process)
        result.update(elapsed_s=time.monotonic()-start, completed_utc=utc_now(), command=command)
        atomic_json(folder / f'{stage}.status.json', result)
        self.manifest.stage(condition.name, stage, **result)
        return result['status'] == 'completed'

    def _worker(self, worker_id):
        env = self.worker_envs[worker_id]
        for condition in self.conditions[worker_id::len(self.worker_envs)]:
            self._check_deadline()
            if self.processes.cancelled.is_set():
                break
            folder = self.out / condition.name
            folder.mkdir()
            self.manifest.condition(condition.name, status='running', worker=worker_id,
                                    device=env['CUDA_VISIBLE_DEVICES'], output=str(folder))
            first = self._stage(condition, 'original', folder, env, False)
            second = self._stage(condition, 'observed', folder, env, first)
            self.manifest.condition(condition.name, status='completed' if first and second else 'partial_or_failed')

    def run(self):
        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            def request(sig, _frame):
                self.processes.request_cancel(signal.Signals(sig).name)
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, request)
        try:
            with ThreadPoolExecutor(max_workers=len(self.worker_envs)) as executor:
                futures = [executor.submit(self._worker, i) for i in range(len(self.worker_envs))]
                pending = set(futures)
                while pending:
                    self._check_deadline()
                    self.processes.stop_owned()
                    done, pending = wait(pending, timeout=self.poll_s)
                    for future in done:
                        error = future.exception()
                        if error is not None:
                            self.processes.request_cancel('worker_error')
                            atomic_json(self.out / 'worker_error.json', dict(error=repr(error)))
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return self.manifest.finish(self.processes.reason, time.monotonic()-self.started)


def resolve_devices(tokens):
    """Canonical UUIDs also reject two spellings of the same physical device."""
    devices = []
    for token in tokens:
        if not token or ',' in token or token == '-1':
            raise ValueError('Pass one explicitly assigned GPU per --devices item')
        result = subprocess.run(['nvidia-smi', '--id', token, '--query-gpu=uuid,name,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                check=True, timeout=20)
        rows = list(csv.reader(io.StringIO(result.stdout.strip())))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError(f'Cannot resolve exactly one GPU for {token!r}')
        uuid, model, memory = [x.strip() for x in rows[0]]
        devices.append(dict(device=uuid, model=model, memory_bytes=int(memory)*2**20))
    if len({d['device'] for d in devices}) != len(devices):
        raise ValueError('The device list contains multiple aliases of one physical GPU; use explicit workers_per_gpu for sharing')
    return devices


def worker_environment(index, device, out, source, source_revision, reference_cache, memory_fraction=.9):
    root = out / 'workers' / f'worker-{index}'
    # Main creates persistent paths after the runner reserves its fresh output
    # root. Compiler temporaries are separately owned short local directories.
    env = dict(os.environ)
    env.pop('JAX_PLATFORMS', None)
    env.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)  # One unambiguous allocator setting.
    env.update(CUDA_VISIBLE_DEVICES=device, PYTHONPATH=str(source/'src'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               NABLA_SOURCE_DIR=str(source), NABLA_SOURCE_REVISION=source_revision,
               MOTION_CACHE_DIR=str(reference_cache), XLA_CLIENT_MEM_FRACTION=str(memory_fraction),
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='8',
               OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    paths = dict(XDG_CACHE_HOME='cache/xdg', DK_CACHE_DIR='cache/engine',
                 JAX_COMPILATION_CACHE_DIR='cache/jax', G1_FAST_GEN_DIR='cache/engine-g1',
                 CUDA_CACHE_PATH='cache/cuda', WARP_CACHE_PATH='cache/warp',
                 PYTHONPYCACHEPREFIX='cache/pycache')
    env.update({key: str(root/value) for key, value in paths.items()})
    # NVRTC-generated compiler paths must remain short even when the persistent
    # shared output path is long. Main owns and removes only this fresh directory.
    env['TMPDIR'] = tempfile.mkdtemp(prefix='nabla-eval-', dir='/tmp')
    return env, [root/value for value in paths.values()]


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--iters', type=int, nargs='+', required=True)
    parser.add_argument('--policies', choices=('ema', 'live'), nargs='+', default=['ema'])
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--workers-per-gpu', type=int, default=1,
                        help='Explicit simultaneous worker count per physical GPU; default1 forbids sharing')
    parser.add_argument('--memory-fraction', type=float,
                        help='JAX allocator fraction per worker; default0.90/workers-per-gpu, preallocation always false')
    parser.add_argument('--devices', nargs='+', required=True,
                        help='Distinct assigned GPU indices or UUIDs; first workers/workers-per-gpu entries are used')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--source', type=Path, default=os.environ.get('NABLA_SOURCE_DIR'))
    parser.add_argument('--source-revision', default=os.environ.get('NABLA_SOURCE_REVISION'))
    parser.add_argument('--reference-cache', type=Path, default=os.environ.get('MOTION_CACHE_DIR'))
    parser.add_argument('--expected-gpu-model', help='Optional exact model restriction, checked on every assigned worker')
    parser.add_argument('--forward-only', action='store_true', help='Use evaluator forward-only initialization; preserve forward rollout mathematics')
    args = parser.parse_args(argv)
    if (args.workers < 1 or args.workers_per_gpu < 1 or args.workers % args.workers_per_gpu
            or args.workers // args.workers_per_gpu > len(args.devices)):
        parser.error('--workers must be a positive multiple of --workers-per-gpu and fit the explicitly supplied devices')
    memory_fraction = .9/args.workers_per_gpu if args.memory_fraction is None else args.memory_fraction
    if (not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1
            or memory_fraction*args.workers_per_gpu > 1.0+1e-12):
        parser.error('Per-worker JAX memory fraction must be positive and sum to at most1 on each GPU')
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not args.iters or min(args.iters) < 0:
        parser.error('Positive time budget and nonnegative checkpoint iterations required')
    if len(set(args.iters)) != len(args.iters) or len(set(args.policies)) != len(args.policies) or 'ema' not in args.policies:
        parser.error('Iterations/policies must be unique, and policies must include ema')
    if not args.source or not args.source_revision or not args.reference_cache:
        parser.error('Explicit frozen source, declared revision and reference cache required')
    run, source, reference, out = (p.resolve() for p in (args.run, args.source, args.reference_cache, args.out))
    if out.exists():
        parser.error(f'Output already exists: {out}')
    harness = Path(__file__).resolve().parent
    input_roots = [run, source, reference, harness]
    input_roots += [Path(os.environ[key]).resolve() for key in ('MENAGERIE_DIR','LAFAN1_G1_DIR') if os.environ.get(key)]
    if any(out.is_relative_to(path) for path in input_roots):
        parser.error('Output must be outside every frozen input/source directory')
    for path in [run/'config.json', source/'src/nablamimic/evaluate.py',
                 *[run/'ckpt'/f'it_{it:06d}.pkl' for it in args.iters]]:
        if not path.is_file():
            parser.error(f'Missing frozen input: {path}')
    if not reference.is_dir() or len(list(reference.glob('*.npz'))) != 40:
        parser.error('Reference cache must contain the fixed forty NPZ inputs')
    cfg = json.loads((run/'config.json').read_text())
    if cfg['clips'] != ['all']:
        parser.error('This evaluation batch requires the fixed all-forty-clip run')
    # Resolve every supplied token before deliberate repetition, so aliases never
    # accidentally enable device sharing. Only declared workers_per_gpu can do so.
    devices = resolve_devices(args.devices)[:args.workers // args.workers_per_gpu]
    worker_devices = [device for device in devices for _ in range(args.workers_per_gpu)]
    if args.expected_gpu_model and any(d['model'] != args.expected_gpu_model for d in devices):
        parser.error('Assigned GPU model differs from --expected-gpu-model')
    conditions = [Condition(it, policy) for it in sorted(args.iters) for policy in args.policies]
    for path in [harness/'bootstrap.py', harness/'instrumentation/observed_evaluate.py']:
        if not path.is_file():
            parser.error(f'Missing frozen harness input: {path}')
    owned_temporary_dirs = []
    try:
        envs, directories = [], []
        for index, device in enumerate(worker_devices):
            env, paths = worker_environment(index, device['device'], out, source, args.source_revision, reference,
                                            memory_fraction=memory_fraction)
            owned_temporary_dirs.append(Path(env['TMPDIR']))
            envs.append(env)
            directories.extend(paths)

        def command_factory(condition, stage, folder, original_success):
            native = ['--run', str(run), '--iters', str(condition.iteration), '--policy', condition.policy,
                      '--solver-iterations', '10', '--out', str(folder/f'{stage}.json')]
            if args.forward_only:
                native.append('--forward-only')
            if stage == 'original':
                return [sys.executable, str(harness/'bootstrap.py'), '--module', 'nablamimic.evaluate', *native]
            comparison = ['--original-json', str(folder/'original.json')] if original_success else []
            return [sys.executable, str(harness/'bootstrap.py'), '--script',
                    str(harness/'instrumentation/observed_evaluate.py'), '--audit-root', str(folder/'observed-traces'),
                    *comparison, '--', *native]

        runner = BatchRunner(conditions=conditions, worker_envs=envs, out=out, seconds=args.seconds,
                             command_factory=command_factory, started=started, workers_per_gpu=args.workers_per_gpu, provenance=dict(
                                 argv=sys.argv if argv is None else argv, python=sys.executable,
                                 source_root=str(source), source_revision=args.source_revision,
                                 harness_root=str(harness), harness_revision=os.environ.get('NABLA_EVALUATION_REVISION'),
                                 image=os.environ.get('NABLA_VERIFIED_IMAGE_REFERENCE'), run=str(run),
                                 reference_cache=str(reference), devices=devices,
                                 initialization='forward_only' if args.forward_only else 'original',
                                 cache_condition='Fresh private persistent caches per worker; short private /tmp compiler directories; no shared writable cache',
                                 mathematical_condition='Unchanged nominal donor scalar/observed rollout, ten solver iterations'))
        for path in directories:
            path.mkdir(parents=True, exist_ok=False)
        result = runner.run()
        print(json.dumps(dict(status=result['status'], elapsed_s=result['elapsed_s'],
                              cancellation_reason=result['cancellation_reason'], output=str(out))), flush=True)
        if result['all_processes_completed']:
            return 0
        return {'deadline': 124, 'SIGINT': 130, 'SIGTERM': 143}.get(result['cancellation_reason'], 1)

    finally:
        # run() has joined workers and reaped owned processes before this cleanup.
        # The list contains only paths returned by our own mkdtemp calls.
        for path in owned_temporary_dirs:
            if path.exists():
                shutil.rmtree(path)


if __name__ == '__main__':
    raise SystemExit(main())
