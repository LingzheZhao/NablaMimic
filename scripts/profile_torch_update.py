"""Time existing Torch training calls on one update in provisioned CUDA compute.

Pass the usual torch_train arguments after --. By default update 3 is measured,
following two ordinary updates. --kernel-trace also writes a Chrome kernel trace.
All profiling files are written inside the training command's fresh --out root.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import time

import torch

from nablamimic import torch_train


class TimedCall:
    def __init__(self, function, phase, records, kernel_trace):
        self.function = function
        self.phase = phase
        self.records = records
        self.kernel_trace = kernel_trace

    def __getattr__(self, name):
        return getattr(self.function, name)

    def __call__(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        region = (torch.profiler.record_function('update_profile/' + self.phase)
                  if self.kernel_trace else nullcontext())
        wall_start = time.perf_counter()
        start.record()
        with region:
            result = self.function(*args, **kwargs)
        end.record()
        self.records.append((self.phase, start, end, time.perf_counter() - wall_start))
        return result


def profile_update(trainer, update, kernel_trace):
    records, originals = [], []

    def wrap(owner, name, phase):
        function = owner[name] if isinstance(owner, dict) else getattr(owner, name)
        originals.append((owner, name, function))
        proxy = TimedCall(function, phase, records, kernel_trace)
        if isinstance(owner, dict):
            owner[name] = proxy
        else:
            setattr(owner, name, proxy)

    wrap(trainer, 'window_inputs', 'window_inputs_sampler_reset')
    wrap(trainer, 'collector_inputs', 'collector_inputs_sampler_reset')
    wrap(trainer, 'target_fn', 'action_targets')
    wrap(trainer, 'td', 'td_targets')
    for name, fallback in (('collector', 'collect'), ('window', 'window'),
                           ('regression', 'regression'), ('critic', 'critic_update'),
                           ('statistics', 'statistics')):
        if name in trainer.graphs:
            wrap(trainer.graphs, name, name)
        else:
            wrap(trainer, fallback, name)

    profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA])
                if kernel_trace else nullcontext())
    started = time.perf_counter()
    try:
        with profiler:
            result = update(trainer)
    finally:
        for owner, name, function in reversed(originals):
            if isinstance(owner, dict):
                owner[name] = function
            else:
                setattr(owner, name, function)
    elapsed = time.perf_counter() - started
    # Trainer.update already synchronizes after statistics. Every event recorded
    # above precedes that synchronization, so reading them adds no phase barriers.
    phases = defaultdict(list)
    for phase, start, end, wall in records:
        phases[phase].append(dict(cuda_event_ms=start.elapsed_time(end), host_call_s=wall))
    span = records[0][1].elapsed_time(records[-1][2])
    report = dict(
        timing=dict(scope='One instrumented update diagnostic with CUDA events, host dispatch and profiling overhead'),
        completed_update=trainer.completed_updates, execution=trainer.args.execution,
        kernel_trace=kernel_trace, update_wall_s=elapsed, trainer_iteration_s=result['iteration_s'],
        cuda_event_span_ms=span,
        phases={name: dict(calls=len(rows), cuda_event_ms=sum(row['cuda_event_ms'] for row in rows),
                           host_call_s=sum(row['host_call_s'] for row in rows), measurements=rows)
                for name, rows in phases.items()},
        synchronization='All phase events precede the existing Trainer.update synchronization after statistics',
        rng_and_weights='Original Trainer.update invokes each wrapped callable once with its original arguments',
    )
    output = Path(trainer.args.out)
    stem = f'update-profile-{trainer.completed_updates:06d}'
    if kernel_trace:
        trace = output / f'{stem}-trace.json'
        profiler.export_chrome_trace(str(trace))
        report['trace'] = str(trace)
    torch_train.write_json(output / f'{stem}.json', report)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-update', type=int, default=3)
    parser.add_argument('--kernel-trace', action='store_true')
    options, training_args = parser.parse_known_args(argv)
    if training_args[:1] == ['--']:
        training_args = training_args[1:]
    if options.profile_update < 3:
        parser.error('--profile-update requires at least two completed warmup updates')
    original = torch_train.Trainer.update
    measured = False

    def update(trainer):
        nonlocal measured
        if trainer.completed_updates + 1 != options.profile_update:
            return original(trainer)
        result = profile_update(trainer, original, options.kernel_trace)
        measured = True
        return result

    torch_train.Trainer.update = update
    try:
        torch_train.main(training_args)
    finally:
        torch_train.Trainer.update = original
    if not measured:
        raise RuntimeError(f'Training did not execute profiling update {options.profile_update}')


if __name__ == '__main__':
    main()
