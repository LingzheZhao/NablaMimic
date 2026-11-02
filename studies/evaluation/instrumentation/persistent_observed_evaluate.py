"""Observed checkpoint sequence with one physics instance and explicit boundaries.

Example: --run RUN --iters 1500 12221 1500 --policy ema --out NEW_ROOT --forward-only
Repeated checkpoints retain separate occurrence directories. Original-donor
executions must run in a separate process and physics instance.
"""
import argparse
import json
import os
from pathlib import Path
import time

from observed_helpers import (augmented_rollout_factory, evaluate_condition,
                              prepare_trace_root, write_failure)


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--iters', nargs='+', type=int, required=True)
    ap.add_argument('--policy', choices=('ema', 'live'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--forward-only', action='store_true')
    ap.add_argument('--writer-workers', type=int, choices=(1,2), default=1)
    args = ap.parse_args()
    if any(it < 0 for it in args.iters):
        ap.error('Checkpoint iterations must be nonnegative')
    if args.out.exists():
        ap.error('A fresh output root is required')
    output_root = args.out.resolve()
    input_roots = [args.run.resolve(), Path(__file__).resolve().parent]
    input_roots.extend(Path(os.environ[key]).resolve() for key in (
        'NABLA_SOURCE_DIR', 'MENAGERIE_DIR', 'LAFAN1_G1_DIR', 'MOTION_CACHE_DIR') if os.environ.get(key))
    if any(output_root.is_relative_to(path) for path in input_roots):
        ap.error('Output must be outside every frozen input and source directory')
    cfg = json.loads((args.run / 'config.json').read_text())
    if cfg['clips'] != ['all']:
        ap.error('This protocol requires the complete 40-clip input')
    for iteration in args.iters:
        if not (args.run / 'ckpt' / f'it_{iteration:06d}.pkl').is_file():
            ap.error(f'Missing checkpoint {iteration}')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = [dict(occurrence=i, iteration=iteration, policy=args.policy, status='pending',
                 output=str(args.out / f'occurrence{i:03d}-it{iteration:06d}-{args.policy}'))
            for i, iteration in enumerate(args.iters)]
    manifest = dict(run=str(args.run), persistent=True, forward_only=args.forward_only,
                    condition_order=args.iters, conditions=rows, status='running',
                    separation='Observed only; original executions use separate processes and physics',
                    boundary_rule='Restore native initial snapshot and observer cursor between complete conditions only')
    write(args.out / 'manifest.json', manifest)
    names = []
    current = None
    try:
        from nablamimic import evaluate as ev
        from evaluation_capacity import install
        names = ev.all_clip_names()
        if len(names) != 40:
            raise ValueError('Expected all 40 reference clip identities')
        for row in rows:
            target = Path(row['output'])
            target.mkdir(exist_ok=False)
            prepare_trace_root(target / 'observed-traces', names)
        audits = install(ev.sim)
        session = ev.EvaluationSession(args.run, solver_iterations=10, forward_only=args.forward_only,
                                       persistent=True, rollout_factory=augmented_rollout_factory(ev))
        names = session.names
        if len(names) != 40 or len(audits) != 1 or len(session.env.starts) != 40:
            raise ValueError('Expected one physics instance containing all 40 clips')
        manifest['initialization_s'] = session.initialization_s
        for current in rows:
            target = Path(current['output'])
            trace_root = target / 'observed-traces'
            current['status'] = 'running'
            write(args.out / 'manifest.json', manifest)
            _, report = evaluate_condition(session, audits[0], current['iteration'], args.policy,
                                           target / 'observed.json', trace_root, writer_workers=args.writer_workers)
            current.update(status='completed', physics_validity=report['physics_validity'],
                           engineering_gate_accepted=report['accepted'],
                           complete_clip_count=report['complete_clip_count'])
            # A failed physical condition remains failed. The next independent
            # condition can run only through the session's complete native reset.
            write(args.out / 'manifest.json', manifest)
        manifest['status'] = 'completed'
    except BaseException as exc:
        if current is not None:
            current.update(status='failed', error=repr(exc))
            trace_root = Path(current['output']) / 'observed-traces'
            if trace_root.is_dir():
                write_failure(trace_root, names, exc)
        for row in rows:
            if row['status'] == 'pending':
                row.update(status='skipped', reason='prior_condition_or_initialization_failed')
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        manifest['elapsed_s'] = time.perf_counter() - started
        manifest['acceptance'] = 'Process completion is not physical validity, fidelity or numerical equivalence'
        write(args.out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
