"""Original nominal evaluations sharing one explicitly reset evaluation session."""
import argparse
import json
import os
from pathlib import Path
import time


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--iters', type=int, nargs='+', required=True)
    parser.add_argument('--policy', choices=('ema', 'live'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--forward-only', action='store_true')
    args = parser.parse_args()
    root = args.out.resolve()
    inputs = [args.run.resolve(), Path(__file__).resolve().parent]
    inputs += [Path(os.environ[k]).resolve() for k in (
        'NABLA_SOURCE_DIR', 'MENAGERIE_DIR', 'MOTION_CACHE_DIR', 'LAFAN1_G1_DIR') if os.environ.get(k)]
    if root.exists() or any(root.is_relative_to(p) for p in inputs):
        parser.error('Fresh output outside frozen inputs required')
    if any(i < 0 or not (args.run/'ckpt'/f'it_{i:06d}.pkl').is_file() for i in args.iters):
        parser.error('Every explicitly requested numbered checkpoint must exist')
    root.mkdir(parents=True)
    rows = [dict(occurrence=n, iteration=it, policy=args.policy, status='pending',
                 output=str(root/f'occurrence{n:03d}-it{it:06d}-{args.policy}'))
            for n, it in enumerate(args.iters)]
    manifest = dict(run=str(args.run), persistent=True, forward_only=args.forward_only,
                    conditions=rows, status='running', condition_order=args.iters)
    write(root/'manifest.json', manifest)
    started = time.perf_counter()
    current = None
    try:
        from nablamimic import evaluate as ev
        session = ev.EvaluationSession(args.run, forward_only=args.forward_only, persistent=True)
        if len(session.names) != 40:
            raise ValueError('Expected the complete forty-clip condition')
        manifest['initialization_s'] = session.initialization_s
        for current in rows:
            target = Path(current['output'])
            target.mkdir()
            current['status'] = 'running'
            write(root/'manifest.json', manifest)
            rec, boundary, phases = session.evaluate_checkpoint(current['iteration'], args.policy)
            output = dict(run=str(session.run), clips=session.names, terms=list(ev.TERMS),
                          solver_iterations=10, forward_only=args.forward_only)
            output[f"it{current['iteration']}/nominal/{args.policy}"] = rec
            write(target/'original.json', output)
            write(target/'original.boundary.json', boundary)
            write(target/'original.timings.json', phases)
            current['status'] = 'completed'
            write(root/'manifest.json', manifest)
        manifest['status'] = 'completed'
    except BaseException as error:
        if current is not None:
            current.update(status='failed', error=repr(error))
        for row in rows:
            if row['status'] == 'pending':
                row.update(status='skipped', reason='prior_condition_or_initialization_failed')
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['elapsed_s'] = time.perf_counter() - started
        manifest['acceptance'] = 'Process completion is not physics validity, fidelity or equivalence'
        write(root/'manifest.json', manifest)


if __name__ == '__main__':
    main()
