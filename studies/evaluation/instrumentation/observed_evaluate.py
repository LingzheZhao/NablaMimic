"""Independent observed-execution protocol after strict donor parity failed.
Own reward/completion/world metrics come from ONE actual observed rollout.
Optional original JSON supplies descriptive cross-condition differences only.
No original result lends validity, no trajectories/scalars are stitched, and
historical failed parity margins remain unchanged.
"""
import argparse
import json
from pathlib import Path

from observed_helpers import (PROTOCOL, augmented_rollout_factory, clean_json,
                              compare_conditions, write_observation)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zero-residual',action='store_true')
    ap.add_argument('--writer-workers',type=int,choices=(1,2),default=1)
    ap.add_argument('--audit-root', type=Path, required=True)
    ap.add_argument('--original-json',type=Path)
    ap.add_argument('native', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    native = args.native[1:] if args.native[:1] == ['--'] else args.native
    if '--out' not in native or '--run' not in native:
        ap.error('explicit --run and fresh --out required')
    output = Path(native[native.index('--out') + 1])
    run = Path(native[native.index('--run') + 1])
    if output.exists():
        ap.error('output already exists')
    cfg = json.loads((run / 'config.json').read_text())
    if cfg['clips'] != ['all']:
        ap.error('this protocol requires clips=all')
    args.audit_root.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nablamimic import evaluate as ev
    from mujoco.mjx._src import math as mjm
    native_args=ev.build_parser().parse_args(native)
    requested=sorted(set(native_args.iters))
    policy=native_args.policy or ('ema' if cfg.get('policy_ema',False) else 'live')
    if len(requested)!=1 or native_args.persistent:
        raise ValueError('Native overflow flags are sticky: use one checkpoint per fresh process/physics instance')
    if args.zero_residual:
        if not cfg.get('residual_action',False):
            raise ValueError('zero-residual requires a residual-action checkpoint')
        ev.actor_apply=lambda actor,hist,goal:jnp.zeros((hist.shape[0],29),dtype=hist.dtype)
    from evaluation_capacity import install
    capacity_audits=install(ev.sim)
    names = ev.all_clip_names()
    if len(names) != 40:
        raise ValueError('exactly40 matching clips required')

    augmented = augmented_rollout_factory(ev)

    calls = 0
    (args.audit_root / 'expected_clips.json').write_text(json.dumps(dict(
        clips=[dict(clip=n,status='pending_no_trajectory_recorded') for n in names],
        limitation='If compilation or a process interruption prevents a trace return, executed support is unknown; do not infer zero executed steps.'),indent=2))
    def make(env,steps):
        nonlocal calls
        if len(env.starts) != 40:
            raise ValueError('expected40 evaluation worlds')
        fn = augmented(env,steps)
        def call(actor,frames,key,tables):
            nonlocal calls
            calls += 1
            checkpoint_iter=requested[calls-1]
            if len(capacity_audits)!=1:
                raise RuntimeError('expected exactly one native physics instance')
            capacity=capacity_audits[0]
            compiled=fn.lower(actor,frames,key,tables).compile()
            capacity.reset()
            res,trace = jax.device_get(compiled(actor,frames,key,tables))
            write_observation(env, names, res, trace, capacity, args.audit_root,
                              checkpoint_iter, policy, zero_residual=args.zero_residual, writer_workers=args.writer_workers)
            return res
        return call
    ev.make_rollout=make
    try:
        ev.main(native)
        observed=json.loads(output.read_text())
        observed.update(protocol=PROTOCOL,execution_condition='observed',policy=policy,control='zero_residual' if args.zero_residual else 'learned')
        output.write_text(json.dumps(clean_json(observed),allow_nan=False))
        comparison=compare_conditions(observed,args.original_json,run,requested[0],names,policy)
        (args.audit_root / 'cross_condition_differences.json').write_text(json.dumps(clean_json(comparison),indent=2,allow_nan=False))
    except BaseException as exc:
        (args.audit_root / 'FAILED.json').write_text(json.dumps(dict(error=repr(exc),accepted=False,per_clip=[dict(clip=n,status='see_existing_call_trace_or_execution_unknown') for n in names]),indent=2))
        raise

if __name__ == '__main__':
    main()
