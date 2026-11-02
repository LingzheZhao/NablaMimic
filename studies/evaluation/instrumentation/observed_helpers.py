"""Shared observed rollout mathematics and per-condition evidence writers.

Both fresh and persistent workers call these definitions. Native state restoration
belongs to EvaluationSession; these helpers never reset a physical trajectory.
"""
import inspect
import json
import math
import textwrap
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROTOCOL='nabla-observed-execution-v1'


def clean_json(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean_json(v) for v in value]
    return value


def nonfinite_paths(value,path=''):
    if isinstance(value,float) and not math.isfinite(value):
        return [path]
    if isinstance(value,dict):
        return [p for k,v in value.items() for p in nonfinite_paths(v,f'{path}/{k}')]
    if isinstance(value,(list,tuple)):
        return [p for i,v in enumerate(value) for p in nonfinite_paths(v,f'{path}/{i}')]
    return []


def apply_gate(report):
    # Explicit NEW practical threshold; never applied to an original donor run.
    thresholds=dict(p95_world_mpjpe_m=.30,p95_torso_xy_m=.30,
                    p95_abs_torso_yaw_rad=math.radians(20),mean_aligned_mpjpe_m=.10,
                    required_qualifying_clips=36,required_completions=40)
    for row in report['per_clip']:
        available=(row['completion'] and row['planned_steps']>0 and row['physics_status']=='no_observed_capacity_loss'
                   and row['valid_metric_steps']==row['planned_steps'])
        row['world_gate_qualifies']=bool(available and
            row['p95']['world_body_mpjpe_m']<=.30 and row['p95']['torso_xy_m']<=.30 and
            row['p95']['torso_abs_yaw_rad']<=math.radians(20) and row['means']['donor_aligned_body_mpjpe_m']<=.10)
        row['same_execution_donor_scalar']=next(x for x in report['same_execution_scalars']['per_clip'] if x['clip']==row['clip'])
    invalid=sum(r['invalid_metric_steps'] for r in report['per_clip'])
    scalar_nonfinite=nonfinite_paths(report['same_execution_scalars'])
    report['same_execution_scientific_scalars_finite']=not scalar_nonfinite
    report['nonfinite_scientific_scalar_paths']=scalar_nonfinite
    report['physics_validity']='no_observed_invalid_active_states_or_capacity_loss' if invalid==0 else 'invalid'
    report['engineering_gate']=dict(protocol=PROTOCOL,thresholds=thresholds,clip_denominator=40,
        raw_donor_completions=report['complete_clip_count'],qualifying_clips=sum(r['world_gate_qualifies'] for r in report['per_clip']),
        invalid_active_steps=invalid,same_execution_scientific_scalars_finite=not scalar_nonfinite)
    report['accepted']=bool(not scalar_nonfinite and invalid==0 and report['complete_clip_count']==40 and report['engineering_gate']['qualifying_clips']>=36)
    report['engineering_gate']['accepted']=report['accepted']


def compare_conditions(observed,original_path,run,it,names,policy):
    own_rows={r['clip']:r for r in observed.get(f'it{it}/nominal/{policy}',{}).get('per_clip',[])}
    out=dict(protocol=PROTOCOL,original_json=str(original_path) if original_path is not None else None,comparison='observed minus original',equivalence_claim=False,causality_claim=False,
             validity_transfer=False,original_physics_validity='unverified_unobserved_execution',
             support_note='Per-alive-step supports may differ; all clip termination/alive support is published. No matched-prefix trajectory comparison is claimed.',checkpoint_iter=it,observed_condition='observed',original_condition='original_donor',
             per_clip=[dict(clip=n,status='original_unavailable',observed=own_rows.get(n),original=None) for n in names])
    try:
        if original_path is None:
            raise ValueError('No independent original result supplied')
        original=json.loads(original_path.read_text())
        if Path(original['run']).resolve()!=run.resolve() or original['clips']!=names:
            raise ValueError('Original identity/clip mismatch; descriptive comparison unavailable')
        if original.get('control','learned')!=observed['control']:
            raise ValueError('Original control type mismatch')
        key=f'it{it}/nominal/{policy}';a=observed[key];b=original[key]
        if a['iter']!=it or b['iter']!=it:
            raise ValueError('Checkpoint iteration mismatch')
        def difference(x,y):
            return float(x-y) if isinstance(x,(int,float)) and isinstance(y,(int,float)) and math.isfinite(x) and math.isfinite(y) else None
        out['status']='descriptive_only'
        out['aggregate']={f:dict(observed=a[f],original=b[f],difference=difference(a[f],b[f])) for f in ('full_norm_return','alive_frac','failed_frac','return_per_alive_step')}
        rows_a={r['clip']:r for r in a['per_clip']};rows_b={r['clip']:r for r in b['per_clip']}
        rows=[]
        for name in names:
            x,y=rows_a.get(name),rows_b.get(name)
            if x is None or y is None:
                rows.append(dict(clip=name,status='row_unavailable',observed=x,original=y));continue
            cx=x['fall_s'] is None and x['alive_frac']==1.0
            cy=y['fall_s'] is None and y['alive_frac']==1.0
            rows.append(dict(clip=name,status='descriptive_only',observed=x,original=y,
                observed_donor_complete=cx,original_donor_complete=cy,donor_completion_difference=int(cx)-int(cy),
                return_difference=difference(x['return'],y['return']),alive_fraction_difference=difference(x['alive_frac'],y['alive_frac'])))
        out['per_clip']=rows
        out['donor_completion_counts']=dict(observed=sum(r.get('observed_donor_complete',False) for r in rows),original=sum(r.get('original_donor_complete',False) for r in rows),denominator=40)
        out['scope']='Separate executions. Own scalars stay attached to own trajectories; observed physics validity never validates original execution.'
    except Exception as exc:
        out.update(status='comparison_unavailable',reason=repr(exc))
    return out



def make_record():
    import jax
    import jax.numpy as jnp
    from mujoco.mjx._src import math as mjm

    def record(e, s, active, terminated, prev, reward_terms):
        source_frame=prev.frame
        pos, quat, body_lin, body_ang, aligned, _, ref_body_lin, _, torso, torso_q, ref_torso = jax.vmap(e._track)(s.dx, s.frame)
        ref_pos = e.ref_body_pos[s.frame]
        ref_q = e.ref_qpos[s.frame]
        ref_torso_q = e.ref_body_quat[s.frame, e.anchor_k]
        yaw = lambda q: jnp.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2))
        dyaw = yaw(torso_q) - yaw(ref_torso_q)
        dyaw = jnp.arctan2(jnp.sin(dyaw), jnp.cos(dyaw))
        root_delta = s.dx.qpos[:,:3] - ref_q[:,:3]
        torso_delta = torso-ref_torso
        finite = jnp.all(jnp.isfinite(s.dx.qpos),-1) & jnp.all(jnp.isfinite(s.dx.qvel),-1) & jnp.all(jnp.isfinite(pos),(-2,-1))
        setpoint=jax.vmap(e.ctrl_of)(s.prev_a,source_frame)
        prev_base=e.ref_qpos[source_frame,7:] if e.cfg.residual_action else jnp.broadcast_to(e.q0,s.prev_a.shape)
        prev_setpoint=jnp.clip(prev_base+e.action_scale*prev.prev_a,e.ctrl_lo,e.ctrl_hi)
        prev_setpoint=jnp.where((prev.ep==0)[:,None],e.q0,prev_setpoint)
        soft_excess=jnp.maximum(s.dx.qpos[:,7:]-e.soft_hi,0)+jnp.maximum(e.soft_lo-s.dx.qpos[:,7:],0)
        root_world_vel=s.dx.qvel[:,:3]
        root_body_vel=jax.vmap(lambda q,v:mjm.rotate(v,mjm.quat_inv(q)))(s.dx.qpos[:,3:7],root_world_vel)
        metrics = jnp.stack((jnp.mean(jnp.linalg.norm(pos-ref_pos,axis=-1),-1),
            jnp.mean(jnp.linalg.norm(pos-aligned,axis=-1),-1),
            jnp.linalg.norm(root_delta,axis=-1),jnp.linalg.norm(root_delta[:,:2],axis=-1),root_delta[:,2],
            jnp.linalg.norm(torso_delta,axis=-1),jnp.linalg.norm(torso_delta[:,:2],axis=-1),torso_delta[:,2],
            dyaw,jnp.abs(dyaw),2*jnp.arccos(jnp.clip(jnp.abs(jnp.sum(torso_q*ref_torso_q,axis=-1)),0,1)),
            jnp.mean(jnp.abs(s.dx.qpos[:,7:]-ref_q[:,7:]),axis=-1),
            jnp.sqrt(jnp.mean((setpoint-prev_setpoint)**2,axis=-1)),
            jnp.mean(soft_excess,axis=-1),jnp.sum(soft_excess>0,axis=-1),
            root_world_vel[:,0],root_world_vel[:,1],root_world_vel[:,2],
            root_body_vel[:,0],root_body_vel[:,1],root_body_vel[:,2],
            jnp.linalg.norm(root_world_vel-e.ref_qvel[s.frame,:3],axis=-1),
            jnp.mean(jnp.linalg.norm(body_lin-ref_body_lin,axis=-1),axis=-1)),axis=-1)
        finite = finite & jnp.all(jnp.isfinite(setpoint),-1) & jnp.all(jnp.isfinite(body_lin),(-2,-1)) & jnp.all(jnp.isfinite(body_ang),(-2,-1)) & jnp.all(jnp.isfinite(s.prev_a),-1) & jnp.all(jnp.isfinite(e.ref_body_quat[s.frame]),(-2,-1)) & jnp.all(jnp.isfinite(quat),(-2,-1)) & jnp.all(jnp.isfinite(ref_pos),(-2,-1)) & jnp.all(jnp.isfinite(ref_q),-1) & jnp.all(jnp.isfinite(ref_torso_q),-1) & jnp.all(jnp.isfinite(metrics),-1)
        reward_finite=jnp.all(jnp.isfinite(reward_terms),axis=-1)
        finite=finite & reward_finite
        torso_gravity_z=-1+2*(torso_q[:,1]**2+torso_q[:,2]**2)
        reference_torso_gravity_z=-1+2*(ref_torso_q[:,1]**2+ref_torso_q[:,2]**2)
        return dict(frame=s.frame,active=active,finite=finite,reward_finite=reward_finite,native_reward_terms=reward_terms,terminated=terminated,
                    pelvis_height=pos[:,0,2],reference_pelvis_height=ref_pos[:,0,2],torso_gravity_z=torso_gravity_z,reference_torso_gravity_z=reference_torso_gravity_z,
                    action=s.prev_a,setpoint=setpoint,body_world_lin_vel=body_lin,body_world_ang_vel=body_ang,root_world_velocity=root_world_vel,root_body_velocity=root_body_vel,source_frame=source_frame,qpos=s.dx.qpos,qvel=s.dx.qvel,body_pos=pos,body_quat=quat,metrics=metrics)

    return record

FIELDS = ['world_body_mpjpe_m','donor_aligned_body_mpjpe_m','free_root_position_m','free_root_xy_m','free_root_signed_z_m',
          'torso_position_m','torso_xy_m','torso_signed_z_m','torso_signed_yaw_rad','torso_abs_yaw_rad','torso_orientation_rad','joint_mae_rad','clipped_setpoint_delta_rms_rad','soft_limit_mean_excess_rad','soft_limit_joint_count',
          'root_world_vx_mps','root_world_vy_mps','root_world_vz_mps','root_body_vx_mps','root_body_vy_mps','root_body_vz_mps','root_world_velocity_error_mps','body_world_velocity_error_mps']

def augmented_rollout_factory(ev):
    record = make_record()
    original = ev.make_rollout
    source = textwrap.dedent(inspect.getsource(original))
    replacements = [
        ('return (s2, alive, fall_t, cause, tsum, tbin, cbin, jerr, jabs), None',
         'return (s2, alive, fall_t, cause, tsum, tbin, cbin, jerr, jabs), _record(e, s2, am[:, 0], term, state, tv)'),
        ('(_, _, fall_t, cause, tsum, tbin, cbin, jerr, jabs), _ = jax.lax.scan(',
         '(_, _, fall_t, cause, tsum, tbin, cbin, jerr, jabs), trace = jax.lax.scan('),
        ('return dict(fall_t=fall_t, cause=cause, tsum=tsum, tbin=tbin, cbin=cbin, jerr=jerr, jabs=jabs)',
         'return dict(fall_t=fall_t, cause=cause, tsum=tsum, tbin=tbin, cbin=cbin, jerr=jerr, jabs=jabs), trace')]
    for old,new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Pinned evaluator changed; refuse textual instrumentation')
        source = source.replace(old,new)
    namespace = dict(vars(ev),_record=record)
    exec(compile(source,'<native evaluator with recording>','exec'),namespace)
    augmented = namespace['make_rollout']
    return augmented

def prepare_trace_root(audit_root, names):
    audit_root.mkdir(parents=True, exist_ok=False)
    (audit_root / "expected_clips.json").write_text(json.dumps(dict(
        clips=[dict(clip=n,status="pending_no_trajectory_recorded") for n in names],
        limitation="If compilation or a process interruption prevents a trace return, executed support is unknown; do not infer zero executed steps."),indent=2))


def _atomic_npz(path, payload):
    """Publish a complete NPZ; retain an incomplete .partial file on failure."""
    import numpy as np

    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing scientific output refused: {path}')
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('xb') as stream:
        np.savez_compressed(stream, **payload)
    partial.replace(path)


def _write_owned_npz(path, payload):
    try:
        _atomic_npz(path, payload)
    finally:
        # Drop owned arrays before the Future becomes done. Reserving the next
        # slot cannot overlap a third payload with worker teardown.
        payload.clear()


class _NPZWrites:
    """At most two owned per-clip payloads, drained before leaving a condition.

    Reserve a slot before invoking the payload factory. Threads receive only its
    per-clip copies, never the factory or a closure over the full trace. Metrics
    remain on the calling thread. Compression workspace is additional to the
    bounded payload copies; the original host trace stays owned by the caller.
    """
    def __init__(self, workers):
        if type(workers) is not int or workers not in (1, 2):
            raise ValueError('writer_workers must be 1 or 2')
        self.executor = (ThreadPoolExecutor(max_workers=2, thread_name_prefix='nabla-observed-npz')
                         if workers == 2 else None)
        self.pending = deque()

    def __enter__(self):
        return self

    def submit(self, path, payload_factory):
        import numpy as np

        if len(self.pending) == 2:
            self.pending.popleft().result()
        payload = payload_factory()
        for value in payload.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        if self.executor is None:
            _write_owned_npz(path, payload)
        else:
            self.pending.append(self.executor.submit(_write_owned_npz, path, payload))

    def __exit__(self, exc_type, exc, traceback):
        first_error = None
        try:
            while self.pending:
                try:
                    self.pending.popleft().result()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
        if first_error is not None:
            raise first_error from exc
        return False


def write_observation(env, names, res, trace, capacity, audit_root, checkpoint_iter, policy,
                      zero_residual=False, writer_workers=1):
    import numpy as np
    from nablamimic import evaluate as ev
    if type(writer_workers) is not int or writer_workers not in (1, 2):
        raise ValueError('writer_workers must be 1 or 2')
    fields = FIELDS
    own_scalars=ev.summarize(res,env,names)
    _atomic_npz(audit_root / f'it{checkpoint_iter}_native_aggregates.npz', res)
    (audit_root / f'it{checkpoint_iter}_same_execution_scalars.json').write_text(json.dumps(clean_json(dict(
        protocol=PROTOCOL,execution_condition='observed',checkpoint_iter=checkpoint_iter,policy=policy,scalars=own_scalars)),indent=2,allow_nan=False))
    try:
        capacity_report,physics_invalid,contacts,rows_requested,nnz_requested,overflow_flags=capacity.read(trace['active'],env.warp[40].nconmax*40,env.warp[40].njmax)
    except Exception as exc:
        with _NPZWrites(writer_workers) as writes:
            for i,name in enumerate(names):
                support=np.asarray(trace['active'][:,i],bool)
                writes.submit(audit_root / f'it{checkpoint_iter}_{name}_capacity_unverified.npz',
                    lambda i=i,support=support: dict(checkpoint_iter=checkpoint_iter,policy=policy,
                        **{k:np.asarray(v)[support,i] for k,v in trace.items()}))
        (audit_root / f'it{checkpoint_iter}_capacity_unverified.json').write_text(json.dumps(dict(
            protocol=PROTOCOL,checkpoint_iter=checkpoint_iter,policy=policy,fields=fields,error=repr(exc),physics_validity='unverified',accepted=False,per_clip=[dict(clip=n,status='raw_trace_preserved_capacity_unverified') for n in names]),indent=2))
        raise
    _atomic_npz(audit_root / f'it{checkpoint_iter}_capacity.npz', dict(contacts=contacts,rows_requested=rows_requested,sparse_nnz_requested=nnz_requested,overflow_flags=overflow_flags,active=trace['active']))
    invalid_lineage=np.maximum.accumulate(physics_invalid | (np.asarray(trace['active'],bool) & ~np.asarray(trace['finite'],bool)),axis=0)
    trace['capacity_invalid_direct']=physics_invalid
    trace['invalid_lineage']=invalid_lineage
    rows=[]
    with _NPZWrites(writer_workers) as writes:
        for i,name in enumerate(names):
            support=np.asarray(trace['active'][:,i],bool)
            finite=np.asarray(trace['finite'][:,i],bool)
            valid=support & finite & ~invalid_lineage[:,i]
            writes.submit(audit_root / f'it{checkpoint_iter}_{name}.npz',
                lambda i=i,support=support: dict(checkpoint_iter=checkpoint_iter,policy=policy,
                    **{k:np.asarray(v)[support,i] for k,v in trace.items()}))
            vals=np.asarray(trace['metrics'][:,i])
            means={f:float(vals[valid,j].mean()) if valid.any() else None for j,f in enumerate(fields)}
            summaries={}
            for label,mask in [('p95',valid),('last_second',valid & (np.arange(len(valid)) >= max(int(support.sum())-50,0)))]:
                summaries[label]={f:float(np.percentile(vals[mask,j],95) if label=='p95' else vals[mask,j].mean()) if mask.any() else None for j,f in enumerate(fields)}
            bins=[]
            for b in range(0,int(support.sum()),500):
                mask=valid & (np.arange(len(valid))>=b) & (np.arange(len(valid))<b+500)
                bins.append(dict(start_s=b*.02,planned_bin_steps=min(500,int(support.sum())-b),valid_steps=int(mask.sum()),means={f:float(vals[mask,j].mean()) if mask.any() else None for j,f in enumerate(fields)}))
            def sustained_flag(height,gravity_z):
                flags=((np.asarray(height)<.25)|(np.asarray(gravity_z)>-.2)) & support
                streak=longest=0
                for flag in flags:
                    streak=streak+1 if flag else 0
                    longest=max(longest,streak)
                return dict(triggered=longest>=10,longest_consecutive_steps=int(longest),duration_s=.02*longest,
                    scope='Diagnostic only; low/get-up reference poses may trigger; no contact-force inference')
            observed_pose_diagnostic=sustained_flag(trace['pelvis_height'][:,i],trace['torso_gravity_z'][:,i])
            reference_pose_diagnostic=sustained_flag(trace['reference_pelvis_height'][:,i],trace['reference_torso_gravity_z'][:,i])
            complete=bool(support.sum()==env.lengths[i]-1 and res['fall_t'][i]<0)
            final_drift={f:float(vals[int(support.sum())-1,j]) if complete and support.any() and valid[support].all() else None for j,f in enumerate(fields) if f in ('world_body_mpjpe_m','torso_xy_m','torso_position_m','torso_abs_yaw_rad')}
            rows.append(dict(checkpoint_iter=checkpoint_iter,policy=policy,clip=name,planned_steps=int(env.lengths[i]-1),executed_steps=int(support.sum()),
                valid_metric_steps=int(valid.sum()),nonfinite_reward_steps=int((support & ~np.asarray(trace['reward_finite'][:,i],bool)).sum()),invalid_metric_steps=int((support & ~valid).sum()),
                donor_tracking_termination=bool(res['fall_t'][i]>0),fall_step=int(res['fall_t'][i]),
                completion=complete,observed_pose_diagnostic=observed_pose_diagnostic,reference_pose_diagnostic=reference_pose_diagnostic,final_drift=final_drift,means=means,p95=summaries['p95'],last_second=summaries['last_second'],bins_10s=bins,
                direct_capacity_invalid_steps=int((physics_invalid[:,i]&support).sum()),invalid_lineage_steps=int((invalid_lineage[:,i]&support).sum()),physics_status='invalid' if (invalid_lineage[:,i]&support).any() else 'no_observed_capacity_loss'))
    # Means on declared support; no survivor-only filtering or imputation.
    macro={f:float(np.mean([r['means'][f] for r in rows])) if all(r['means'][f] is not None for r in rows) else None for f in fields}
    weights=np.array([r['valid_metric_steps'] for r in rows])
    pooled={f:float(sum((r['means'][f] or 0)*w for r,w in zip(rows,weights))/weights.sum()) if weights.sum() else None for f in fields}
    report=dict(protocol=PROTOCOL,execution_condition='observed',same_execution_scalars=own_scalars,control='zero_residual' if zero_residual else 'learned',checkpoint_iter=checkpoint_iter,policy=policy,call=1,fields=fields,per_clip=rows,macro_executed_support=macro,
        pooled_executed_support=pooled,capacity=capacity_report,clip_denominator=40,metric_defined_clip_count=sum(bool(r['valid_metric_steps']) for r in rows),complete_clip_count=sum(r['completion'] for r in rows),
        interpretation='Observed execution only. Original donor scalars/validity are independent; failed historical equivalence margins remain failed.')
    apply_gate(report)
    eligible=[r for r in rows if r['completion'] and r['invalid_metric_steps']==0 and r['valid_metric_steps']>0]
    report['worst_five_complete_valid_by_p95_world_mpjpe']=[dict(clip=r['clip'],p95_world_mpjpe_m=r['p95']['world_body_mpjpe_m']) for r in sorted(eligible,key=lambda r:r['p95']['world_body_mpjpe_m'],reverse=True)[:5]]
    report['ranking_excluded_clip_identities']=[r['clip'] for r in rows if not (r['completion'] and r['invalid_metric_steps']==0 and r['valid_metric_steps']>0)]
    (audit_root / f'it{checkpoint_iter}_metrics.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
    return own_scalars, report


def evaluate_condition(session, capacity, iteration, policy, output, audit_root, zero_residual=False, writer_workers=1):
    """Execute and save one condition; reset and timing evidence stay in sidecars."""
    import jax

    if type(writer_workers) is not int or writer_workers not in (1, 2):
        raise ValueError('writer_workers must be 1 or 2')
    if output.exists():
        raise ValueError(f"Existing output refused: {output}")
    started = time.perf_counter()
    checkpoint, policy, actor = session.load_checkpoint(iteration, policy)
    load_s = time.perf_counter() - started
    (res, trace), boundary, phases = session.run_actor(actor, before_rollout=capacity.reset)
    # Persist boundary evidence before capacity/metrics processing can fail.
    output.with_suffix('.boundary.json').write_text(json.dumps(boundary, indent=2, allow_nan=False))
    phases['checkpoint_load_s'] = load_s
    phases['writer_workers'] = writer_workers
    phases['maximum_in_flight_clip_writes'] = writer_workers
    output.with_suffix('.timings.json').write_text(json.dumps(phases, indent=2, allow_nan=False))
    write_started = time.perf_counter()
    own_scalars, report = write_observation(
        session.env, session.names, res, trace, capacity, audit_root, iteration, policy,
        zero_residual=zero_residual, writer_workers=writer_workers)
    phases['trace_and_metrics_write_s'] = time.perf_counter() - write_started
    rec = dict(own_scalars)
    rec.update(iter=int(checkpoint['iter']), wall_s=float(checkpoint['wall_s']),
               seconds=round(time.perf_counter() - started, 1), gpu=jax.devices()[0].device_kind,
               solver_iterations=session.solver_iterations, action_rate_space='setpoint', policy=policy,
               forward_only=session.forward_only, completed_updates=checkpoint['completed_updates'],
               ema_updates=checkpoint['ema_updates'])
    from nablamimic import evaluate as ev
    observed = dict(clips=session.names, terms=list(ev.TERMS), run=str(session.run),
                    solver_iterations=session.solver_iterations, forward_only=session.forward_only,
                    protocol=PROTOCOL, execution_condition='observed', policy=policy,
                    control='zero_residual' if zero_residual else 'learned')
    observed[f'it{iteration}/nominal/{policy}'] = rec
    scalar_write_started = time.perf_counter()
    output.write_text(json.dumps(clean_json(observed), allow_nan=False))
    phases.update(scientific_json_write_s=time.perf_counter() - scalar_write_started,
                  total_condition_s=time.perf_counter() - started)
    output.with_suffix('.timings.json').write_text(json.dumps(phases, indent=2, allow_nan=False))
    return observed, report


def write_failure(audit_root, names, error):
    (audit_root / 'FAILED.json').write_text(json.dumps(dict(
        error=repr(error), accepted=False,
        per_clip=[dict(clip=n, status='see_existing_call_trace_or_execution_unknown') for n in names]),
        indent=2, allow_nan=False))
