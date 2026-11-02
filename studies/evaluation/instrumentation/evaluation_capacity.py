"""External read-only substep instrumentation of native MJWarp `_Sim.forward`.

The pinned source keeps requested d.nacon (global) and d.nefc (per world)
above capacity. Pinned make_constraint also keeps an actual temporary requested
NNZ counter per world; its existing zero-count launch is observed to retain that
array, without modifying it. One tiny extra kernel per substep writes these values; no
simulation arrays or solver options are modified. Host sync happens once/rollout.
"""
import numpy as np
import warp as wp

@wp.kernel(enable_backward=False)
def _record(nacon: wp.array(dtype=int), nefc: wp.array(dtype=int), requested_nnz: wp.array(dtype=int), overflow: wp.array(dtype=int),
            contacts: wp.array(dtype=int), rows: wp.array2d(dtype=int), nnz: wp.array2d(dtype=int), flags: wp.array2d(dtype=int),
            cursor: wp.array(dtype=int), overrun: wp.array(dtype=int), nworld: int):
    k = cursor[0]
    if k < contacts.shape[0]:
        contacts[k] = nacon[0]
        for i in range(nworld):
            rows[k, i] = nefc[i]
            nnz[k, i] = requested_nnz[i]
            flags[k, i] = overflow[i]
    else:
        overrun[0] = 1
    cursor[0] = k + 1

class CapacityAudit:
    def __init__(self, sim, slots):
        self.sim=sim
        self.contacts=wp.zeros(slots,dtype=int)
        self.rows=wp.zeros((slots,sim.nworld),dtype=int)
        self.nnz=wp.zeros((slots,sim.nworld),dtype=int)
        self.flags=wp.zeros((slots,sim.nworld),dtype=int)
        self.nnz_request=None
        self._request_buffers=[]
        self.cursor=wp.zeros(1,dtype=int)
        self.overrun=wp.zeros(1,dtype=int)
    def reset(self):
        self.cursor.zero_();self.overrun.zero_()
    def append(self,d):
        if self.nnz_request is None:
            raise RuntimeError('Actual sparse NNZ request counter unavailable; refuse capacity evidence')
        wp.launch(_record,dim=1,inputs=[d.nacon,d.nefc,self.nnz_request,d.overflow,self.contacts,self.rows,self.nnz,self.flags,self.cursor,self.overrun,self.sim.nworld])
    def read(self,active,nconmax,njmax):
        if nconmax!=self.sim.df.naconmax or njmax!=self.sim.df.njmax:
            raise RuntimeError('Reported/native capacity mismatch')
        count=int(self.cursor.numpy()[0]); overrun=bool(self.overrun.numpy()[0])
        expected=len(active)*4
        if overrun or count!=expected:
            raise RuntimeError(f'capacity audit support mismatch: count={count}, expected={expected}, overrun={overrun}')
        contacts=self.contacts.numpy()[:count].reshape(-1,4)
        rows=self.rows.numpy()[:count].reshape(-1,4,self.sim.nworld)
        nnz=self.nnz.numpy()[:count].reshape(-1,4,self.sim.nworld)
        flags=self.flags.numpy()[:count].reshape(-1,4,self.sim.nworld)
        from mujoco_warp._src.types import OverflowType
        known_mask=sum(int(bit) for bit in OverflowType)
        diagnostic_mask=int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        # All other written bits are retained and conservatively invalidate.
        # Unknown bits invalidate too; availability is never synthesized as zero.
        bad_flags=(flags & ~diagnostic_mask)!=0
        active=np.asarray(active,bool)
        # Conservative >= for row capacity also catches multicomponent equality
        # kernels' early-drop boundary. No equality constraints in G1 protocol.
        contact_over=contacts>nconmax
        row_at_limit=rows>=njmax
        nnz_over=nnz>self.sim.df.njmax_nnz if self.sim.m.is_sparse else np.zeros_like(nnz,dtype=bool)
        invalid=(np.any(contact_over,axis=1)[:,None] | np.any(row_at_limit | nnz_over | bad_flags,axis=1)) & active
        report=dict(substeps=count,expected_substeps=expected,global_contact_capacity=int(nconmax),
            per_world_constraint_capacity=int(njmax),max_requested_contacts=int(contacts.max(initial=0)),
            max_requested_rows=int(rows.max(initial=0)),contact_overflow_substeps=int(contact_over.sum()),
            sparse=bool(self.sim.m.is_sparse),per_world_sparse_nnz_capacity=int(self.sim.df.njmax_nnz),
            max_requested_sparse_nnz=int(nnz.max(initial=0)) if self.sim.m.is_sparse else None,
            sparse_nnz_request_observed=True,overflow_flags_observed=True,
            overflow_flag_values=[int(x) for x in np.unique(flags)],
            overflow_bits={bit.name:int(bit) for bit in OverflowType},
            active_flag_observations={bit.name:int((((flags & int(bit))!=0) & active[:,None,:]).sum()) for bit in OverflowType},
            ignored_diagnostic_mask=diagnostic_mask,unknown_flag_mask=int(np.bitwise_or.reduce(flags.ravel(),initial=0)) & ~known_mask,
            active_lost_physics_flag_observations=int((bad_flags & active[:,None,:]).sum()),
            overflow_semantics='native sticky per-world bits within each condition; no resets during its trajectory; persistent initial state is restored only between conditions; counts are observations, not newly raised events',active_sparse_nnz_overflow_events=int((nnz_over & active[:,None,:]).sum()),
            active_constraint_capacity_events=int((row_at_limit & active[:,None,:]).sum()),
            active_invalid_world_steps=int(invalid.sum()),
            validity='invalid' if invalid.any() else 'no_observed_capacity_loss',
            sampling_hz=200,semantics='actual requested global contacts and per-world rows/pooled sparse NNZ; contact/NNZ overflow strictly > capacity; row equality conservatively rejected',
            per_row_semantics='No per-row NNZ capacity. Stored J_rownnz/J_rowadr describe admitted rows, not dropped NNZ requests; never substituted for actual requested world total.')
        return report,invalid,contacts,rows,nnz,flags


def install(sim_module):
    """Patch factory so importing WarpPhysics follows native mjw_options configuration."""
    original_factory=sim_module.make_warp
    audits=[]
    installed=False
    def make_warp(env,sizes,*args,**kwargs):
        nonlocal installed
        from nablamimic import mjw_options
        # Exact default nominal options of donor make_warp; refuse training here.
        if kwargs.get('train',True) is not False or args:
            raise ValueError('capacity recorder supports native nominal train=False only')
        mjw_options.configure(False,0,1e-3,0,0.0,0)
        from nablamimic import warp_physics as native
        if not installed:
            original_init=native._Sim.__init__
            original_step=native.mjw.step
            registered={}
            by_nefc={}
            from mujoco_warp._src import constraint
            original_launch=wp.launch
            def launch(*a,**kw):
                kernel=a[0] if a else kw.get("kernel")
                if kernel is constraint._zero_constraint_counts:
                    inputs=kw.get("inputs")
                    if inputs is None or len(inputs)!=6:
                        raise RuntimeError("Pinned constraint counter launch signature changed")
                    audit=by_nefc.get(id(inputs[3]))
                    if audit is not None:
                        audit.nnz_request=inputs[5]
                        audit._request_buffers.append(inputs[5])
                return original_launch(*a,**kw)
            wp.launch=launch
            slots=4*(int(env.lengths.max()-1)+1024)
            def init(self,*a,**kw):
                original_init(self,*a,**kw)
                audit=CapacityAudit(self,slots)
                audits.append(audit)
                registered[id(self.df)]=audit
                by_nefc[id(self.df.nefc)]=audit
            def step(m,d,*a,**kw):
                audit=registered.get(id(d))
                if audit is not None:
                    audit.nnz_request=None
                result=original_step(m,d,*a,**kw)
                if audit is not None:
                    audit.append(d)
                return result
            native._Sim.__init__=init
            native.mjw.step=step
            installed=True
        return original_factory(env,sizes,*args,**kwargs)
    sim_module.make_warp=make_warp
    return audits
