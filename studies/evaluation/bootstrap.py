"""Set private Warp cache and observe actual Engine cache reuse in this trial."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import warp
warp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
mode, target, *tail = sys.argv[1:]
sys.argv = [target, *tail]
if mode=='--script':
    sys.path.insert(0,str(Path(target).resolve().parent))  # External helper siblings only.
observed=[]
allocator_reading=None
audit_path=Path(os.environ['NABLA_ENGINE_CACHE_AUDIT'])
if mode=='--script' and Path(target).name=='guarded_train.py':
    from nablaengine import jax_ffi
    native_init=jax_ffi.init
    def audited_init(*args, **kwargs):
        global allocator_reading
        result=native_init(*args, **kwargs)
        stats=dict(jax_ffi.STATS)
        if allocator_reading is None:
            # Native Engine initialization has already initialized this child's
            # CUDA context. The parent runner never calls a JAX device API.
            import jax
            device=jax.local_devices()[0]
            memory=device.memory_stats() or {}
            allocator_reading=dict(device_kind=device.device_kind,
                options={key:os.environ.get(key) for key in (
                    'XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION',
                    'XLA_PYTHON_CLIENT_PREALLOCATE','XLA_PYTHON_CLIENT_ALLOCATOR')},
                memory_stats={key:int(memory[key]) for key in (
                    'bytes_limit','bytes_in_use','peak_bytes_in_use','bytes_reserved','pool_bytes')
                    if key in memory},
                timing='Once, after native Engine initialization in training subprocess')
        observed.append(dict(monotonic_s=time.monotonic(), stats=stats, precompiled_cache_hit=(stats.get('builds') == 0 and stats.get('loads', 0) > 0)))
        audit_path.write_text(json.dumps(dict(initializations=observed,allocator=allocator_reading),indent=2)+'\n')
        # Observe genuine native initialization. Cache misses may compile with
        # real nvcc under the existing setup budget; they are startup evidence.
        return result
    jax_ffi.init=audited_init
try:
    if mode == '--module':
        runpy.run_module(target, run_name='__main__', alter_sys=True)
    elif mode == '--script':
        runpy.run_path(target, run_name='__main__')
    else:
        raise ValueError(mode)
finally:
    ffi=sys.modules.get('nablaengine.jax_ffi')
    audit_path.write_text(json.dumps(dict(initializations=observed,allocator=allocator_reading,
        final_stats=dict(ffi.STATS) if ffi else None,
        interpretation='Observed build/load counts and cache reuse; cache misses may compile under the setup budget. Zero builds is an environment cache-reuse test requirement only.'),indent=2)+'\n')
