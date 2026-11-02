"""Strict same-model NablaEngine Torch/JAX parity and compiled CUDA-graph proof.

GPU execution belongs in the provisioned pinned environment. Every required stage
must pass; unsupported compile/capture never selects an eager replacement. Packed
tape bytes are compared as bytes, while decoded physical float fields are checked
for finiteness (integer bit fields may legitimately look like NaNs as float32).
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from engine_contract import (add_engine_capacity_arguments, expected_engine_capacity,
                             capacity_validator_command, capacity_validator_environment)

INPUTS = ('qpos','qvel','warm','ctrl','friction','com')
STATE_OUTPUTS = ('qpos1','qvel1','warm1','qpos_n','xpos','xquat','cvel','subtree_com')
KIN_OUTPUTS = STATE_OUTPUTS[3:]
GRAD_OUTPUTS = ('grad_qpos','grad_qvel','grad_ctrl')
BAD_DIAGNOSTICS = ('fwd_unsupported_worlds','fwd_contact_overflow_worlds','fwd_row_overflow_worlds',
                   'nonfinite_rows','nonfinite_cotangent_rows','replay_mismatch_worlds')
OPTIONS = dict(iterations=10,substeps=4,clamp_kv=0,normal=3,clamp_soft=None,grad_decay=1.0)


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def host_inputs(reference_cache, model):
    import numpy as np
    from nablamimic.motion import all_clip_names
    from nablamimic.env import BM_BODIES
    names=all_clip_names()
    if len(names)!=40 or len(list(reference_cache.glob('*.npz')))!=40:
        raise ValueError('Exactly forty current reference CSV/cache identities are required')
    rows=[]; q=[];v=[];u=[];global_offset=0
    required=('qpos','qvel','body_pos','body_quat','body_lin_vel','body_ang_vel')
    for name in names:
        files=list(reference_cache.glob(name+'_50hz_*.npz'))
        if len(files)!=1:raise ValueError(f'Expected one current 50Hz cache for {name}')
        with np.load(files[0],allow_pickle=False) as cache:
            if set(cache.files)!=set(required):raise ValueError(f'Unexpected reference fields: {name}')
            qp, qv=cache['qpos'],cache['qvel']
            if qp.ndim!=2 or qp.shape[1]!=36 or qv.shape!=(len(qp),35) or len(qp)<2:
                raise ValueError(f'Invalid reference state dimensions: {name}')
            if cache['body_pos'].shape!=(len(qp),len(BM_BODIES),3):
                raise ValueError(f'Invalid fourteen-body reference layout: {name}')
            for frame in (0,(len(qp)-1)//2):
                q.append(qp[frame]);v.append(qv[frame]);u.append(np.clip(qp[min(frame+1,len(qp)-1),7:],
                    model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]))
                rows.append(dict(clip=name,frame=int(frame),global_frame=global_offset+int(frame),reference=str(files[0])))
            global_offset+=len(qp)
    q,v,u=(np.ascontiguousarray(value,np.float32) for value in (q,v,u))
    batch=len(q)
    rng=np.random.default_rng(20261103)
    # Fixed supported training randomization inputs, shared exactly by both APIs.
    friction=rng.uniform(.3,1.2,batch).astype(np.float32)
    com=rng.uniform([-0.025,-0.05,-0.05],[0.025,0.05,0.05],(batch,3)).astype(np.float32)
    values=[q,v,np.zeros_like(v),u,friction,com]
    if not all(np.isfinite(value).all() for value in values):raise ValueError('Nonfinite input state')
    if not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,rtol=0,atol=1e-5):
        raise ValueError('Reference root quaternions are not normalized')
    return values,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference-cache',type=Path,required=True)
    parser.add_argument('--engine-root',type=Path,required=True)
    parser.add_argument('--capacity-validator',type=Path,required=True)
    parser.add_argument('--jax-cache-seed',type=Path,default=os.environ.get('NABLA_ENGINE_CACHE_SEED'))
    parser.add_argument('--engine-base-revision',required=True)
    parser.add_argument('--engine-capacity-revision',required=True)
    parser.add_argument('--expected-gpu-model',default='NVIDIA CMP 170HX')
    parser.add_argument('--torch-memory-fraction',type=float,default=.45)
    parser.add_argument('--jax-memory-fraction',type=float,default=.45)
    add_engine_capacity_arguments(parser)
    args=parser.parse_args()
    capacity = expected_engine_capacity(args)
    if (not 0<args.torch_memory_fraction<=1 or not 0<args.jax_memory_fraction<=1
            or args.torch_memory_fraction+args.jax_memory_fraction>1):
        parser.error('Positive framework memory fractions must sum to at most one')
    if os.environ.get('DK_ALLOW_UNSUPPORTED'):
        parser.error('Unsupported-physics bypass is forbidden')
    if os.environ.get('JAX_PLATFORMS') in ('cpu','interpreter'):
        parser.error('This probe requires explicitly provisioned CUDA execution')
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    record=dict(expected_engine_capacity=capacity,status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                argv=sys.argv,python=sys.executable,options=OPTIONS,checks=[],failures=[],
                engine_base_revision=args.engine_base_revision,engine_capacity_revision=args.engine_capacity_revision,
                reference_cache=str(args.reference_cache),expected_gpu_model=args.expected_gpu_model,
                scope='Native one-step/interface/compiler checks only; no skill, training-speed or full-horizon claim')
    started=time.monotonic();private_tmp=None
    def save():write(root/'result.json',record)
    def check(name,passed,**fields):
        record['checks'].append(dict(name=name,passed=bool(passed),**fields))
        if not passed:record['failures'].append(name)
        save()
        if not passed:raise AssertionError(name)
    save()
    try:
        private_tmp=tempfile.mkdtemp(prefix='nabla-torch-',dir='/tmp')
        os.environ['TMPDIR']=private_tmp;tempfile.tempdir=private_tmp
        record['compiler_tmpdir']=private_tmp
        # Pin private writable compilation/cache paths before importing either GPU runtime.
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
        os.environ['XLA_CLIENT_MEM_FRACTION']=str(args.jax_memory_fraction)
        os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION',None)
        for key,name in (('DK_CACHE_DIR','native'),('JAX_COMPILATION_CACHE_DIR','jax'),
                         ('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),
                         ('CUDA_CACHE_PATH','cuda')):
            path=root/'cache'/name;path.mkdir(parents=True);os.environ[key]=str(path)
        record['jax_cache_seed']=None
        if args.jax_cache_seed:
            seed=args.jax_cache_seed.resolve()
            libraries=sorted(seed.glob('*.so'))
            if not seed.is_dir() or not libraries:raise ValueError('Declared JAX cache seed contains no native libraries')
            for path in libraries:shutil.copyfile(path,root/'cache'/'native'/path.name)
            record['jax_cache_seed']=dict(path=str(seed),copied_libraries=[p.name for p in libraries])
        # The prepared reference cache must be used without rebuilding it in the source tree.
        os.environ['MOTION_CACHE_DIR']=str(args.reference_cache.resolve())
        validation=subprocess.run(capacity_validator_command(args, root), capture_output=True, text=True,
                                  env=capacity_validator_environment(), timeout=180)
        (root/'capacity-validator.log').write_text(validation.stdout+validation.stderr)
        check('corrected_source_capacity_validator',validation.returncode==0,returncode=validation.returncode)

        import numpy as np
        import torch
        import mujoco
        from nablamimic.env import EnvConfig,build_model
        from nablaengine import torch_ffi as tf
        from nablaengine.g1_engine import step as engine_step,tape as engine_tape
        from nablaengine.model import capability
        expected=args.engine_root.resolve()
        check('engine_source_location',Path(tf.__file__).resolve().is_relative_to(expected),module=str(Path(tf.__file__).resolve()))
        check('single_visible_gpu',torch.cuda.device_count()==1,visible_count=torch.cuda.device_count())
        torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(args.torch_memory_fraction,0)
        check('gpu_model',torch.cuda.get_device_name(0)==args.expected_gpu_model,actual=torch.cuda.get_device_name(0))
        model,kp=build_model(EnvConfig(solver_iterations=10,residual_action=True))
        model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        foot_pairs=np.asarray([i for i in range(model.npair) if '_foot' in model.pair(i).name
                               and model.pair(i).name.endswith('_floor')],np.int64)
        capability.require(model,foot_pairs,capability.COM_BODY,'engine','torch-jax native check')
        check('model_contract',model.nq==36 and model.nv==35 and model.nu==29 and
              model.opt.timestep==.005 and model.opt.iterations==10 and engine_step.NEFC==capacity['constraint_rows'] and
              engine_step.maxcon(model)==capacity['max_contacts'] and engine_tape.layout(model)['zone'][1]==capacity['zone_words'] and
              engine_tape.rec_words(model)==capacity['record_words'] and np.count_nonzero(model.dof_frictionloss)==0 and
              np.count_nonzero(model.jnt_limited)==29,
              nq=model.nq,nv=model.nv,nu=model.nu,nbody=model.nbody,rows=engine_step.NEFC,
              contacts=engine_step.maxcon(model),record_words=engine_tape.rec_words(model),
              frictionloss_rows=int(np.count_nonzero(model.dof_frictionloss)),limited_joints=int(np.count_nonzero(model.jnt_limited)),
              tape_layout={key:list(value) for key,value in engine_tape.layout(model).items()},
              timestep=float(model.opt.timestep),foot_pairs=foot_pairs.tolist())
        host,rows=host_inputs(args.reference_cache,model)
        record['input_rows']=rows;record['nworld']=len(host[0]);save()
        np.savez(root/'inputs.npz',**dict(zip(INPUTS,host)))
        # Both libraries are built explicitly from this exact MjModel, not a default fixture.
        info=tf.init(model=model,foot_pairs=foot_pairs)
        allowed=('path','artifact','arch','version','nq','nv','nu','nbody','maxcon','nefc','rec_words',
                 'timestep','foot_pairs','variants','forward_unit','built','arch_flags','forward_defines','kinematics_error')
        record['torch_library']={key:info[key] for key in allowed if key in info}
        record['torch_version']=torch.__version__;record['torch_cuda']=torch.version.cuda
        record['package_versions']={name:metadata.version(name) for name in ('torch','triton','jax','jaxlib','numpy','mujoco','warp-lang','nablaengine')}
        check('torch_tape_shape_contract',info['rec_words']==engine_tape.rec_words(model) and
              info['nq']==model.nq and info['nv']==model.nv and info['nu']==model.nu and
              info['foot_pairs']==foot_pairs.tolist() and info['kinematics_error'] is None)
        tf.set_diagnostics(True)
        def tensor(value):return torch.from_numpy(np.ascontiguousarray(value)).to('cuda')
        tin=[tensor(value) for value in host]
        def cpu(value):return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        def compare(name,left,right,exact=True,packed=False):
            a,b=np.ascontiguousarray(cpu(left)),np.ascontiguousarray(cpu(right))
            same=a.shape==b.shape and a.dtype==b.dtype
            equal=same and (a.tobytes()==b.tobytes() if exact else np.allclose(a,b,rtol=1e-5,atol=1e-6))
            finite=(np.isfinite(a).all() and np.isfinite(b).all()) if a.dtype.kind in 'fc' else True
            check(name,equal and (packed or finite),shape=list(a.shape),dtype=str(a.dtype),bitwise=exact,finite=bool(finite),packed_integer_fields=packed,
                  max_abs_difference=(float(np.max(np.abs(a-b))) if same and finite and a.size else None))
        def guards(name):
            diagnostic=tf.read_diagnostics(reset=True)
            check(name,all(key in diagnostic and diagnostic[key]==0 for key in BAD_DIAGNOSTICS),diagnostics=diagnostic)
            return diagnostic
        def arrays(prefix,outputs,names):
            values={name:cpu(value) for name,value in zip(names,outputs)}
            np.savez(root/(prefix+'.npz'),**values)
            return values
        initial=tf.forward(*tin,tape=True,kinematics=True)
        for name,value in zip(STATE_OUTPUTS,(*initial[:3],*initial[5:])):
            check('initial_finite_'+name,np.isfinite(cpu(value)).all())
        decoded=engine_tape.decode(cpu(initial[4]).reshape(-1,info['rec_words']),model)
        for key in ('qpos','qvel','qacc','qacc_adv','L_H','L_Q'):
            check('tape_finite_'+key,np.isfinite(decoded[key]).all())
        check('tape_capacity',np.all(decoded['ncon']<=capacity['max_contacts']) and np.all(decoded['nefc']<=capacity['constraint_rows'])
              and np.all(decoded['nlim']<=29) and not np.any(decoded['unsupported']),
              max_contacts=int(decoded['ncon'].max()),max_rows=int(decoded['nefc'].max()),
              max_limits=int(decoded['nlim'].max()),header_bits=[7,9,8,8])
        arrays('torch_native_forward',initial,('qpos1','qvel1','warm1','info','tape',*KIN_OUTPUTS))
        rng=np.random.default_rng(7)
        cots=[rng.standard_normal(tuple(value.shape)).astype(np.float32)*.01
              for value in (initial[0],initial[1],*initial[5:])]
        tcots=[tensor(value) for value in cots]
        np.savez(root/'cotangents.npz',**{name:value for name,value in zip(('qpos1','qvel1',*KIN_OUTPUTS),cots)})
        guards('initial_native_diagnostics')

        # Capture native FFI first, before JAX initializes its CUDA backend.
        static=[value.clone() for value in tin]
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                f=tf.forward(*static,tape=True,kinematics=True)
                tf.backward(f[4],static[4],static[5],tcots[0],tcots[1],
                            kinematics=(f[0],f[1],*tcots[2:]))
        torch.cuda.current_stream().wait_stream(side)
        native_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(native_graph):
            graph_f=tf.forward(*static,tape=True,kinematics=True)
            graph_b=tf.backward(graph_f[4],static[4],static[5],tcots[0],tcots[1],
                                kinematics=(graph_f[0],graph_f[1],*tcots[2:]))
        for shift in (1,7):
            changed=[np.ascontiguousarray(np.roll(value,shift,axis=0)) for value in host]
            check(f'graph_changed_inputs_{shift}',not np.array_equal(changed[0],host[0]))
            for target,value in zip(static,changed):target.copy_(tensor(value))
            native_graph.replay();torch.cuda.synchronize()
            eager_f=tf.forward(*[tensor(value) for value in changed],tape=True,kinematics=True)
            eager_b=tf.backward(eager_f[4],tensor(changed[4]),tensor(changed[5]),tcots[0],tcots[1],
                                kinematics=(eager_f[0],eager_f[1],*tcots[2:]))
            for i,(left,right) in enumerate(zip((*graph_f,*graph_b),(*eager_f,*eager_b))):
                compare(f'native_graph_shift{shift}_output{i}',left,right,packed=i==4)
        guards('native_graph_diagnostics')

        def loss_function(q,v,w,u,fr,com,*cot):
            output=tf.step(q,v,w,u,fr,com,**OPTIONS,kinematics=True)
            loss=sum((value*gradient).sum() for value,gradient in zip((output[0],output[1],*output[3:]),cot))
            return output,loss
        torch._dynamo.config.suppress_errors=False
        x=[value.clone().requires_grad_(i in (0,1,3)) for i,value in enumerate(tin)]
        # Initialize the lazy autograd class before tracing; class construction is
        # not part of the tensor program that fullgraph is intended to prove.
        _,warm_loss=loss_function(*x,*tcots);warm_loss.backward();torch.cuda.synchronize()
        explanation=torch._dynamo.explain(loss_function)(*x,*tcots)
        (root/'compile-explanation.txt').write_text(str(explanation))
        check('compile_full_graph',explanation.graph_count==1 and explanation.graph_break_count==0,
              graph_count=int(explanation.graph_count),graph_break_count=int(explanation.graph_break_count))
        compiled=torch.compile(loss_function,backend='inductor',fullgraph=True,dynamic=False,
                               options={'triton.cudagraphs':False})
        def execute(fn,values,cotangent):
            inputs=[tensor(value).requires_grad_(i in (0,1,3)) for i,value in enumerate(values)]
            output,loss=fn(*inputs,*cotangent);loss.backward();torch.cuda.synchronize()
            return output,loss,[inputs[i].grad for i in (0,1,3)]
        eager_output,eager_loss,eager_grad=execute(loss_function,host,tcots)
        compiled_output,compiled_loss,compiled_grad=execute(compiled,host,tcots)
        for name,a,b in zip(STATE_OUTPUTS,eager_output,compiled_output):compare('compiled_'+name,a,b)
        for name,a,b in zip(GRAD_OUTPUTS,eager_grad,compiled_grad):compare('compiled_'+name,a,b)
        compare('compiled_scalar_loss',eager_loss,compiled_loss,exact=False)
        # CUDA graph contains compiled forward, AOT backward and gradient clearing.
        gx=[value.clone().requires_grad_(i in (0,1,3)) for i,value in enumerate(tin)]
        for i in (0,1,3):gx[i].grad=torch.zeros_like(gx[i])
        gcot=[value.clone() for value in tcots]
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                for i in (0,1,3):gx[i].grad.zero_()
                _,loss=compiled(*gx,*gcot);loss.backward()
        torch.cuda.current_stream().wait_stream(side)
        compiled_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(compiled_graph):
            for i in (0,1,3):gx[i].grad.zero_()
            graph_output,graph_loss=compiled(*gx,*gcot)
            graph_loss.backward()
        for shift in (1,7):
            changed=[np.ascontiguousarray(np.roll(value,shift,axis=0)) for value in host]
            changed_cot=[tensor(np.ascontiguousarray(np.roll(value,shift,axis=0))) for value in cots]
            with torch.no_grad():
                for target,value in zip(gx,changed):target.copy_(tensor(value))
                for target,value in zip(gcot,changed_cot):target.copy_(value)
            compiled_graph.replay();torch.cuda.synchronize()
            direct_output,direct_loss,direct_grad=execute(compiled,changed,changed_cot)
            for name,a,b in zip(STATE_OUTPUTS,graph_output,direct_output):compare(f'compiled_graph_{shift}_{name}',a,b)
            for name,index,b in zip(GRAD_OUTPUTS,(0,1,3),direct_grad):compare(f'compiled_graph_{shift}_{name}',gx[index].grad,b)
            compare(f'compiled_graph_{shift}_loss',graph_loss,direct_loss,exact=False)
        guards('compiled_graph_diagnostics')
        record['compile_and_graph']=dict(backend='inductor',fullgraph=True,dynamic=False,
            automatic_cudagraphs=False,explicit_native_graph=True,explicit_compiled_forward_backward_graph=True,
            changed_input_replays=[1,7],fallback=False);save()

        import jax
        import jax.numpy as jnp
        from nablaengine import jax_ffi as jf
        plan=jf.build_plan(model,foot_pairs,info['arch'])
        relevant=('NQ','NV','NU','NB','MAXCON','NEFC','NFLOSS','REC_WORDS','OFF_FLAGS')
        defines={name:{key:re.findall(r'^#define '+key+r' (.+)$',source,re.M)
                      for key in relevant if re.search(r'^#define '+key+r' ',source,re.M)}
                 for name,source in plan['units'].items()}
        record['native_compile_plan']=dict(arch=info['arch'],defines=plan['defines'],
            unit_arch=plan['unit_arch'],arch_flags=plan['arch_flags'],forward_defines=plan['forward_defines'],
            generated_constants=defines,common_flags=jf.common_flags(info['arch']))
        check('generated_capacity_constants',all(f"#define NEFC {capacity['constraint_rows']}\n" in source and f"#define MAXCON {capacity['max_contacts']}\n" in source
              for name,source in plan['units'].items() if name!='dk_ffi.cu'))
        jax_info=jf.init(model=model,foot_pairs=foot_pairs)
        record['jax_library']={key:jax_info[key] for key in allowed if key in jax_info}
        check('identical_packed_model_constants',tf._STATE['consts'].dtype==jf._STATE['consts'].dtype and
              tf._STATE['consts'].shape==jf._STATE['consts'].shape and
              tf._STATE['consts'].tobytes()==jf._STATE['consts'].tobytes(),
              packed_float_count=int(tf._STATE['consts'].size))
        check('matching_interface_constants',all(getattr(tf,key)==getattr(jf,key) for key in (
            'ITERATIONS','SUBSTEPS','CLAMP_KV','NORMAL','CLAMP_SOFT','GRAD_DECAY','DIAG','VARIANTS',
            'OVERFLOW_CONTACTS','OVERFLOW_ROWS','NONFINITE_SHIFT')))
        jin=[jnp.asarray(value) for value in host]
        jf.read_diagnostics(reset=True);tf.read_diagnostics(reset=True)
        jfwd=jax.block_until_ready(jf.forward(*jin,tape=True,kinematics=True))
        tfwd=tf.forward(*tin,tape=True,kinematics=True)
        for name,a,b in zip(('qpos1','qvel1','warm1','info','tape',*KIN_OUTPUTS),tfwd,jfwd):compare('jax_torch_'+name,a,b,packed=name=='tape')
        jb=jax.block_until_ready(jf.backward(jfwd[4],jin[4],jin[5],jnp.asarray(cots[0]),jnp.asarray(cots[1]),
            kinematics=(jfwd[0],jfwd[1],*[jnp.asarray(value) for value in cots[2:]])))
        tb=tf.backward(tfwd[4],tin[4],tin[5],tcots[0],tcots[1],kinematics=(tfwd[0],tfwd[1],*tcots[2:]))
        for name,a,b in zip((*GRAD_OUTPUTS,'info'),tb,jb):compare('jax_torch_native_'+name,a,b)
        td=tf.read_diagnostics(reset=True);jd=jf.read_diagnostics(reset=True)
        check('jax_torch_native_diagnostics',td=={key:int(jd[key]) for key in tf.DIAG},torch=td,jax=jd)
        check('jax_native_diagnostics_finite_capacity',all(jd[key]==0 for key in BAD_DIAGNOSTICS))
        output,loss,grads=execute(loss_function,host,tcots)
        def jstep(q,v,u):
            values=jf.step(q,v,jin[2],u,jin[4],jin[5],**OPTIONS,kinematics=True)
            return (values[0],values[1],*values[3:])
        jout,pullback=jax.vjp(jstep,jin[0],jin[1],jin[3])
        jgrad=jax.block_until_ready(pullback(tuple(jnp.asarray(value) for value in cots)))
        for name,a,b in zip(GRAD_OUTPUTS,grads,jgrad):compare('jax_torch_autograd_'+name,a,b)
        tk=tf.kinematics(tin[0],tin[1],tin[5]);jk=jax.block_until_ready(jf.kinematics(jin[0],jin[1],jin[5]))
        for name,a,b in zip(KIN_OUTPUTS,tk,jk):compare('jax_torch_kinematics_'+name,a,b)
        kq=tin[0].clone().requires_grad_();kv=tin[1].clone().requires_grad_()
        kout=tf.kinematics(kq,kv,tin[5]);sum((a*b).sum() for a,b in zip(kout,tcots[2:])).backward()
        _,kvjp=jax.vjp(lambda q,v:jf.kinematics(q,v,jin[5]),jin[0],jin[1])
        kg=jax.block_until_ready(kvjp(tuple(jnp.asarray(value) for value in cots[2:])))
        compare('jax_torch_kinematics_grad_qpos',kq.grad,kg[0]);compare('jax_torch_kinematics_grad_qvel',kv.grad,kg[1])
        arrays('shared_native_state',(*tin,*tk),(*INPUTS,*KIN_OUTPUTS))
        arrays('torch_step_state',output,STATE_OUTPUTS)
        arrays('torch_step_vjp',grads,GRAD_OUTPUTS)
        np.savez(root/'jax_step_vjp.npz',**dict(zip(GRAD_OUTPUTS,[np.asarray(value) for value in jgrad])))
        guards('final_torch_diagnostics')
        final_jax=jf.read_diagnostics(reset=True)
        check('final_jax_diagnostics',all(final_jax[key]==0 for key in BAD_DIAGNOSTICS),diagnostics=final_jax)
        record['memory']=dict(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),jax_memory=jax.devices()[0].memory_stats(),
            framework_fraction_limits=dict(torch=args.torch_memory_fraction,jax=args.jax_memory_fraction))
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        record['elapsed_s']=time.monotonic()-started
        save()
        if private_tmp:shutil.rmtree(private_tmp,ignore_errors=True)


if __name__=='__main__':main()
