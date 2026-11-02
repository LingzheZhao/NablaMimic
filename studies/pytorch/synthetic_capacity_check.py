"""Manufactured-contact validation of the current Engine's CUDA capacity paths.

This exercises actual add_contact, contact Jacobian and Hessian code at contact
indices 0..95. Only collision discovery is replaced. It is not a test of natural
collision discovery, whole-step dynamics, the adjoint, or training performance.
The oracle uses MuJoCo's float64 point Jacobians and mass matrix independently.
Run only in a provisioned CUDA environment with the fixed Engine on PYTHONPATH.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

RTOL = 1e-4
ATOL = 1e-5
COUNTS = (0, 1, 31, 32, 63, 64, 65, 95, 96, 97)

FIXTURE_DECL = r'''
DKC int fixture_pairs[96];
DKC float fixture_pos[96 * 3], fixture_frame[96 * 9];
__device__ int fixture_requested;
'''
INJECT = r'''
  // Manufactured collision table: the real add_contact enforces capacity.
  if (lane == 0) {
    for (int c = 0; c < fixture_requested; c++) {
      int k = c % 96;
      add_contact(s, fixture_pairs[k], 0, -0.001f, fixture_pos + 3*k, fixture_frame + 9*k);
    }
  }
  __syncwarp();
'''
WRAPPER = r'''
__global__ void capacity_fixture(const float* qp, const float* ctrl, int count,
                                 float* out_j, float* out_m, float* out_h,
                                 unsigned long long* out_mask, int* out_nd, int* out_info) {
  extern __shared__ __align__(16) unsigned char memory[];
  Sim& s = *reinterpret_cast<Sim*>(memory);
  int lane = threadIdx.x & 31;
  for (int i=lane; i<NQ; i+=32) s.qpos[i] = qp[i];
  for (int i=lane; i<NV; i+=32) { s.qvel[i]=0.f; s.qacc[i]=0.f; }
  if (lane<3) s.com[lane]=0.f;
  if (lane==0) { s.fric=0.8f; s.unsupported=0; fixture_requested=count; }
  __syncwarp();
  substep_pre(s, ctrl, lane, nullptr);
  __syncwarp();
  const int ncon = min(s.ncon, MAXCON);
  // The model pose is strictly inside its limits. Capture actual row metadata
  // before giving the Hessian a controlled mix of active and inactive rows.
  if (lane==0) {
    out_info[0]=s.ncon; out_info[1]=s.nefc; out_info[2]=s.nefc_dof; out_info[3]=s.unsupported;
  }
  for (int e=lane; e<NTRI; e+=32) out_m[e]=s.M[e];
  for (int i=lane; i<96*3*NV; i+=32) out_j[i]=0.f;
  for (int c=lane; c<96; c+=32) { out_mask[c]=0; out_nd[c]=0; }
  __syncwarp();
  for (int c=0; c<ncon; c++) {
    if (lane<s.con_nd[c]) {
      int d=s.con_dof[c][lane];
      for (int f=0; f<3; f++) out_j[(c*3+f)*NV+d]=s.con_J[c][f][lane];
    }
    if (lane==0) { out_mask[c]=s.con_mask[c]; out_nd[c]=s.con_nd[c]; }
    for (int k=lane; k<4; k+=32) {
      int r=s.con_row[c]+k;
      s.efc_D[r]=0.7f+0.003f*(r%37);
      s.efc_jar[r]=(c%11==0 || r%7==0) ? 1.f : -1.f;
    }
  }
  __syncwarp();
WORLD_GROUPS
  hessian(s,ncon,world,groups,lane);
  __syncwarp();
  for (int e=lane; e<NTRI; e+=32) out_h[e]=s.L[e];
}
'''
HOST = r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"CUDA %s: %s\n",#call,cudaGetErrorString(e)); exit(2); } } while(0)
template<class T> std::vector<T> readf(const char* dir,const char* name,size_t n) {
  char path[4096]; snprintf(path,sizeof(path),"%s/%s",dir,name);
  FILE* f=fopen(path,"rb"); std::vector<T> v(n);
  if(!f || fread(v.data(),sizeof(T),n,f)!=n) { fprintf(stderr,"read %s\n",path); exit(2); }
  fclose(f); return v;
}
template<class T> T* up(const std::vector<T>& v) {
  T* d; CHECK(cudaMalloc(&d,v.size()*sizeof(T)));
  CHECK(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return d;
}
template<class T> void save(const char* dir,const char* name,int case_i,T* d,size_t n) {
  std::vector<T> v(n); CHECK(cudaMemcpy(v.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));
  char path[4096]; snprintf(path,sizeof(path),"%s/%02d_%s.bin",dir,case_i,name);
  FILE* f=fopen(path,"wb"); if(!f || fwrite(v.data(),sizeof(T),n,f)!=n) { fprintf(stderr,"write %s\n",path); exit(2); }
  fclose(f);
}
int main(int argc,char** argv) {
  if(argc!=2) return 2; const char* dir=argv[1];
  auto cf=readf<float>(dir,"constants.bin",NCONST); const float* p=cf.data();
LOAD_CONSTS
  auto pairs=readf<int>(dir,"pairs.bin",96);
  auto pos=readf<float>(dir,"positions.bin",96*3);
  auto frames=readf<float>(dir,"frames.bin",96*9);
  CHECK(cudaMemcpyToSymbol(fixture_pairs,pairs.data(),96*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(fixture_pos,pos.data(),96*3*sizeof(float)));
  CHECK(cudaMemcpyToSymbol(fixture_frame,frames.data(),96*9*sizeof(float)));
  float* qp=up(readf<float>(dir,"qpos.bin",NQ));
  float* ctrl=up(readf<float>(dir,"ctrl.bin",NV-6));
  float *j,*m,*h; int *nd,*info; unsigned long long* mask;
  CHECK(cudaMalloc(&j,96*3*NV*sizeof(float))); CHECK(cudaMalloc(&m,NTRI*sizeof(float))); CHECK(cudaMalloc(&h,NTRI*sizeof(float)));
  CHECK(cudaMalloc(&mask,96*sizeof(unsigned long long))); CHECK(cudaMalloc(&nd,96*sizeof(int))); CHECK(cudaMalloc(&info,4*sizeof(int)));
  CHECK(cudaFuncSetAttribute(capacity_fixture,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sizeof(Sim)));
  const int counts[]={COUNTS};
  for(int i=0;i<sizeof(counts)/sizeof(counts[0]);i++) {
    capacity_fixture<<<1,32,sizeof(Sim)>>>(qp,ctrl,counts[i],j,m,h,mask,nd,info);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    save(dir,"jacobian",i,j,96*3*NV); save(dir,"mass",i,m,NTRI); save(dir,"hessian",i,h,NTRI);
    save(dir,"mask",i,mask,96); save(dir,"support_size",i,nd,96); save(dir,"info",i,info,4);
  }
  printf("{\"cases\":%d,\"sizeof_sim\":%zu,\"cuda_status\":\"success\"}\n",(int)(sizeof(counts)/sizeof(counts[0])),sizeof(Sim));
  return 0;
}
'''


def write_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def build_source(model):
    import re
    from nablaengine.g1_engine import adjoint, source, step, tape
    from nablaengine.g1_engine.host import loader
    kernel=step.STEP_KERNEL
    begin='  // collision\n'
    end='  PROF(2);'
    assert kernel.count(begin)==1
    a=kernel.index(begin); b=kernel.index(end,a)
    injected=kernel[:a]+INJECT+kernel[b:]
    # Copy the current production grouping block verbatim; do not reimplement it.
    a=kernel.index('  unsigned world[CONTACT_TILES]')
    b=kernel.index('  // tape (rec): L_H',a)
    groups=kernel[a:b]
    maxd=source.constants(model)[1]
    device=(source.header(model,maxd,1)+step.EXTRA_HEADER+step.step_header(model)+tape.header(model)
            +source.DEVICE+step.STEP_DEVICE+FIXTURE_DECL+injected)
    consts,lays,sizes=adjoint.engine_constants(model,foot_pairs(model))
    load=loader(lays,sizes)
    load=re.sub(r'cudaMemcpyToSymbol\(([^;]+)\);',r'CHECK(cudaMemcpyToSymbol(\1));',load)
    host=HOST.replace('NCONST',str(len(consts))).replace('LOAD_CONSTS',load).replace('COUNTS',','.join(map(str,COUNTS)))
    return device+WRAPPER.replace('WORLD_GROUPS',groups)+host,consts


def foot_pairs(model):
    import numpy as np
    return np.asarray([p for p in range(model.npair) if '_foot' in model.pair(p).name
                       and model.pair(p).name.endswith('_floor')],np.int32)


def fixtures(model):
    import mujoco
    import numpy as np
    data=mujoco.MjData(model)
    qp=model.qpos0.copy()
    qp[:3]=[0.3,-0.2,1.2]
    quat=np.asarray([0.91,0.13,-0.21,0.29]); qp[3:7]=quat/np.linalg.norm(quat)
    for j in range(1,model.njnt):
        if model.jnt_limited[j]:
            lo,hi=model.jnt_range[j]
            qp[model.jnt_qposadr[j]]=(lo+hi)/2+0.1*(hi-lo)*np.sin(j)
    qp=qp.astype(np.float32)
    data.qpos[:]=qp
    mujoco.mj_forward(model,data)
    mass=np.zeros((model.nv,model.nv),np.float64)
    mujoco.mj_fullM(model,data,mass)
    world=[]; own=[]
    for p in range(model.npair):
        b1=int(model.geom_bodyid[model.pair_geom1[p]]); b2=int(model.geom_bodyid[model.pair_geom2[p]])
        if model.pair_dim[p]!=3 or b1==b2: continue
        (world if b1==0 or b2==0 else own).append(p)
    if len(world)<2 or len(own)<2: raise ValueError('Need multiple real model world/self pairs')
    # Alternation puts both groups in every tile. Repeated world bodies exercise
    # grouped accumulation; changing self pairs exercises distinct DOF supports.
    pairs=np.asarray([world[(c//2)%len(world)] if c%2==0 else own[(7*(c//2)+3)%len(own)]
                      for c in range(96)],np.int32)
    pos=np.empty((96,3),np.float32); frames=np.empty((96,3,3),np.float32)
    jac=np.empty((96,3,model.nv),np.float64); masks=np.empty(96,np.uint64)
    ordered_geoms=np.empty((96,2),np.int32)
    mus=np.empty(96,np.float32); foot=set(foot_pairs(model).tolist())
    for c,p in enumerate(pairs):
        g1=int(model.pair_geom1[p]); g2=int(model.pair_geom2[p])
        # The native contact convention orders supported primitive types as
        # plane, sphere, capsule, box (also their MuJoCo enum order). Its
        # Jacobian is body2 minus body1 in that order, even for swapped XML pairs.
        if int(model.geom_type[g1])>int(model.geom_type[g2]): g1,g2=g2,g1
        ordered_geoms[c]=[g1,g2]
        b1=int(model.geom_bodyid[g1]); b2=int(model.geom_bodyid[g2])
        pos[c]=(data.geom_xpos[g1]+data.geom_xpos[g2])/2+np.asarray([.01,-.02,.015])
        angle=.17*c+.31
        normal=np.asarray([.8*np.cos(angle),.8*np.sin(angle),.6])
        tangent=np.asarray([-np.sin(angle),np.cos(angle),0.])
        frames[c]=np.stack([normal,tangent,np.cross(normal,tangent)])
        j1=np.zeros((3,model.nv)); j2=j1.copy(); rot=j1.copy()
        mujoco.mj_jac(model,data,j1,rot,pos[c].astype(np.float64),b1)
        mujoco.mj_jac(model,data,j2,rot,pos[c].astype(np.float64),b2)
        jac[c]=frames[c].astype(np.float64)@(j2-j1)
        def ancestors(body):
            answer=set()
            while body>0: answer.add(body); body=int(model.body_parentid[body])
            return answer
        exclusive=ancestors(b1)^ancestors(b2)
        masks[c]=sum(1<<i for i in range(model.nv) if int(model.dof_bodyid[i]) in exclusive)
        mus[c]=.8 if p in foot else model.pair_friction[p,0]
    return dict(qpos=qp,ctrl=qp[7:].copy(),pairs=pairs,positions=pos,frames=frames,
                expected_jacobian=jac,expected_masks=masks,expected_mass=mass,friction=mus,ordered_geom_ids=ordered_geoms)


def dense_hessian(fixture,count):
    import numpy as np
    total=fixture['expected_mass'].copy()
    contributions=[]
    for c in range(count):
        normal,t1,t2=fixture['expected_jacobian'][c]; mu=float(fixture['friction'][c])
        rows=np.stack([normal+mu*t1,normal-mu*t1,normal+mu*t2,normal-mu*t2])
        weights=[]
        for k in range(4):
            r=4*c+k
            d=np.float32(np.float32(.7)+np.float32(.003)*np.float32(r%37))
            weights.append(0. if c%11==0 or r%7==0 else float(d))
        contribution=rows.T@(np.asarray(weights)[:,None]*rows)
        contributions.append(contribution);total+=contribution
    return total,contributions


def run(out,engine_capacity_revision,*,source_revision,source_root,engine_root):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',engine_capacity_revision=engine_capacity_revision,scope=__doc__,
                source_revision=source_revision,source_root=str(Path(source_root).resolve()),
                engine_root=str(Path(engine_root).resolve()),argv=list(sys.argv),script_path=str(Path(__file__).resolve()),
                tolerances=dict(rtol=RTOL,atol=ATOL),counts=list(COUNTS),checks=[],failures=[])
    write_json(out/'result.json',result)
    def check(name,passed,**meta):
        result['checks'].append(dict(name=name,passed=bool(passed),**meta))
        if not passed:result['failures'].append(name)
    def close(name,got,want):
        finite=bool(np.isfinite(got).all() and np.isfinite(want).all())
        error=float(np.max(np.abs(got-want),initial=0)) if finite else None
        check(name,finite and np.allclose(got,want,rtol=RTOL,atol=ATOL),max_abs=error)
    try:
        import numpy as np
        import mujoco
        import nablamimic.env as donor_env
        from nablaengine import cuda
        from nablaengine.g1_engine import step
        result['loaded_engine_path']=str(Path(step.__file__).resolve())
        result['loaded_donor_path']=str(Path(donor_env.__file__).resolve())
        if not Path(step.__file__).resolve().is_relative_to(Path(engine_root).resolve()):
            raise ValueError('Loaded Engine does not come from declared engine root')
        if not Path(donor_env.__file__).resolve().is_relative_to(Path(source_root).resolve()):
            raise ValueError('Loaded donor does not come from declared source root')
        if os.environ.get('DK_MAXCON') or os.environ.get('DK_ALLOW_UNSUPPORTED'):
            raise ValueError('No capacity or unsupported-model override is permitted')
        model,_=donor_env.build_model(donor_env.EnvConfig(solver_iterations=10,residual_action=True))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        assert (model.nq,model.nv,model.nu,model.nbody,model.npair)==(36,35,29,31,502)
        assert len(foot_pairs(model))==14
        assert step.maxcon(model)==96 and step.NEFC==416
        f=fixtures(model)
        np.savez(out/'fresh_fixtures.npz',**f)
        for name in ('qpos','ctrl','pairs','positions','frames'): f[name].tofile(out/(name+'.bin'))
        src,consts=build_source(model); (out/'capacity_fixture.cu').write_text(src);consts.tofile(out/'constants.bin')
        command=cuda.compile_command(out/'capacity_fixture.cu',out/'capacity_fixture',unit='forward')
        result['compile_command']=list(map(str,command));write_json(out/'result.json',result)
        with (out/'compile.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        with (out/'execute.log').open('w') as log:
            subprocess.run([str(out/'capacity_fixture'),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        result['device_run']=json.loads((out/'execute.log').read_text().strip().splitlines()[-1])
        tri=np.tril_indices(model.nv); ntri=len(tri[0])
        check('contact95_nonzero_independent_jacobian',float(np.linalg.norm(f['expected_jacobian'][95]))>1e-3)
        _,contributions=dense_hessian(f,96)
        check('contact95_nonzero_independent_hessian',float(np.linalg.norm(contributions[95]))>1e-3)
        for tile in range(3):
            p=f['pairs'][32*tile:32*(tile+1)]
            b1=model.geom_bodyid[model.pair_geom1[p]]; b2=model.geom_bodyid[model.pair_geom2[p]]
            isworld=(b1==0)|(b2==0)
            check(f'tile{tile}_world_and_self',np.any(isworld) and np.any(~isworld))
        for i,requested in enumerate(COUNTS):
            count=min(requested,96);prefix=f'count{requested}'
            def read(name,dtype,shape):return np.fromfile(out/f'{i:02d}_{name}.bin',dtype=dtype).reshape(shape)
            j=read('jacobian',np.float32,(96,3,model.nv));m=read('mass',np.float32,(ntri,))
            h=read('hessian',np.float32,(ntri,));mask=read('mask',np.uint64,(96,))
            nd=read('support_size',np.int32,(96,));info=read('info',np.int32,(4,))
            np.savez(out/f'{prefix}_device.npz',jacobian=j,mass=m,hessian=h,support_mask=mask,support_size=nd,info=info)
            check(prefix+'_counts_and_capacity_flags',np.array_equal(info,[requested,4*count,0,4 if requested>96 else 0]),info=info.tolist())
            expected_j=np.zeros_like(j,dtype=np.float64);expected_j[:count]=f['expected_jacobian'][:count]
            close(prefix+'_jacobian',j,expected_j)
            expected_mask=np.zeros(96,np.uint64);expected_mask[:count]=f['expected_masks'][:count]
            check(prefix+'_support_masks_exact',np.array_equal(mask,expected_mask))
            check(prefix+'_support_sizes_exact',np.array_equal(nd,[int(v).bit_count() for v in expected_mask]))
            close(prefix+'_mass',m,f['expected_mass'][tri])
            expected_h,_=dense_hessian(f,count)
            close(prefix+'_hessian',h,expected_h[tri])
        # Removing the final contact must change the computed CUDA Hessian by
        # precisely its independent contribution, within the same frozen margin.
        h95=np.fromfile(out/'07_hessian.bin',np.float32)
        h96=np.fromfile(out/'08_hessian.bin',np.float32)
        check('contact95_nonzero_device_hessian_increment',float(np.linalg.norm(h96-h95))>1e-3)
        close('contact95_hessian_increment',h96,h95.astype(np.float64)+contributions[95][tri])
        result['status']='failed' if result['failures'] else 'passed'
    except BaseException as error:
        result.update(status='failed',exception=repr(error));raise
    finally:
        write_json(out/'result.json',result)
    if result['failures']:raise RuntimeError('Synthetic capacity checks failed: '+', '.join(result['failures']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--engine-capacity-revision',required=True)
    parser.add_argument('--source-revision',required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--engine-root',type=Path,required=True)
    args=parser.parse_args()
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    run(args.out,args.engine_capacity_revision,source_revision=args.source_revision,
        source_root=args.source_root,engine_root=args.engine_root)


if __name__=='__main__':main()
