"""Read-only package/compiler inspection; no simulator, GPU context or installation."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    versions={}
    for name in ('torch','triton','jax','jaxlib','numpy','optax','mujoco','mujoco-mjx',
                 'warp-lang','nablamimic','nablaengine','nvidia-cuda-runtime-cu12',
                 'nvidia-cuda-nvrtc-cu12','nvidia-cudnn-cu12','nvidia-cublas-cu12'):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    modules={}
    for name in ('torch','triton','jax','nablaengine','mujoco_warp'):
        spec=importlib.util.find_spec(name)
        modules[name]=spec.origin if spec else None
    result=dict(python=sys.executable,python_version=sys.version,versions=versions,modules=modules,
                env={key:os.environ.get(key) for key in ('NVCC','CUDA_HOME','NVCC_CCBIN','MENAGERIE_DIR','NABLA_ENGINE_CACHE_SEED')},
                gpu_execution='not_requested')
    if modules['torch']:
        try:
            import torch
            result['torch_import']=dict(version=torch.__version__,cuda_version=torch.version.cuda,
                compile_callable=callable(getattr(torch,'compile',None)),
                cuda_graph_api=hasattr(torch.cuda,'CUDAGraph'),cuda_initialized=torch.cuda.is_initialized())
        except Exception as error:result['torch_import_error']=repr(error)
    compiler=os.environ.get('NVCC') or shutil.which('nvcc')
    result['nvcc']=compiler
    if compiler:
        for flag in ('--version','--list-gpu-code'):
            try:
                completed=subprocess.run([compiler,flag],text=True,capture_output=True,timeout=20)
                result['nvcc'+flag]=dict(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
            except Exception as error:result['nvcc'+flag]=dict(error=repr(error))
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as handle:handle.write(encoded)
    print(encoded,end='')


if __name__=='__main__':main()
