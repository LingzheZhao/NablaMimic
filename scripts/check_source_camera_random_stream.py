"""Compare source camera RNG consumption in eager, compiled and CUDA graph calls.

Generated optical depth exercises the actual noise and camera-input switch.
No model, raycast or physics step is constructed. Random-stream agreement is
measured within each execution mode, without requiring eager/compiled identity.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from nablamimic.source_depth_noise import SourceDepthNoise
from nablamimic.source_perceptive_env import SourcePerceptiveEnv
from nablamimic.torch_execution import CudaGraphCall, compile_fullgraph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--replays',type=int,default=3)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.batch_size<1 or args.replays<1:
        raise ValueError('Batch size and replay count must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('This generated-input diagnostic requires provisioned CUDA')
    args.output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda',torch.cuda.current_device())
    optical=torch.linspace(.05,2.,27*48,device=device).reshape(1,27,48,1).expand(args.batch_size,-1,-1,-1).contiguous()
    noise=SourceDepthNoise(device)
    torch.manual_seed(args.seed)
    initial_cpu=torch.get_rng_state().clone()
    initial_cuda=torch.cuda.get_rng_state(device).clone()
    def restore():
        torch.set_rng_state(initial_cpu)
        torch.cuda.set_rng_state(initial_cuda,device)
    evidence={}
    cases={}
    failures=[]
    for execution in ('eager','compile','compile-graph'):
        for condition in ('measured','constant-far'):
            name=f'{execution}/{condition}'
            try:
                env=SourcePerceptiveEnv.__new__(SourcePerceptiveEnv)
                env.cfg=SimpleNamespace(source_camera_input=condition,source_depth_gradient=False)
                # The real depth_image switch consumes the complete source noise.
                env._depth_image=lambda value:noise.noised(value)
                function=env.depth_image
                graph_record=None
                if execution=='compile':
                    function=compile_fullgraph(function)
                    function(optical)
                    torch.cuda.synchronize(device)
                elif execution=='compile-graph':
                    function=CudaGraphCall(function,(optical,),restore)
                    graph_record=function.record
                restore()
                rows=[]
                tensors=[]
                for replay in range(args.replays):
                    before=torch.cuda.get_rng_state(device).clone()
                    image=function(optical)
                    torch.cuda.synchronize(device)
                    after=torch.cuda.get_rng_state(device).clone()
                    sentinel=torch.rand(16,device=device).cpu()
                    constant=bool(torch.equal(image,torch.ones_like(image)))
                    finite=bool(torch.isfinite(image).all())
                    if not finite or (condition=='constant-far' and not constant):
                        raise RuntimeError('Camera input is nonfinite or differs from declared constant far')
                    rows.append(dict(replay=replay,rng_advanced=not torch.equal(before,after),
                        constant_far=constant,image_finite=finite,image_mean=float(image.mean())))
                    tensors.append(dict(before=before,after=after,sentinel=sentinel))
                evidence[name]=tensors
                cases[name]=dict(readings=rows,graph=graph_record)
            except Exception as error:
                failures.append(dict(execution=execution,condition=condition,
                                     error_type=type(error).__name__,error=str(error)))
    comparisons={}
    for execution in ('eager','compile','compile-graph'):
        left=evidence.get(f'{execution}/measured')
        right=evidence.get(f'{execution}/constant-far')
        if left is not None and right is not None:
            comparisons[execution]=[dict(replay=index,
                rng_before_equal=torch.equal(a['before'],b['before']),
                rng_after_equal=torch.equal(a['after'],b['after']),
                next_random_sentinel_equal=torch.equal(a['sentinel'],b['sentinel']))
                for index,(a,b) in enumerate(zip(left,right))]
    torch.save(evidence,args.output/'rng_evidence.pt')
    (args.output/'result.json').write_text(json.dumps(dict(
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        gpu_model=torch.cuda.get_device_name(device),batch_size=args.batch_size,replays=args.replays,seed=args.seed,
        input='Generated optical metres .05..2, 27x48; actual SourceDepthNoise and SourcePerceptiveEnv.depth_image',
        output_shape=[args.batch_size,1,18,32],physics_steps=0,
        random_support='Each call followed by the same-size external random sentinel; paired cases reset identical RNG',
        cases=cases,comparisons=comparisons,failed_conditions=failures,skipped_conditions=[]),indent=2))
    if failures:
        raise RuntimeError('One or more generated camera execution conditions failed; see result.json')


if __name__=='__main__':
    main()
