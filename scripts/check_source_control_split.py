"""Provisioned-GPU value and gradient comparison of fused four and measured 3+1 substeps."""
import argparse
import json
from pathlib import Path

import torch
from nablaengine import torch_ffi as engine
from nablamimic.torch_train import build_environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',required=True)
    parser.add_argument('--absolute-tolerance',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text())
    for key in ('terrain_model','terrain_model_contract'):
        config[key]=Path(config[key])
    env,_=build_environment(argparse.Namespace(**config))
    frame=torch.tensor(args.frames,device=env.device,dtype=torch.int32)
    reset_noise={name:torch.zeros_like(value) for name,value in env.sample_reset_noise(len(frame)).items()}
    original=env.reset(frame,reset_noise)
    values=[]
    gradients=[]
    torque_error=None
    for split in (False,True):
        state={name:value.detach().clone() for name,value in original.items()}
        state['qpos'].requires_grad_()
        state['qvel'].requires_grad_()
        action=torch.zeros(len(frame),29,device=env.device,requires_grad=True)
        if split:
            stepped=env.step(state,action,torch.zeros(len(frame),93,device=env.device))
            torque_error=float((stepped['source_actual_torque']-stepped['source_reward_torque']).abs().max())
            outputs=(stepped['qpos'],stepped['qvel'],stepped['warm'],stepped['xpos'])
        else:
            result=engine.step(state['qpos'],state['qvel'],state['warm'],env.ctrl_of(action,frame).contiguous(),
                state['friction'],state['com'],iterations=env.cfg.solver_iterations,substeps=4,
                clamp_kv=0,normal=3,clamp_soft=None,grad_decay=1.,kinematics=True)
            outputs=(result[3],result[1],result[2],result[4])
        objective=outputs[0].square().mean()+outputs[1].square().mean()+outputs[3].square().mean()
        gradients.append([value.detach() for value in torch.autograd.grad(objective,(state['qpos'],state['qvel'],action))])
        values.append([value.detach() for value in outputs])
    value_errors=[float((a-b).abs().max()) for a,b in zip(*values)]
    gradient_errors=[float((a-b).abs().max()) for a,b in zip(*gradients)]
    matches=max(value_errors+gradient_errors+[torque_error])<=args.absolute_tolerance
    (args.output/'result.json').write_text(json.dumps(dict(frames=args.frames,value_absolute_errors=value_errors,
        gradient_absolute_errors=gradient_errors,
        action_vjp_absolute_max=[float(value[-1].abs().max()) for value in gradients],
        model_actuator_ctrllimited=env.m.actuator_ctrllimited.tolist(),
        model_actuator_ctrlrange=env.m.actuator_ctrlrange.tolist(),
        actual_analytic_torque_absolute_error=torque_error,
        absolute_tolerance=args.absolute_tolerance,matches=matches),indent=2))
    if not matches:
        raise SystemExit(1)


if __name__=='__main__':
    main()
