"""Check the camera-to-actor VJP through one real native control transition."""
import argparse
import json
import pickle
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[0,100,500,1000,2000,3000,4000])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from nablamimic import torch_nets as nets
    from nablamimic.torch_train import build_environment
    config=json.loads(args.config.read_text())
    if not config['source_depth_gradient']:
        raise ValueError('A current resolved source depth-gradient configuration is required')
    for name in ('terrain_model','terrain_model_contract'):
        config[name]=Path(config[name])
    env,bindings=build_environment(argparse.Namespace(**config))
    if config['task_inputs']!='instinct_perceptive_native':
        raise ValueError('This checker requires the source perceptive task')
    env.camera_noise=None
    with args.checkpoint.open('rb') as handle:
        checkpoint=pickle.load(handle)
    if checkpoint['observation']!=env.observation_contract or checkpoint['perception']!=env.perception_contract:
        raise ValueError('Checkpoint observation or perception binding differs')
    actor=nets.tree_from_jax(checkpoint['actor_ema'],env.device)
    frames=torch.tensor(args.frames,device=env.device,dtype=torch.int32)
    if not bool(env.valid[frames].all()):
        raise ValueError('Requested frames must be valid reference states')
    noise={name:torch.zeros_like(value) for name,value in env.sample_reset_noise(len(frames)).items()}
    state=env.reset(frames,noise)
    action=torch.zeros(len(frames),env.nu,device=env.device,requires_grad=True)
    stepped=env.step(state,action,torch.zeros(len(frames),93,device=env.device))
    history,goal=env.policy_obs(stepped,torch.zeros(len(frames),670,device=env.device))
    outputs=[]
    gradients=[]
    cotangent=torch.linspace(.5,1.5,env.nu,device=env.device)
    for enabled in (False,True):
        output=nets.actor_apply(actor,history,goal,differentiate_depth=enabled)
        outputs.append(output.detach().clone())
        gradient=torch.autograd.grad((output*cotangent).sum(),action,retain_graph=True)[0]
        gradients.append(gradient.detach().clone())
    depth=history[:,744:]
    pixel_weight=torch.linspace(.5,1.5,depth.shape[-1],device=env.device)
    image_gradient=torch.autograd.grad((depth*pixel_weight).sum(),action)[0].detach()
    # Both modes share exactly the same optical image tensor and stepped graph.
    forward_equal=torch.equal(outputs[0],outputs[1])
    difference=gradients[1]-gradients[0]
    finite=all(bool(torch.isfinite(value).all()) for value in (*gradients,image_gradient))
    threshold=torch.finfo(action.dtype).eps
    effect=bool((difference.abs().amax(-1)>threshold).any())
    rows=[]
    for index,frame in enumerate(args.frames):
        rows.append(dict(frame=frame,
            stopped_image_action_vjp_absolute_max=float(gradients[0][index].abs().max()),
            enabled_image_action_vjp_absolute_max=float(gradients[1][index].abs().max()),
            image_path_action_vjp_difference_absolute_max=float(difference[index].abs().max()),
            weighted_image_action_vjp_absolute_max=float(image_gradient[index].abs().max())))
    result=dict(checkpoint=str(args.checkpoint),weights='ema',completed_updates=checkpoint['completed_updates'],
        reset=bindings['reset'],reset_noise='zero',goal_noise='zero',proprioception_noise='zero',sensor_noise='off',
        action='zero in the configured absolute or residual action domain',
        physics_controls=1,shared_stepped_graph=True,shared_depth_tensor=True,
        future_action_forward_bitwise_equal=forward_equal,vjps_finite=finite,
        difference_threshold=float(threshold),at_least_one_image_path_effect=effect,
        frames=rows,passed=forward_equal and finite and effect)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    import numpy as np
    np.savez(args.output/'arrays.npz',depth=depth.detach().cpu().numpy(),
        future_action_stopped=outputs[0].cpu().numpy(),future_action_enabled=outputs[1].cpu().numpy(),
        action_vjp_stopped=gradients[0].cpu().numpy(),action_vjp_enabled=gradients[1].cpu().numpy(),
        image_action_vjp=image_gradient.cpu().numpy())
    if not result['passed']:
        raise SystemExit('Camera-control gradient integration condition failed')


if __name__=='__main__':
    main()
