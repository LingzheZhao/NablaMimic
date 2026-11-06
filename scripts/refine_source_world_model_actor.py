"""Refine an actor head through frozen learned dynamics and predicted feedback."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch
import yaml

from evaluate_source_perceptive import set_evaluation_conditions
from nablamimic import torch_nets
from nablamimic.source_world_model import SourceWorldModel
from nablamimic.source_world_model_data import SourceWorldModelDataset
from nablamimic.source_world_model_imagination import SourceWorldModelImagination
from nablamimic.source_world_model_render import SourceWorldModelContext
from nablamimic.source_world_model_vision import SourceWorldModelVision
from nablamimic.torch_env import quat_to_matrix
from nablamimic.torch_train import build_environment


FORMAT = 'nablamimic_source_world_model_actor_v1'


def initial_batch(dataset, indices, device, horizon):
    rows = [dataset[int(index)] for index in indices]
    if any(len(row['pd_target']) < horizon or not row['state_valid'][0] for row in rows):
        raise ValueError('Selected initial state or available prediction horizon is invalid')
    def value(name):
        return torch.as_tensor(np.stack([row[name][0] for row in rows]),device=device,dtype=torch.float32)
    torso = value('torso_state')
    state = (torso[:,:3],quat_to_matrix(torso[:,3:7]),torso[:,7:10],torso[:,10:13],
             value('qpos')[:,7:],value('qvel')[:,6:])
    history = value('history')
    frame = value('frame').to(torch.int32)
    return state,history,history[:,-1,64:93],frame,torso.new_zeros((len(rows),3))


def validate_fit_inputs(checkpoint, dataset, env, bindings, context, study, dataset_path,
                        teacher_path, teacher_checkpoint):
    """Bind the current learned tensors to their actual producer inputs."""
    if checkpoint.get('format') != 'nablamimic_source_world_model_fit_v1':
        raise ValueError('Current source world-model fit format required')
    for field in ('model','vision','optimizer','condition','resolved_fit_configuration','data_binding'):
        if field not in checkpoint:
            raise ValueError(f'Fit checkpoint is missing {field}')
    condition = checkpoint['condition']
    configuration = checkpoint['resolved_fit_configuration']
    if condition not in ('render','latent') or configuration['vision'] != condition:
        raise ValueError('Fit visual condition differs from its model configuration')
    expected_study = {key:value for key,value in study.items() if key!='checkpoint_every_updates'}
    if configuration['study'] != expected_study:
        raise ValueError('Fit and actor refinement study settings differ')
    expected = dict(dataset=str(dataset_path.resolve()),dataset_manifest=dataset.manifest,
        checkpoint=str(teacher_path.resolve()),checkpoint_updates=teacher_checkpoint['completed_updates'],
        observation=env.observation_contract,perception=env.perception_contract,
        physical_model=bindings['physical_model'],context=context.metadata,
        torch_version=str(torch.__version__),control_dt_s=dataset.manifest['clock']['control_dt_s'])
    if checkpoint['data_binding'] != expected:
        raise ValueError('Fit inputs differ from the actual dataset, teacher, model, scene or observation bindings')
    teacher = dataset.manifest['teacher']
    if (teacher['weights']!='ema' or teacher['completed_updates']!=teacher_checkpoint['completed_updates']
            or Path(teacher['path']).resolve()!=teacher_path.resolve()
            or teacher['observation']!=env.observation_contract or teacher['perception']!=env.perception_contract):
        raise ValueError('Dataset teacher and selected original teacher differ')
    return condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','study-config','actor-config','dataset','world-model','teacher-checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--updates',type=int,required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    objective = yaml.safe_load(args.actor_config.read_text())
    required = {'horizon','learning_rate','gradient_norm_limit','body_position_weight','body_velocity_weight',
                'torso_orientation_weight','joint_position_weight','joint_velocity_weight','termination_weight','action_smoothness_weight','teacher_action_weight',
                'height_margin_start_m','gravity_margin_start','history_perturbation'}
    if set(objective)!=required or type(objective['horizon']) is not int or not 1<=objective['horizon']<=32:
        raise ValueError('Actor configuration differs from the declared finite-horizon protocol')
    if any(not np.isfinite(value) or value<0 for value in objective.values()):
        raise ValueError('Actor objective parameters must be finite and nonnegative')
    if objective['learning_rate']==0 or objective['gradient_norm_limit']==0 or args.updates<1 or args.batch_size<1:
        raise ValueError('Optimization dimensions and limits must be positive')
    if objective['height_margin_start_m']>=.25 or objective['gravity_margin_start']>=.8:
        raise ValueError('Termination penalty margins must precede the actual termination thresholds')
    study = yaml.safe_load(args.study_config.read_text())
    configuration = json.loads(args.config.read_text())
    set_evaluation_conditions(configuration,'source-reset')
    for key in ('terrain_model','terrain_model_contract'):
        configuration[key] = Path(configuration[key])
    args.output.mkdir(parents=True,exist_ok=False)
    env,bindings = build_environment(argparse.Namespace(**configuration),device=args.device)
    context = SourceWorldModelContext(env,heightmap_resolution=study['heightmap_resolution_m'],
        scan_max_distance=study['scan_max_distance_m'],margin=study['margin_m'],z_top=study['z_top_m'])
    with args.teacher_checkpoint.open('rb') as handle:
        teacher_checkpoint = pickle.load(handle)
    if teacher_checkpoint['observation']!=env.observation_contract or teacher_checkpoint['perception']!=env.perception_contract:
        raise ValueError('Frozen teacher and environment bindings differ')
    actor = torch_nets.tree_from_jax(teacher_checkpoint['actor_ema'],env.device)
    teacher = SourceWorldModelVision(actor).to(env.device)
    vision = SourceWorldModelVision(actor).to(env.device)
    checkpoint = torch.load(args.world_model,map_location=env.device,weights_only=False)
    dataset = SourceWorldModelDataset(args.dataset,split='train')
    if not len(dataset):
        raise ValueError('Actor refinement requires nonempty training initial states')
    condition = validate_fit_inputs(checkpoint,dataset,env,bindings,context,study,args.dataset,
                                    args.teacher_checkpoint,teacher_checkpoint)
    vision.load_state_dict(checkpoint['vision'])
    model = SourceWorldModel(scan_dim=context.scan_dim,physics_dim=context.physics_dim,
        latent_dim=32 if condition=='latent' else 0,contact_dim=env.m.nbody-1,
        hidden=tuple(study['model_hidden'])).to(env.device)
    model.load_state_dict(checkpoint['model'])
    imagination = SourceWorldModelImagination(env,context,model,vision,condition,objective,teacher=teacher)
    optimizer = torch.optim.Adam(vision.actor_layers.parameters(),lr=objective['learning_rate'])
    random = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    start = 0
    binding = dict(world_model=str(args.world_model.resolve()),teacher_checkpoint=str(args.teacher_checkpoint.resolve()),
                   dataset=str(args.dataset.resolve()),fit_configuration=checkpoint['resolved_fit_configuration'],
                   fit_data_binding=checkpoint['data_binding'],objective=objective,condition=condition,
                   observation=env.observation_contract,perception=env.perception_contract,study=study,context=context.metadata,
                   environment_configuration=configuration,physical_binding=bindings['engine'])
    if args.resume is not None:
        resumed = torch.load(args.resume,map_location=env.device,weights_only=False)
        if resumed['format']!=FORMAT or resumed['binding']!=binding:
            raise ValueError('Resume data and current learned-model/physical bindings differ')
        vision.load_state_dict(resumed['vision'])
        optimizer.load_state_dict(resumed['optimizer'])
        random.bit_generator.state = resumed['numpy_rng']
        torch.set_rng_state(resumed['torch_rng'].cpu())
        if resumed['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in resumed['cuda_rng']])
        start = resumed['completed_updates']
    protocol = dict(binding=binding,cli={name:str(value) if isinstance(value,Path) else value for name,value in vars(args).items()},
        context=context.metadata,study=study,environment_configuration=configuration,dataset_manifest=dataset.manifest,
        teacher_action_error_units=imagination.teacher_action_error_units,
        latent_regularizer_additional_render=imagination.teacher_requires_additional_render,
        feedback='Predicted native FK, proprioception, shifted history, current reference goals; only initial real state/history',
        objective='Finite-horizon pose/velocity, termination margin, smoothness and optional frozen teacher-action penalty; no critic or force reward')
    (args.output/'resolved_protocol.json').write_text(json.dumps(protocol,default=str,indent=2))
    def export_actor(path, completed_updates):
        exported = torch_nets.tree_map(lambda value:value.detach().cpu().numpy(),vision.export_actor())
        with path.open('wb') as handle:
            pickle.dump(dict(actor=exported,observation=env.observation_contract,perception=env.perception_contract,
                completed_updates=completed_updates,training_metadata=dict(method='Closed-loop learned-state finite-horizon actor head refinement',
                binding=binding,world_model_frozen=True,encoder_frozen=True,ema=False)),handle)
    export_actor(args.output/'initial_native_actor.pkl', start)
    diagnostic_indices = np.random.default_rng(args.seed+1000).integers(len(dataset),size=args.batch_size)
    diagnostic_initial = initial_batch(dataset,diagnostic_indices,env.device,objective['horizon'])
    device = next(vision.parameters()).device
    devices = [device.index] if device.type=='cuda' else []
    diagnostic_actions = []
    with torch.no_grad():
        for perturbation in (0.,objective['history_perturbation']):
            changed = list(diagnostic_initial)
            changed[1] = changed[1]+perturbation
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(args.seed+1000)
                _,_,predicted = imagination.rollout(*changed,objective['horizon'])
                diagnostic_actions.append(predicted['actions'])
    history_response = dict(perturbation=objective['history_perturbation'],
        action_absolute_max_change=float((diagnostic_actions[1]-diagnostic_actions[0]).abs().max()),
        window_indices=diagnostic_indices.tolist(),camera_rng_replayed=True)
    (args.output/'feedback_diagnostic.json').write_text(json.dumps(history_response,indent=2))
    records = []
    failure = None
    try:
        for update in range(start,start+args.updates):
            indices = random.integers(len(dataset),size=args.batch_size)
            initial = initial_batch(dataset,indices,env.device,objective['horizon'])
            optimizer.zero_grad(set_to_none=True)
            loss,terms,_ = imagination.rollout(*initial,objective['horizon'])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(vision.actor_layers.parameters(),objective['gradient_norm_limit'],error_if_nonfinite=True)
            if not torch.isfinite(loss) or not torch.isfinite(norm) or norm==0:
                raise FloatingPointError('Actor loss/gradient must be finite and gradient nonzero')
            if any(parameter.grad is not None for parameter in model.parameters()) or any(parameter.grad is not None for parameter in vision.encoder.parameters()):
                raise RuntimeError('Frozen world model or encoder acquired parameter gradients')
            optimizer.step()
            records.append(dict(completed_updates=update+1,loss=float(loss.detach()),gradient_norm=float(norm),
                terms={name:float(value.detach()) for name,value in terms.items()},window_indices=indices.tolist(),
                world_model_parameter_gradients=False,encoder_parameter_gradients=False))
            torch.save(dict(format=FORMAT,binding=binding,vision=vision.state_dict(),optimizer=optimizer.state_dict(),
                completed_updates=update+1,numpy_rng=random.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),args.output/'resume.partial.pt')
            (args.output/'resume.partial.pt').replace(args.output/'resume.pt')
            (args.output/'progress.json').write_text(json.dumps(records[-1],indent=2))
    except Exception as error:
        failure = error
    if failure is None:
        export_actor(args.output/'native_actor.pkl', start+args.updates)
    (args.output/'result.json').write_text(json.dumps(dict(status='failed' if failure else 'complete',updates=records,history_response=history_response,
        actor_checkpoint='native_actor.pkl' if failure is None else None,actual_execution=False,failed_conditions=[] if failure is None else [dict(error_type=type(failure).__name__,error=str(failure))],skipped_conditions=[]),indent=2))

    if failure is not None:
        raise failure


if __name__=='__main__':
    main()
