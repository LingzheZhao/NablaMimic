"""Fit fresh source transition sequences and measure learned forecast gradients."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch
import yaml

from nablamimic.source_world_model import SourceWorldModel, state_loss_components, increment_targets, invariant_features
from nablamimic.source_world_model_vision import SourceWorldModelVision
from nablamimic.torch_env import quat_to_matrix


def state_of(batch, time):
    torso = batch['torso_state'][:, time]
    return (torso[:, :3], quat_to_matrix(torso[:, 3:7]), torso[:, 7:10],
            torso[:, 10:13], batch['qpos'][:, time, 7:], batch['qvel'][:, time, 6:])


def sample_batch(dataset, indices, device):
    rows = [dataset[int(index)] for index in indices]
    if any(not row['transition_valid'].all() for row in rows):
        raise ValueError('Selected sequence contains a recorded invalid transition; no samples were silently skipped')
    names = ('torso_state', 'qpos', 'qvel', 'policy_observation', 'goal',
             'depth_clean', 'depth_noisy', 'frame', 'actor_action', 'transition_valid', 'pd_target', 'substep_all_force')
    return {name: torch.as_tensor(np.stack([row[name] for row in rows]),
                                  device=device, dtype=torch.float32) for name in names}


def encode_time(vision, batch, time):
    return vision.encode(batch['policy_observation'][:, time], batch['goal'][:, time])[0]


@torch.no_grad()
def initialize_training_statistics(model, vision, batch, context, condition, study):
    features, increments = [], []
    for time in range(study['fit_horizon']):
        state = state_of(batch, time)
        scan, physics = context(state)
        features.append(torch.cat((invariant_features(state), batch['pd_target'][:, time], scan, physics), -1))
        increments.append(increment_targets(state, state_of(batch, time+1)))
    features = torch.cat(features)
    increments = torch.cat(increments)
    model.input_mean.copy_(features.mean(0))
    model.input_std.copy_(features.std(0, unbiased=False).clamp_min(1e-3))
    model.increment_scale.copy_(increments.std(0, unbiased=False).clamp_min(1e-3))
    if condition == 'latent':
        z_states = encoded_sequence(vision, batch, study['fit_horizon'])
        model.update_latent_statistics(z_states, batch['transition_valid'][:, :study['fit_horizon']].bool(),
                                       rate=study['latent_statistics_rate'])


def normalized_state_loss(model, components):
    scales = model.increment_scale.split((3,3,3,3,29,29))
    return sum(value.mean()/scale.square().mean() for value, scale in zip(components.values(), scales))


def actor_regression_loss(vision, batch, horizon):
    """Teacher-action regression with a detached current visual feature.

    The target is the frozen collector actor output before exploration noise.
    This is the declared NablaMimic teacher-action adaptation, independently
    optimized from the image reconstruction gradients.
    """
    observation = batch['policy_observation'][:, :horizon].flatten(0,1)
    goal = batch['goal'][:, :horizon].flatten(0,1)
    feature, normalized_goal, normalized_history = vision.encode(observation, goal)
    predicted = vision.actor_from_features(feature, normalized_goal, normalized_history)
    target = batch['actor_action'][:, :horizon].flatten(0,1).detach()
    return torch.nn.functional.mse_loss(predicted, target)


@torch.no_grad()
def encoded_sequence(vision, batch, horizon):
    depth = batch['depth_noisy'][:, :horizon+1]
    latent = vision.prediction_target(depth.flatten(0,1))
    return latent.reshape(depth.shape[0],horizon+1,-1)


def sequence_loss(model, vision, batch, horizon, context, *, condition, contact_force_threshold_n,
                  latent_statistics_rate=None, latent_variance_floor=None):
    state = state_of(batch, 0)
    latent = (encode_time(vision, batch, 0).detach() if condition == 'latent'
              else state[0].new_empty((state[0].shape[0], 0)))
    if condition == 'latent':
        if latent_statistics_rate is None or latent_variance_floor is None:
            raise ValueError('Latent training requires declared statistics rate and variance floor')
        target_latents = encoded_sequence(vision, batch, horizon)
        model.update_latent_statistics(target_latents, batch['transition_valid'][:, :horizon].bool(),
                                       rate=latent_statistics_rate)
        target_variance = target_latents[:,1:].var(dim=(0,1),unbiased=False).mean().clamp_min(latent_variance_floor)
    losses = []
    for time in range(horizon):
        scan, physics = context(state)
        state, contacts, latent = model(state, batch['pd_target'][:, time], scan, physics, latent)
        components = state_loss_components(state, state_of(batch, time+1))
        force = batch['substep_all_force'][:, time].norm(dim=-1).amax(1)
        contact_target = (force[:, 1:] > contact_force_threshold_n).to(contacts.dtype)
        contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(contacts, contact_target)
        latent_loss = contacts.new_zeros(())
        if condition == 'latent':
            target_latent = target_latents[:,time+1]
            latent_loss = (latent-target_latent).square().mean()/target_variance
        losses.append(normalized_state_loss(model, components) + contact_loss + latent_loss)
    encoded = encode_time(vision, batch, 0)
    reconstruction = vision.reconstruction_loss(encoded, batch['depth_clean'][:, 0])
    return torch.stack(losses).mean() + reconstruction, reconstruction


@torch.no_grad()
def measure(model, vision, batch, context, condition, study):
    state = state_of(batch, 0)
    latent = (encode_time(vision, batch, 0) if condition == 'latent'
              else state[0].new_empty((state[0].shape[0], 0)))
    persistence_latent = latent.clone()
    readings = {}
    for time in range(max(study['forecast_horizons'])):
        scan, physics = context(state)
        state, _, latent = model(state, batch['pd_target'][:, time], scan, physics, latent)
        if time+1 in study['forecast_horizons']:
            target = state_of(batch, time+1)
            values = {name: float(value.mean()) for name, value in state_loss_components(state, target).items()}
            if condition == 'latent':
                target_latent = encode_time(vision, batch, time+1)
                values['latent_mse'] = float((latent-target_latent).square().mean())
                values['current_encoder_latent_persistence_mse'] = float((persistence_latent-target_latent).square().mean())
                variance = target_latent.var(dim=0,unbiased=False).mean().clamp_min(study['latent_variance_floor'])
                values['current_encoder_target_variance'] = float(variance)
                values['latent_variance_normalized_mse'] = values['latent_mse']/float(variance)
                values['latent_persistence_variance_normalized_mse'] = values['current_encoder_latent_persistence_mse']/float(variance)
            else:
                values['rendered_noisy_to_clean_depth_l1'] = float((context.render(state)-batch['depth_clean'][:, time+1]).abs().mean())
            relative = state[1].transpose(-1,-2) @ target[1]
            sine = .5*torch.stack((relative[...,2,1]-relative[...,1,2],
                relative[...,0,2]-relative[...,2,0], relative[...,1,0]-relative[...,0,1]),-1).norm(dim=-1)
            cosine = ((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
            values['rotation_mean_rad'] = float(torch.atan2(sine,cosine).mean())
            values['joint_rmse_rad'] = float((state[4]-target[4]).square().mean().sqrt())
            values['position_mean_m'] = float((state[0]-target[0]).norm(dim=-1).mean())
            readings[str(time+1)] = values
    readings['teacher_actor_action_mse'] = float(actor_regression_loss(vision, batch, study['fit_horizon']))
    readings['reconstruction_l1'] = float(vision.reconstruction_loss(encode_time(vision, batch, 0),batch['depth_clean'][:, 0]))
    return readings



def measure_replayed_camera_rng(model, vision, batch, context, condition, seed, study):
    devices = [batch['qpos'].device.index] if batch['qpos'].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed+1000)
        return measure(model, vision, batch, context, condition, study)


def policy_gradient_reading(model, vision, batch, context, env, condition, study):
    """One 32-control actor update through learned dynamics, without execution.

    Task goals and proprioceptive histories are the recorded sequence context.
    This isolates the action-to-forecast gradient, without claiming closed-loop
    imagined proprioception or an improved executed policy.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vision.zero_grad(set_to_none=True)
    state = state_of(batch, 0)
    latent = (encode_time(vision, batch, 0).detach() if condition == 'latent'
              else state[0].new_empty((state[0].shape[0], 0)))
    for time in range(max(study['forecast_horizons'])):
        if condition == 'render':
            depth = context.render(state)
            observation = torch.cat((batch['policy_observation'][:, time, :744], depth.flatten(1)), -1)
            actor_feature = vision.encode(observation, batch['goal'][:, time])[0].detach()
        _, goal, history = vision.encode(batch['policy_observation'][:, time], batch['goal'][:, time])
        actor_feature = latent if condition == 'latent' else actor_feature
        action = env.bound_action(vision.actor_from_features(actor_feature, goal, history))
        control = env.ctrl_of(action, batch['frame'][:, time].to(torch.int32))
        scan, physics = context(state)
        state, _, latent = model(state, control, scan, physics, latent)
    loss = sum(value.mean() for value in state_loss_components(state, state_of(batch, max(study['forecast_horizons']))).values())
    loss.backward()
    actor_gradients = [p.grad for p in vision.actor_layers.parameters() if p.grad is not None]
    encoder_gradients = [p.grad for p in vision.encoder.parameters() if p.grad is not None]
    reading = dict(terminal_loss=float(loss.detach()),
        actor_gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in actor_gradients))) if actor_gradients else 0.,
        actor_gradient_finite=all(bool(torch.isfinite(g).all()) for g in actor_gradients),
        encoder_gradient_present=bool(encoder_gradients),
        support=f"{max(study['forecast_horizons'])} predicted controls; recorded nonvisual history/goals; actor latent input detached",
        simulator_execution=False)
    if (not actor_gradients or reading['actor_gradient_norm'] == 0.
            or not reading['actor_gradient_finite'] or reading['encoder_gradient_present']):
        raise RuntimeError('Policy gradient routing failed')
    optimizer = torch.optim.Adam(vision.actor_layers.parameters(), lr=study['actor_learning_rate'])
    optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-config', type=Path, required=True)
    parser.add_argument('--vision', choices=('render','latent'), required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    study = yaml.safe_load(args.study_config.read_text())
    required = {'model_hidden','fit_horizon','forecast_horizons','statistics_windows',
                'contact_force_threshold_n','actor_learning_rate','heightmap_resolution_m',
                'scan_max_distance_m','margin_m','z_top_m','actor_regression_weight',
                'latent_statistics_rate','latent_variance_floor'}
    if not isinstance(study,dict) or set(study) != required:
        raise ValueError('Study configuration fields differ from the declared schema')
    for name in ('fit_horizon','statistics_windows'):
        if type(study[name]) is not int or study[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('model_hidden','forecast_horizons'):
        if not isinstance(study[name],list) or not study[name] or any(type(v) is not int or v < 1 for v in study[name]):
            raise ValueError(f'{name} must contain positive integers')
    if max(study['forecast_horizons']) > 32 or study['fit_horizon'] > 32:
        raise ValueError('Study horizon exceeds the producer 32-control windows')
    for name in required-{'model_hidden','fit_horizon','forecast_horizons','statistics_windows'}:
        if not np.isfinite(study[name]) or study[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if args.updates < 1 or args.batch_size < 1:
        raise ValueError('Updates and batch size must be positive')
    if study['latent_statistics_rate'] > 1:
        raise ValueError('Latent statistics EMA rate must be at most one')
    from nablamimic.source_world_model_data import SourceWorldModelDataset
    from nablamimic.source_world_model_render import SourceWorldModelContext
    from nablamimic.torch_train import build_environment
    from nablamimic import torch_nets

    args.output.mkdir(parents=True, exist_ok=False)
    from evaluate_source_perceptive import set_evaluation_conditions
    configuration = json.loads(args.config.read_text())
    set_evaluation_conditions(configuration, 'source-reset')
    for key in ('terrain_model', 'terrain_model_contract'):
        configuration[key] = Path(configuration[key])
    env, bindings = build_environment(argparse.Namespace(**configuration), device=args.device)
    context = SourceWorldModelContext(env, heightmap_resolution=study['heightmap_resolution_m'],
        scan_max_distance=study['scan_max_distance_m'], margin=study['margin_m'], z_top=study['z_top_m'])
    training = SourceWorldModelDataset(args.dataset, split='train')
    heldout = training.split_view('heldout')
    if not len(training) or not len(heldout):
        raise ValueError('Both independently partitioned sequence splits must be nonempty')
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
        raise ValueError('Source actor observation or perception binding differs')
    actor = torch_nets.tree_from_jax(checkpoint['actor_ema'], env.device)
    random = np.random.default_rng(args.seed)
    # Predeclare identical sampled windows for both independently trained models.
    update_indices = random.integers(len(training), size=(args.updates, args.batch_size))
    statistics_indices = random.integers(len(training), size=study['statistics_windows'])
    statistics_batch = sample_batch(training, statistics_indices, env.device)
    heldout_indices = random.choice(len(heldout), size=min(args.batch_size, len(heldout)), replace=False)
    test_batch = sample_batch(heldout, heldout_indices, env.device)
    protocol = dict(cli={name:str(value) if isinstance(value,Path) else value for name,value in vars(args).items()},
        environment_configuration=configuration, study_configuration=study, context=context.metadata,
        dataset_manifest=training.manifest, dataset_split='Last20percent per episode, 32control gap; producer indices retained',
        train_statistics_indices=statistics_indices.tolist(), update_indices=update_indices.tolist(),
        heldout_indices=heldout_indices.tolist(), fit_horizon=study['fit_horizon'], forecast_horizons=study['forecast_horizons'],
        source_noise_condition=dict(camera_noise_enabled=env.camera_noise is not None,
            definition='Render uses bound environment camera pipeline; noisy-to-clean depth error includes enabled camera noise'),
        measurement_rng_seed=args.seed+1000,
        physical_condition='Fixed nominal model, actual PD targets, no randomized parameter vector')
    (args.output/'resolved_protocol.json').write_text(json.dumps(protocol, default=str, indent=2))
    reports = {}
    failed_conditions = []
    failure = None
    try:
        for condition in (args.vision,):
            torch.manual_seed(args.seed)
            vision = SourceWorldModelVision(actor).to(env.device)
            model = SourceWorldModel(scan_dim=context.scan_dim, physics_dim=context.physics_dim, latent_dim=32 if condition == 'latent' else 0,
                                     contact_dim=env.m.nbody-1, hidden=tuple(study['model_hidden'])).to(env.device)
            initialize_training_statistics(model, vision, statistics_batch, context, condition, study)
            initial = measure_replayed_camera_rng(model, vision, test_batch, context, condition, args.seed, study)
            persistence = {str(horizon): {name: float(value.mean()) for name, value in
                state_loss_components(state_of(test_batch, 0), state_of(test_batch, horizon)).items()}
                for horizon in study['forecast_horizons']}
            optimizer = torch.optim.Adam(list(model.parameters())+list(vision.parameters()), lr=args.learning_rate)
            losses = []
            for update, indices in enumerate(update_indices, 1):
                batch = sample_batch(training, indices, env.device)
                optimizer.zero_grad(set_to_none=True)
                loss, reconstruction = sequence_loss(model, vision, batch, study['fit_horizon'], context, condition=condition,
                    contact_force_threshold_n=study['contact_force_threshold_n'],
                    latent_statistics_rate=study['latent_statistics_rate'], latent_variance_floor=study['latent_variance_floor'])
                actor_regression = actor_regression_loss(vision, batch, study['fit_horizon'])
                loss = loss + study['actor_regression_weight']*actor_regression
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite world-model training loss')
                loss.backward()
                parameters = list(model.parameters())+list(vision.parameters())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(dict(loss=float(loss.detach()), reconstruction=float(reconstruction.detach()),
                                   teacher_actor_action_mse=float(actor_regression.detach()),
                                   gradient_norm=float(norm)))
                if update % 100 == 0 or update == args.updates:
                    (args.output/'progress.json').write_text(json.dumps(dict(
                        condition=condition, completed_updates=update, latest=losses[-1]), indent=2))
            final = measure_replayed_camera_rng(model, vision, test_batch, context, condition, args.seed, study)
            torch.save(dict(model=model.state_dict(), vision=vision.state_dict(), optimizer=optimizer.state_dict(),
                            condition=condition, completed_updates=args.updates,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            numpy_rng=random.bit_generator.state, sampling_step=args.updates,
                            update_indices=update_indices, heldout_indices=heldout_indices), args.output/f'{condition}.pt')
            actor_metadata = dict(parent_checkpoint=str(args.checkpoint), parent_weights='ema',
                world_model_updates=args.updates, actor_teacher_regression_updates=args.updates, vision_method=condition,
                pilot_objective=f"{study['fit_horizon']}-step state/contact fit with latent prediction only for latent condition, plus depth reconstruction and detached-feature teacher-action regression", ema=False)
            def save_actor(name, completed_updates):
                payload = dict(actor=torch_nets.tree_map(lambda value: value.detach().cpu().numpy(), vision.export_actor()), observation=env.observation_contract,
                    perception=env.perception_contract, completed_updates=completed_updates,
                    training_metadata={**actor_metadata, 'dynamic_head_updates':completed_updates})
                with (args.output/name).open('wb') as handle:
                    pickle.dump(payload, handle)
            save_actor(f'{condition}_fit_actor.pkl', 0)
            policy_batch = sample_batch(training, update_indices[-1], env.device)
            gradient_reading = policy_gradient_reading(model, vision, policy_batch, context, env, condition, study)
            gradient_reading['split'] = 'train'
            gradient_reading['window_indices'] = update_indices[-1].tolist()
            save_actor(f'{condition}_one_update_actor.pkl', 1)
            torch.save(dict(actor=vision.export_actor(), policy_gradient=gradient_reading),
                       args.output/f'{condition}_actor_one_update.pt')
            reports[condition] = dict(initial=initial, persistence=persistence, heldout=final,
                                     training=losses, policy_gradient=gradient_reading,
                                     actor_checkpoints=dict(after_fit=f'{condition}_fit_actor.pkl',
                                         after_one_dynamic_update=f'{condition}_one_update_actor.pkl'))
    except Exception as error:
        failure = error
        failed_conditions.append(dict(condition=args.vision, error_type=type(error).__name__, error=str(error)))
    (args.output/'result.json').write_text(json.dumps(dict(
        dataset=str(args.dataset), source_checkpoint=str(args.checkpoint), seed=args.seed,
        updates=args.updates, batch_size=args.batch_size, fit_horizon=study['fit_horizon'], forecast_horizons=study['forecast_horizons'],
        model_hidden=study['model_hidden'], control='actual PD target29', physics_condition='fixed nominal model; no randomized parameter vector',
        execution='Learned forecast and detached-feature teacher-action fitting plus one diagnostic dynamic head update; simulator execution is a separate reading',
        physical_binding=bindings['engine'], context=context.metadata,
        source_noise_condition=protocol['source_noise_condition'], measurement_rng_seed=args.seed+1000,
        failed_conditions=failed_conditions, skipped_conditions=[], status='failed' if failed_conditions else 'complete',
        heldout_window_indices=heldout_indices.tolist(), training_statistics_window_indices=statistics_indices.tolist(),
        normalization='Train-only sampled windows and fit horizon as configured; input std floor .001, local increment std floor .001; group state loss divided by mean increment variance',
        contact_target=dict(definition='Non-world body net all-contact force magnitude threshold in any of four 200Hz substeps', threshold_n=study['contact_force_threshold_n']),
        forecast_error_definition='Dimension-mean squared state errors; rotation matrix chord error; position_mean_m is Euclidean root error',
        conditions=reports), indent=2))
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
