"""Fit source transition sequences and read matched physical forecast baselines."""
import argparse
import json
from pathlib import Path
import pickle
import random as python_random

import numpy as np
import torch
import yaml

from nablamimic.source_world_model import SourceWorldModel, state_loss_components, increment_targets, invariant_features, so3_exp
from nablamimic.source_world_model_vision import SourceWorldModelVision
from nablamimic.torch_env import DT, quat_to_matrix


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


def constant_velocity_prediction(state, control_dt_s, controls):
    """Persist world velocities and advance position, body rotation and joints."""
    position, rotation, velocity, angular_velocity, joints, joint_velocity = state
    duration = control_dt_s * controls
    body_angular_velocity = (rotation.transpose(-1, -2) @ angular_velocity.unsqueeze(-1)).squeeze(-1)
    return (position + velocity * duration, rotation @ so3_exp(body_angular_velocity * duration),
            velocity, angular_velocity, joints + joint_velocity * duration, joint_velocity)


def physical_forecast_metrics(prediction, target):
    values = state_loss_components(prediction, target)
    relative = prediction[1].transpose(-1, -2) @ target[1]
    sine = .5 * torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
        relative[..., 0, 2]-relative[..., 2, 0], relative[..., 1, 0]-relative[..., 0, 1]), -1).norm(dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*.5).clamp(-1, 1)
    values['rotation_mean_rad'] = torch.atan2(sine, cosine)
    values['position_mean_m'] = (prediction[0]-target[0]).norm(dim=-1)
    return values


def window_reading_metadata(dataset, indices, batch, env, phase_boundaries):
    """Bind selected window indices to recorded episodes and reference phase."""
    frames = batch['frame'][:, 0].detach().cpu().tolist()
    starts, lengths = env.starts.cpu().tolist(), env.lengths.cpu().tolist()
    result = []
    for index, frame in zip(indices, frames):
        episode_index, start, stop = map(int, dataset.window_index[int(index)])
        episode = dataset.episode_metadata[episode_index]
        motion = episode['clip_index']
        phase = (frame-starts[motion])/(lengths[motion]-1)
        if not 0 <= phase <= 1:
            raise ValueError('Selected source window frame is outside its declared reference')
        phase_bin = min(int(np.searchsorted(phase_boundaries, phase, side='right'))-1,
                        len(phase_boundaries)-2)
        result.append(dict(window_index=int(index), episode_index=episode_index,
            episode_seed=episode['seed'], motion_index=motion, reference_clip=episode['reference_clip'],
            window_start=start, window_stop=stop, initial_frame=int(frame),
            initial_reference_phase=phase, phase_bin=phase_bin))
    return result


def summarize_forecasts(matrix, columns, indices):
    if not indices:
        return None
    means = matrix[np.asarray(indices)].mean(0)
    result = {}
    for (group, metric), value in zip(columns, means):
        result.setdefault(group, {})[metric] = float(value)
    for group in ('model', 'persistence', 'constant_velocity'):
        result[group]['joint_rmse_rad'] = float(np.sqrt(result[group]['joint_position']))
    return result


def stratified_forecasts(matrix, columns, metadata, phase_boundaries):
    motions = sorted({row['motion_index'] for row in metadata})
    phases = range(len(phase_boundaries)-1)
    groups = dict(motion={str(m): [i for i, row in enumerate(metadata) if row['motion_index'] == m]
                          for m in motions},
                  phase={str(p): [i for i, row in enumerate(metadata) if row['phase_bin'] == p] for p in phases},
                  motion_phase={f'{m}/{p}': [i for i, row in enumerate(metadata)
                      if row['motion_index'] == m and row['phase_bin'] == p] for m in motions for p in phases})
    return {kind: {name: dict(count=len(indices), window_indices=[metadata[i]['window_index'] for i in indices],
                              metrics=summarize_forecasts(matrix, columns, indices))
                   for name, indices in rows.items()} for kind, rows in groups.items()}


@torch.no_grad()
def measure(model, vision, batch, context, condition, study, *, control_dt_s, window_metadata):
    initial_state = state_of(batch, 0)
    state = initial_state
    latent = (encode_time(vision, batch, 0) if condition == 'latent'
              else state[0].new_empty((state[0].shape[0], 0)))
    persistence_latent = latent.clone()
    if len(window_metadata) != state[0].shape[0]:
        raise ValueError('Every measured forecast needs its selected window metadata')
    readings = {}
    for time in range(max(study['forecast_horizons'])):
        scan, physics = context(state)
        state, _, latent = model(state, batch['pd_target'][:, time], scan, physics, latent)
        horizon = time+1
        if horizon not in study['forecast_horizons']:
            continue
        target = state_of(batch, horizon)
        groups = dict(model=physical_forecast_metrics(state, target),
            persistence=physical_forecast_metrics(initial_state, target),
            constant_velocity=physical_forecast_metrics(
                constant_velocity_prediction(initial_state, control_dt_s, horizon), target))
        values = groups['model']
        if condition == 'latent':
            target_latent = encode_time(vision, batch, horizon)
            values['latent_mse'] = (latent-target_latent).square().mean(-1)
            values['current_encoder_latent_persistence_mse'] = (persistence_latent-target_latent).square().mean(-1)
            variance = target_latent.var(dim=0, unbiased=False).mean().clamp_min(study['latent_variance_floor'])
            values['current_encoder_target_variance'] = variance.expand(state[0].shape[0])
            values['latent_variance_normalized_mse'] = values['latent_mse']/variance
            values['latent_persistence_variance_normalized_mse'] = values['current_encoder_latent_persistence_mse']/variance
        else:
            difference = context.render(state)-batch['depth_clean'][:, horizon]
            values['rendered_noisy_to_clean_depth_l1'] = difference.abs().flatten(1).mean(-1)
        target_positions = batch['torso_state'][:, :horizon+1, :3]
        groups['target_motion'] = dict(
            root_displacement_m=(target[0]-initial_state[0]).norm(dim=-1),
            root_path_speed_m_s=target_positions.diff(dim=1).norm(dim=-1).sum(1)/(horizon*control_dt_s))
        columns = [(group, name) for group, metrics in groups.items() for name in metrics]
        matrix = torch.stack([groups[group][name] for group, name in columns], -1).cpu().numpy()
        if not np.isfinite(matrix).all():
            raise FloatingPointError('Nonfinite measured forecast values')
        summary = summarize_forecasts(matrix, columns, list(range(len(window_metadata))))
        readings[str(horizon)] = dict(**summary['model'],
            baselines={name: summary[name] for name in ('persistence', 'constant_velocity')},
            target_motion=summary['target_motion'],
            support=dict(window_count=len(window_metadata), requested_controls=horizon,
                         duration_s=horizon*control_dt_s, states_per_window=horizon+1),
            stratified=stratified_forecasts(matrix, columns, window_metadata, study['reading_phase_boundaries']),
            window_readings=dict(metadata=window_metadata, columns=[list(pair) for pair in columns], values=matrix.tolist()))
    readings['teacher_actor_action_mse'] = float(actor_regression_loss(vision, batch, study['fit_horizon']))
    readings['reconstruction_l1'] = float(vision.reconstruction_loss(encode_time(vision, batch, 0), batch['depth_clean'][:, 0]))
    return readings


def measure_replayed_camera_rng(model, vision, batch, context, condition, seed, study, **reading_inputs):
    devices = [batch['qpos'].device.index] if batch['qpos'].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed+1000)
        return measure(model, vision, batch, context, condition, study, **reading_inputs)


FIT_FORMAT = 'nablamimic_source_world_model_fit_v1'
PROGRESS_FIELDS = ('completed_updates', 'statistics_indices', 'heldout_indices', 'update_indices', 'training_records')


def save_fit_checkpoint(destination, model, vision, optimizer, sampler, configuration, binding, progress):
    """Publish one complete optimizer step and its current-format continuation state."""
    if destination.exists():
        raise FileExistsError(destination)
    state = dict(format=FIT_FORMAT, condition=configuration['vision'], model=model.state_dict(),
        vision=vision.state_dict(), optimizer=optimizer.state_dict(), resolved_fit_configuration=configuration,
        data_binding=binding, **progress,
        rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 numpy=np.random.get_state(), python=python_random.getstate(), sampler=sampler.bit_generator.state))
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    torch.save(state, temporary)
    temporary.replace(destination)


def restore_fit_checkpoint(source, model, vision, optimizer, sampler, configuration, binding):
    """Restore only the current producer's checked data/configuration binding."""
    state = torch.load(source, map_location=next(model.parameters()).device, weights_only=False)
    if state['format'] != FIT_FORMAT:
        raise ValueError('Current source world-model fit checkpoint required')
    if state['resolved_fit_configuration'] != configuration or state['data_binding'] != binding:
        raise ValueError('Fit resume configuration or data binding differs')
    progress = {name: state[name] for name in PROGRESS_FIELDS}
    if (progress['completed_updates'] != len(progress['training_records'])
            or progress['completed_updates'] != len(progress['update_indices'])):
        raise ValueError('Fit resume update and sampling history disagree')
    model.load_state_dict(state['model'])
    vision.load_state_dict(state['vision'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng']['torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['rng']['cuda']])
    np.random.set_state(state['rng']['numpy'])
    python_random.setstate(state['rng']['python'])
    sampler.bit_generator.state = state['rng']['sampler']
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-config', type=Path, required=True)
    parser.add_argument('--vision', choices=('render', 'latent'), required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=1000, help='Cumulative fit updates, including a restored prefix')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    study = yaml.safe_load(args.study_config.read_text())
    required = {'model_hidden', 'fit_horizon', 'forecast_horizons', 'statistics_windows',
        'contact_force_threshold_n', 'heightmap_resolution_m', 'scan_max_distance_m', 'margin_m', 'z_top_m',
        'actor_regression_weight', 'latent_statistics_rate', 'latent_variance_floor',
        'reading_phase_boundaries', 'checkpoint_every_updates'}
    if not isinstance(study, dict) or set(study) != required:
        raise ValueError('Study configuration fields differ from the declared schema')
    for name in ('fit_horizon', 'statistics_windows', 'checkpoint_every_updates'):
        if type(study[name]) is not int or study[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('model_hidden', 'forecast_horizons'):
        if not isinstance(study[name], list) or not study[name] or any(type(v) is not int or v < 1 for v in study[name]):
            raise ValueError(f'{name} must contain positive integers')
    if max(study['forecast_horizons']) > 32 or study['fit_horizon'] > 32:
        raise ValueError('Study horizon exceeds the producer 32-control windows')
    phases = study['reading_phase_boundaries']
    if (not isinstance(phases, list) or len(phases) < 2 or phases[0] != 0 or phases[-1] != 1
            or not np.isfinite(phases).all() or any(b <= a for a, b in zip(phases, phases[1:]))):
        raise ValueError('Reading phase boundaries must increase from zero through one')
    integer_fields = {'fit_horizon', 'statistics_windows', 'checkpoint_every_updates'}
    for name in required-integer_fields-{'model_hidden', 'forecast_horizons', 'reading_phase_boundaries'}:
        if not np.isfinite(study[name]) or study[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if (args.updates < 1 or args.batch_size < 1 or not np.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or study['latent_statistics_rate'] > 1):
        raise ValueError('Fit dimensions, learning rate or latent EMA rate are invalid')
    from nablamimic.source_world_model_data import SourceWorldModelDataset
    from nablamimic.source_world_model_render import SourceWorldModelContext
    from nablamimic.torch_train import build_environment
    from nablamimic import torch_nets
    from evaluate_source_perceptive import set_evaluation_conditions

    args.output.mkdir(parents=True, exist_ok=False)
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
    if training.manifest['clock']['control_dt_s'] != DT:
        raise ValueError('Dataset clock differs from the source task control interval')
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['observation'] != env.observation_contract or checkpoint['perception'] != env.perception_contract:
        raise ValueError('Source actor observation or perception binding differs')
    teacher = training.manifest['teacher']
    if (teacher['weights'] != 'ema' or teacher['completed_updates'] != checkpoint['completed_updates']
            or Path(teacher['path']).resolve() != args.checkpoint.resolve()
            or teacher['observation'] != env.observation_contract or teacher['perception'] != env.perception_contract):
        raise ValueError('Dataset teacher and selected source EMA checkpoint binding differ')
    fit_configuration = dict(vision=args.vision, seed=args.seed, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device_type=torch.device(args.device).type,
        study={key: value for key, value in study.items() if key != 'checkpoint_every_updates'})
    binding = dict(dataset=str(args.dataset.resolve()), dataset_manifest=training.manifest,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_updates=checkpoint['completed_updates'],
        observation=env.observation_contract, perception=env.perception_contract,
        physical_model=bindings['physical_model'], context=context.metadata,
        torch_version=str(torch.__version__), control_dt_s=training.manifest['clock']['control_dt_s'])
    actor = torch_nets.tree_from_jax(checkpoint['actor_ema'], env.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    python_random.seed(args.seed)
    vision = SourceWorldModelVision(actor).to(env.device)
    model = SourceWorldModel(scan_dim=context.scan_dim, physics_dim=context.physics_dim,
        latent_dim=32 if args.vision == 'latent' else 0, contact_dim=env.m.nbody-1,
        hidden=tuple(study['model_hidden'])).to(env.device)
    optimizer = torch.optim.Adam(list(model.parameters())+list(vision.parameters()), lr=args.learning_rate)
    sampler = np.random.default_rng(args.seed)
    if args.resume is None:
        selection = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
        statistics_indices = selection.integers(len(training), size=study['statistics_windows'])
        heldout_indices = selection.choice(len(heldout), size=min(args.batch_size, len(heldout)), replace=False)
        progress = dict(completed_updates=0, statistics_indices=statistics_indices.tolist(),
            heldout_indices=heldout_indices.tolist(), update_indices=[], training_records=[])
        statistics_batch = sample_batch(training, statistics_indices, env.device)
        initialize_training_statistics(model, vision, statistics_batch, context, args.vision, study)
    else:
        progress = restore_fit_checkpoint(args.resume, model, vision, optimizer, sampler, fit_configuration, binding)
    starting_updates = progress['completed_updates']
    if args.updates <= starting_updates:
        raise ValueError('Cumulative fit target must exceed the restored completed updates')
    test_batch = sample_batch(heldout, progress['heldout_indices'], env.device)
    metadata = window_reading_metadata(heldout, progress['heldout_indices'], test_batch, env, phases)
    reading_inputs = dict(control_dt_s=binding['control_dt_s'], window_metadata=metadata)
    protocol = dict(cli={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        environment_configuration=configuration, study_configuration=study, context=context.metadata,
        dataset_manifest=training.manifest, dataset_split=training.manifest['window_split'],
        fit_configuration=fit_configuration, starting_completed_updates=starting_updates,
        train_statistics_indices=progress['statistics_indices'], heldout_indices=progress['heldout_indices'],
        heldout_windows=metadata, phase_definition='Initial recorded reference frame relative to the full reference control span',
        phase_intervals='Left-inclusive and right-exclusive, with phase one included in the final interval',
        sampling='One saved NumPy Generator draws each training batch with replacement; independent SeedSequence([seed,1]) selects fixed training-statistics and heldout windows',
        source_noise_condition=dict(camera_noise_enabled=env.camera_noise is not None,
            definition='Render uses the bound camera pipeline; noisy-to-clean depth error includes enabled camera noise'),
        measurement_rng_seed=args.seed+1000, physical_condition='Fixed nominal model and actual PD targets')
    (args.output/'resolved_protocol.json').write_text(json.dumps(protocol, default=str, indent=2))
    reports, failed_conditions = {}, []
    failure = None
    final_checkpoint = args.output/f'{args.vision}.pt'
    try:
        initial = measure_replayed_camera_rng(model, vision, test_batch, context, args.vision, args.seed, study, **reading_inputs)
        while progress['completed_updates'] < args.updates:
            indices = sampler.integers(len(training), size=args.batch_size)
            batch = sample_batch(training, indices, env.device)
            optimizer.zero_grad(set_to_none=True)
            loss, reconstruction = sequence_loss(model, vision, batch, study['fit_horizon'], context,
                condition=args.vision, contact_force_threshold_n=study['contact_force_threshold_n'],
                latent_statistics_rate=study['latent_statistics_rate'], latent_variance_floor=study['latent_variance_floor'])
            regression = actor_regression_loss(vision, batch, study['fit_horizon'])
            loss = loss+study['actor_regression_weight']*regression
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite world-model training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(vision.parameters()), 1., error_if_nonfinite=True)
            optimizer.step()
            progress['completed_updates'] += 1
            progress['update_indices'].append(indices.tolist())
            progress['training_records'].append(dict(loss=float(loss.detach()), reconstruction=float(reconstruction.detach()),
                teacher_actor_action_mse=float(regression.detach()), gradient_norm=float(norm)))
            update = progress['completed_updates']
            if update % study['checkpoint_every_updates'] == 0 or update == args.updates:
                if update < args.updates:
                    save_fit_checkpoint(args.output/f'{args.vision}_fit_it{update:06d}.pt', model, vision, optimizer,
                                        sampler, fit_configuration, binding, progress)
                (args.output/'progress.json').write_text(json.dumps(dict(condition=args.vision,
                    completed_updates=update, latest=progress['training_records'][-1]), indent=2))
        save_fit_checkpoint(final_checkpoint, model, vision, optimizer, sampler, fit_configuration, binding, progress)
        final = measure_replayed_camera_rng(model, vision, test_batch, context, args.vision, args.seed, study, **reading_inputs)
        actor_name = f'{args.vision}_fit_actor.pkl'
        payload = dict(actor=torch_nets.tree_map(lambda value: value.detach().cpu().numpy(), vision.export_actor()),
            observation=env.observation_contract, perception=env.perception_contract,
            completed_updates=progress['completed_updates'], training_metadata=dict(parent_checkpoint=str(args.checkpoint),
                parent_weights='ema', world_model_updates=progress['completed_updates'],
                actor_teacher_regression_updates=progress['completed_updates'], vision_method=args.vision,
                dynamic_head_updates=0, ema=False))
        with (args.output/actor_name).open('xb') as handle:
            pickle.dump(payload, handle)
        reports[args.vision] = dict(initial=initial, initial_completed_updates=starting_updates, heldout=final,
            persistence={str(h): final[str(h)]['baselines']['persistence'] for h in study['forecast_horizons']},
            constant_velocity={str(h): final[str(h)]['baselines']['constant_velocity'] for h in study['forecast_horizons']},
            training=progress['training_records'], actor_checkpoints=dict(after_fit=actor_name),
            fit_checkpoint=final_checkpoint.name)
    except Exception as error:
        failure = error
        failed_conditions.append(dict(condition=args.vision, error_type=type(error).__name__, error=str(error)))
    protocol['update_indices'] = progress['update_indices']
    (args.output/'resolved_protocol.json').write_text(json.dumps(protocol, default=str, indent=2))
    (args.output/'result.json').write_text(json.dumps(dict(dataset=str(args.dataset), source_checkpoint=str(args.checkpoint),
        resume=str(args.resume) if args.resume is not None else None, seed=args.seed,
        starting_completed_updates=starting_updates, completed_updates=progress['completed_updates'],
        updates=args.updates, batch_size=args.batch_size, fit_horizon=study['fit_horizon'],
        forecast_horizons=study['forecast_horizons'], model_hidden=study['model_hidden'], control='actual PD target29',
        execution='Learned forecast, reconstruction and detached-feature teacher-action fitting',
        physical_binding=bindings['engine'], context=context.metadata, dataset_split=training.manifest['window_split'],
        source_noise_condition=protocol['source_noise_condition'], measurement_rng_seed=args.seed+1000,
        failed_conditions=failed_conditions, skipped_conditions=[], status='failed' if failure is not None else 'complete',
        heldout_window_indices=progress['heldout_indices'], heldout_windows=metadata,
        training_statistics_window_indices=progress['statistics_indices'],
        normalization='Train-only physical input/local increment statistics, current detached-encoder latent EMA, target-variance latent loss',
        contact_target=dict(definition='Non-world body net all-contact force magnitude threshold in any of four 200Hz substeps',
                            threshold_n=study['contact_force_threshold_n']),
        forecast_error_definition='Dimension-mean squared state errors and rotation chord error; Euclidean root position mean and global joint RMSE',
        baseline_definition='Same selected windows and horizons; persistence or world constant velocity, body-frame angular rotation and joint constant velocity',
        latent_variance_support='Each horizon uses all selected heldout windows for current-encoder target variance; strata share that denominator',
        conditions=reports), indent=2))
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
