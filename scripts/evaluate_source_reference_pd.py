"""Run all ten references through the source task's declared joint PD action map.

This is a reference-joint-PD baseline, using the current reference frame only.
Each clip starts with one nominal reset and then follows one physical state chain.
Execute on provisioned compute. The --self-test checks generated action algebra.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nablamimic.source_evaluation_metrics import summarize_source_chain


def reference_action(target, default, scale, action_limit):
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Declared source action scales must be finite and positive')
    requested = (target-default)/scale
    executed = np.clip(requested, -action_limit, action_limit)
    return requested, executed, default+scale*executed


def self_test():
    default = np.linspace(-.4, .4, 29)
    scale = np.linspace(.05, .6, 29)
    action = np.linspace(-19., 19., 29)
    target = default+scale*action
    requested, executed, actual = reference_action(target, default, scale, 20.)
    np.testing.assert_allclose(requested, action, atol=2e-14)
    np.testing.assert_allclose(executed, action, atol=2e-14)
    np.testing.assert_allclose(actual, target, atol=2e-14)
    target[0] = default[0]+scale[0]*21
    requested, executed, actual = reference_action(target, default, scale, 20.)
    assert requested[0] > 20 and executed[0] == 20 and actual[0] != target[0]
    print(json.dumps(dict(self_test='passed', action_limit=20., unrepresentable_target_detected=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--code-revision')
    parser.add_argument('--record-substeps', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.config is None or args.output is None or args.code_revision is None:
        parser.error('--config, --output and --code-revision are required')
    import torch
    from nablaengine import torch_ffi as engine
    from nablamimic.torch_train import build_environment

    config = json.loads(args.config.read_text())
    if config['task_inputs'] != 'instinct_perceptive_native':
        raise ValueError('This baseline requires the source perceptive task')
    if config['action_limit'] != 20.:
        raise ValueError('This declared baseline uses the source normalized action limit 20')
    config.update(clips=['all'], source_reset_profile='nominal', source_depth_noise=False)
    for key in ('terrain_model', 'terrain_model_contract'):
        config[key] = Path(config[key])
    env, bindings = build_environment(argparse.Namespace(**config))
    if len(env.starts) != 10 or env.cfg.dr or env.cfg.reset_noise:
        raise ValueError('Expected ten clips with nominal physical parameters and nominal resets')
    body_names = bindings['physical_model']['tracked_body_names']
    torso_index = body_names.index('torso_link')
    joint_names = [env.m.joint(int(joint)).name for joint in env.m.actuator_trnid[:, 0]]
    default = env.q0.cpu().numpy()
    scale = env.action_scale.cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        state='started', baseline='reference-joint-PD', code_revision=args.code_revision,
        configuration=str(args.config), reference_directory=str(config['motion_lib']),
        control_dt_s=4*float(env.m.opt.timestep), solver_iterations=env.cfg.solver_iterations,
        controller='At each control use only the current reference joint positions: (reference-default)/action_scale',
        termination='SourcePerceptiveEnv.terminated at every physical state, including the terminal state',
        substep_recording=dict(enabled=args.record_substeps, sampling_hz=1/float(env.m.opt.timestep),
            time_support='Physics input t_k before implicitfast integration',
            body_names=[env.m.body(i).name for i in range(env.m.nbody)],
            torque_joint_names=joint_names, wrench_units='world force N and moment about world origin N m',
            torque_units='executed generalized actuator torque N m'),
        initialization='One nominal reference-state reset at the first frame of each clip',
        normalized_action_limit=20., unrepresentable_targets='Report and execute the declared bounded action',
        physical_randomization='nominal prepared source model', reset=bindings['reset'], reset_noise='zero', proprioception_noise='zero',
        policy_weights=None, perception_used=False, tracked_body_names=body_names, joint_names=joint_names,
        source_default_joint_position=default.tolist(), source_action_scale=scale.tolist(),
        physical_model=bindings['physical_model'], observation=env.observation_contract, reports=[])

    def write_report():
        (args.output/'result.json').write_text(json.dumps(report, indent=2))

    write_report()
    with torch.no_grad():
        for clip, start in enumerate(env.starts):
            output = args.output/f'clip_{clip:02d}'
            output.mkdir()
            engine.read_diagnostics(reset=True)
            frame = torch.tensor([int(start)], device=env.device, dtype=torch.int32)
            noise = {name: torch.zeros_like(value) for name, value in env.sample_reset_noise(1).items()}
            state = env.reset(frame, noise)
            initial_state_error=dict(
                qpos_reference_absolute_max=float((state['qpos']-env.ref_qpos[frame]).abs().max()),
                qvel_reference_absolute_max=float((state['qvel']-env.ref_qvel[frame]).abs().max()),
                tracked_body_world_position_error_max_m=float(torch.linalg.vector_norm(
                    state['xpos'][:,env.body_ids]-env.ref_body_pos[frame],dim=-1).max()))
            requested_controls = int(env.lengths[clip])-1
            rows, errors, terminations, positions, up_dots, substep_rows = [], [], [], [], [], []
            reason = 'requested_controls'
            failure = None

            def record_state():
                actual = state['xpos'][:, env.body_ids]
                reference = env.ref_body_pos[state['frame']]
                errors.append(torch.linalg.vector_norm(actual-reference, dim=-1)[0].cpu().numpy())
                positions.append(actual[0].cpu().numpy())
                terminations.append(bool(env.terminated(state).item()))
                quaternion = state['xquat'][0, env.torso_id]
                up_dots.append(float((1-2*(quaternion[1]**2+quaternion[2]**2)).item()))

            record_state()
            try:
                for control in range(requested_controls):
                    if terminations[-1]:
                        reason = 'tracking_termination'
                        break
                    target = env.ref_qpos[state['frame'], 7:]
                    action = (target-env.q0)/env.action_scale
                    bounded_action = env.bound_action(action)
                    actual_target = env.ctrl_of(bounded_action, state['frame'])
                    observation_noise = torch.zeros((1, 93), device=env.device)
                    if args.record_substeps:
                        stepped, substeps = env.step_recorded(state, action, observation_noise)
                        substep_rows.append({name: value[0].cpu().numpy() for name, value in substeps.items()})
                    else:
                        stepped = env.step(state, action, observation_noise)
                    done = bool(env.terminated(stepped).item())
                    ended = bool(env.clip_ended(stepped).item())
                    rows.append(dict(
                        qpos=state['qpos'][0].cpu().numpy(), qvel=state['qvel'][0].cpu().numpy(),
                        action=stepped['prev_a'][0].cpu().numpy(), requested_action=action[0].cpu().numpy(),
                        requested_joint_target=target[0].cpu().numpy(), executed_joint_target=actual_target[0].cpu().numpy(),
                        frame=int(state['frame'][0].item()), tracking_terminated=done, clip_ended=ended))
                    state = stepped
                    record_state()
                    if done or ended:
                        reason = 'tracking_termination' if done else 'reference_complete'
                        break
            except Exception as error:
                reason = 'execution_failure'
                failure = dict(type=type(error).__name__, message=str(error))
            controls = len(rows)
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else dict(
                qpos=np.empty((0, env.m.nq)), qvel=np.empty((0, env.m.nv)), action=np.empty((0, env.nu)),
                requested_action=np.empty((0, env.nu)), requested_joint_target=np.empty((0, env.nu)),
                executed_joint_target=np.empty((0, env.nu)), frame=np.empty(0, dtype=np.int32),
                tracking_terminated=np.empty(0, dtype=bool), clip_ended=np.empty(0, dtype=bool))
            substep_arrays = {}
            if substep_rows:
                substep_arrays = {f'substep_{key}': np.stack([row[key] for row in substep_rows]) for key in substep_rows[0]}
                substep_arrays['substep_time_s'] = (np.arange(controls)[:, None]*4+np.arange(4)[None])*env.m.opt.timestep
            np.savez(output/'trajectory.npz', **arrays, **substep_arrays,
                final_qpos=state['qpos'].cpu().numpy(), final_qvel=state['qvel'].cpu().numpy(),
                final_frame=state['frame'].cpu().numpy(), world_body_error_m=np.asarray(errors),
                torso_error_m=np.asarray(errors)[:, torso_index], body_world_position=np.asarray(positions),
                torso_updot=np.asarray(up_dots), state_tracking_terminated=np.asarray(terminations))
            excess = np.maximum(np.abs(arrays['requested_action'])-20., 0)
            target_error = np.abs(arrays['requested_joint_target']-arrays['executed_joint_target'])
            reading = dict(clip=clip, start_frame=int(start), stop_reason=reason, failure=failure,initial_state_error=initial_state_error,
                controls_exceeding_action_limit=int((excess > 0).any(axis=1).sum()),
                action_limit_excess_max=float(excess.max(initial=0)),
                controls_exceeding_action_limit_by_joint=dict(zip(joint_names, (excess > 0).sum(axis=0).tolist())),
                requested_target_max_error_rad=float(target_error.max(initial=0)),
                diagnostics=engine.read_diagnostics(reset=True))
            try:
                reading.update(summarize_source_chain(errors, torso_index, terminations,
                    reason == 'reference_complete', controls, requested_controls, report['control_dt_s']))
            except ValueError as error:
                reading.update(controls=controls, requested_controls=requested_controls,
                               reference_complete=False, strict_pass=False, reading_failure=str(error))
            report['reports'].append(reading)
            write_report()
    report.update(state='completed', reference_completes=sum(row['reference_complete'] for row in report['reports']),
                  strict_passes=sum(row['strict_pass'] for row in report['reports']))
    write_report()


if __name__ == '__main__':
    main()
