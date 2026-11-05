"""Short reference-reset tracking diagnostics at fixed frames of the jumpsit task."""
import argparse
import json
import pickle
from pathlib import Path

from nablamimic import torch_nets as nets
from nablamimic.torch_train import build_environment
from nablamimic.torch_terrain_evaluate import evaluate


REFERENCE_STARTS = (0, 100, 190, 225, 275, 350, 450)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', choices=('ema', 'live'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    for name in ('terrain_model', 'terrain_model_contract', 'terrain_sensor'):
        config[name] = Path(config[name])
    env, bindings = build_environment(argparse.Namespace(**config))
    with args.checkpoint.open('rb') as handle:
        checkpoint = pickle.load(handle)
    if checkpoint['perception'] != env.perception_contract or checkpoint['observation'] != env.observation_contract:
        raise ValueError('Checkpoint physical and observation bindings differ from constructed environment')
    if len(env.starts) != 1 or int(env.starts[0]) != 0 or int(env.lengths[0]) != 537:
        raise ValueError('Selected diagnostic requires the complete 537-frame jumpsit reference')
    actor_key = 'actor_ema' if args.weights == 'ema' else 'actor'
    actor = nets.tree_from_jax(checkpoint[actor_key], env.device)
    identity = dict(checkpoint_path=str(args.checkpoint), completed_updates=checkpoint['completed_updates'],
                    weights=args.weights)
    reports = []
    original = env.sensor_ablation
    try:
        for start in REFERENCE_STARTS:
            controls = min(100, 536-start)
            for sensor in ('measured', 'constant'):
                env.sensor_ablation = sensor
                output = args.output/f'frame{start:03d}_{sensor}'
                report = evaluate(env, actor, output, controls, start_frame=start)
                # A requested short window has its own interpretation. Full-reference
                # policy acceptance remains owned by the full twenty-start panel.
                report.pop('passing_trial')
                report.update(diagnostic='reference_reset_segment', sensor_intervention=sensor,
                    requested_window_end_frame=start+controls,
                    reached_window_end=report['controls']==controls and report['stop_reason']!='tracking_termination',
                    **identity)
                (output/'result.json').write_text(json.dumps(report, indent=2))
                reports.append(report)
    finally:
        env.sensor_ablation = original
    (args.output/'segments.json').write_text(json.dumps(dict(identity=identity,
        reference_starts=list(REFERENCE_STARTS), maximum_segment_controls=100,
        reset='Nominal physical reference state, zero reset and observation noise',
        reading='Local tracking diagnostic with every selected start retained', reports=reports), indent=2))


if __name__ == '__main__':
    main()
