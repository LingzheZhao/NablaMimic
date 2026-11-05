"""Read paired current-format terrain trajectories on explicit state supports."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_chain(root, report, torso_index):
    directory = root / f"{report['weights']}_start{report['start']:02d}_{report['sensor_intervention']}"
    with np.load(directory / 'trajectory.npz') as data:
        body = np.concatenate((data['world_body_error_m'], data['final_world_body_error_m'][None]), axis=0)
        torso = np.concatenate((data['torso_error_m'], data['final_world_body_error_m'][torso_index:torso_index+1]))
        frames = np.concatenate((data['frame'], data['final_frame']))
    if len(body) != report['controls']+1 or len(torso) != len(body) or len(frames) != len(body):
        raise ValueError('Recorded state support differs from declared transition count')
    if not np.isfinite(body).all() or not np.isfinite(torso).all():
        raise ValueError('Nonfinite recorded physical error')
    return dict(body=body, torso=torso, frames=frames, report=report)


def readings(chain, states):
    report = chain['report']
    return dict(states=states, controls=states-1, duration_s=(states-1)/report['control_hz'],
                mean_world_body_error_m=float(chain['body'][:states].mean()),
                mean_torso_error_m=float(chain['torso'][:states].mean()),
                stop_reason=report['stop_reason'], reference_complete=report['stop_reason']=='reference_complete',
                posture_fall=report['posture_fall'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--torso-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    identity = panel['identity']
    groups = {}
    for report in panel['reports']:
        if report['checkpoint_path'] != identity['checkpoint_path'] or report['completed_updates'] != identity['completed_updates']:
            raise ValueError('Panel mixes checkpoint identities')
        key = (report['weights'], report['start'])
        condition = report['sensor_intervention']
        if condition not in ('measured', 'constant') or condition in groups.setdefault(key, {}):
            raise ValueError('Duplicate or undeclared sensor condition')
        groups[key][condition] = read_chain(args.panel.parent, report, args.torso_index)
    pairs = []
    for (weights, start), group in sorted(groups.items()):
        if set(group) != {'measured', 'constant'}:
            raise ValueError('Incomplete measured/constant pair')
        measured, constant = group['measured'], group['constant']
        a, b = measured['report'], constant['report']
        for name in ('control_hz', 'start_frame', 'requested_controls', 'execution'):
            if a[name] != b[name]:
                raise ValueError(f'Paired physical reading conditions differ for {name}')
        states = min(len(measured['body']), len(constant['body']))
        if not np.array_equal(measured['frames'][:states], constant['frames'][:states]):
            raise ValueError('Paired reference-frame supports differ')
        full = {name: readings(chain, len(chain['body'])) for name, chain in group.items()}
        common = {name: readings(chain, states) for name, chain in group.items()}
        difference = {name: common['measured'][name]-common['constant'][name]
                      for name in ('mean_world_body_error_m', 'mean_torso_error_m')}
        pairs.append(dict(weights=weights, start=start, full_support=full, common_support=common,
                          measured_minus_constant_common=difference))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(dict(identity=identity, torso_index=args.torso_index,
        definition='Per-pair full and common prefix state supports include each recorded terminal state. Stop and posture labels refer to each full observed chain.',
        pairs=pairs), indent=2))


if __name__ == '__main__':
    main()
