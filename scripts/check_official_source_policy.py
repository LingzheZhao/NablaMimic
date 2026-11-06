"""Compare the frozen release reader with the pinned original actor classes on CPU."""
import argparse
from collections import OrderedDict
import contextlib
import io
import json
from pathlib import Path
import sys

import torch

from nablamimic.official_source_policy import OfficialSourcePolicy, native_actor_from_official
from nablamimic.perceptive_task_inputs import source_actor_apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observations',type=Path,help='Optional .npy raw source observations [B,1990], including nominal states')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.upstream))
    from instinct_rl.modules import EncoderActorCritic
    from instinct_rl.modules.normalizer import EmpiricalNormalization

    segments = OrderedDict((name, shape) for name, shape in (
        ('joint_pos_ref', (290,)), ('joint_vel_ref', (290,)),
        ('position_ref', (30,)), ('rotation_ref', (60,)),
        ('depth_image', (1, 18, 32)), ('projected_gravity', (24,)),
        ('base_ang_vel', (24,)), ('joint_pos', (232,)),
        ('joint_vel', (232,)), ('last_action', (232,))))
    with contextlib.redirect_stdout(io.StringIO()):
        original = EncoderActorCritic(
            obs_format={'policy': segments, 'critic': {'privileged': (1667,)}},
            num_actions=29, actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128], activation='elu',
            encoder_configs={'depth_image': dict(
                class_name='Conv2dHeadModel', channels=[32, 32], kernel_sizes=[3, 3],
                strides=[1, 1], paddings=[1, 1], hidden_sizes=[32],
                nonlinearity='ReLU', use_maxpool=False, output_size=32,
                component_names=['depth_image'], takeout_input_components=True)})
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    original.load_state_dict(checkpoint['model_state_dict'], strict=True)
    original.eval()
    normalizer = EmpiricalNormalization(shape=(1990,))
    normalizer.load_state_dict(checkpoint['policy_normalizer_state_dict'], strict=True)
    normalizer.eval()
    reader = OfficialSourcePolicy(checkpoint)
    torch.manual_seed(0)
    errors = []
    native_errors=[]
    native=native_actor_from_official(checkpoint,'cpu')
    with torch.no_grad():
        for batch in (1, 3, 8):
            raw = torch.randn(batch, 1990)
            history_depth = torch.cat((raw[:, 1246:], raw[:, 670:1246]), -1)
            expected = original.act_inference(normalizer(raw))
            actual = reader(history_depth, raw[:, :670])
            errors.append(float((expected - actual).abs().max()))
            native_errors.append(float((actual-source_actor_apply(native,history_depth,raw[:,:670])).abs().max()))
    nominal_error=None
    if args.observations:
        import numpy as np
        raw=torch.from_numpy(np.load(args.observations)).float()
        if raw.ndim!=2 or raw.shape[1]!=1990:
            raise ValueError('Raw source observations must have shape [B,1990]')
        history_depth=torch.cat((raw[:,1246:],raw[:,670:1246]),-1)
        with torch.no_grad():
            nominal_error=float((reader(history_depth,raw[:,:670])-source_actor_apply(native,history_depth,raw[:,:670])).abs().max())
    result = dict(native_initialization_random_maximum_absolute_errors=native_errors,
                  nominal_observations_path=str(args.observations) if args.observations else None,
                  nominal_initialization_maximum_absolute_error=nominal_error,
                  stored_std_vs_sqrt_variance_absolute_max=float((reader.std-checkpoint['policy_normalizer_state_dict']['_var'].sqrt()).abs().max()),
                  batch_sizes=[1, 3, 8], maximum_absolute_errors=errors,
                  matches=all(error <= 1e-6 for error in errors),
                  normalization='stored standard deviation plus .01',
                  actor_input_order=['goal', 'history', 'depth_feature'])
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    if not result['matches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
