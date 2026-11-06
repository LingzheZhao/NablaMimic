"""Generated release-schema tensors test official-to-native actor initialization."""
import unittest

import torch

from nablamimic.official_source_policy import OfficialSourcePolicy, native_actor_from_official
from nablamimic.perceptive_task_inputs import source_actor_apply, source_actor_init
from nablamimic import torch_nets as nets


def checkpoint_fixture():
    torch.manual_seed(12)
    native=source_actor_init('cpu')
    state={}
    for (weight,bias),index in zip(native['dec'],(0,2,4,6)):
        state[f'actor.{index}.weight']=weight.T.clone()
        state[f'actor.{index}.bias']=bias.clone()
    for native_prefix,original_prefix in (
            ('convolutions.0','conv.conv.0'),('convolutions.2','conv.conv.2'),
            ('head.0','head.model.0'),('head.2','head.model.2')):
        for name in ('weight','bias'):
            state['encoders._parallel_blocks.depth_image.'+original_prefix+'.'+name]=native['enc'][native_prefix+'.'+name].clone()
    variance=torch.rand(1,1990)+.5
    return dict(model_state_dict=state,policy_normalizer_state_dict={
        '_mean':torch.randn(1,1990)*.1,'_var':variance,'_std':variance.sqrt(),
        'count':torch.tensor(123,dtype=torch.int64)},iter=8000)


class OfficialActorInitializationTest(unittest.TestCase):
    def test_mapping_preserves_output_and_trains_encoder_and_actor(self):
        checkpoint=checkpoint_fixture()
        native=native_actor_from_official(checkpoint,'cpu')
        official=OfficialSourcePolicy(checkpoint)
        history=torch.randn(3,1320)
        goal=torch.randn(3,670)
        torch.testing.assert_close(source_actor_apply(native,history,goal),official(history,goal),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(native['dec'][0][0].T,checkpoint['model_state_dict']['actor.0.weight'],rtol=0,atol=0)
        self.assertEqual(int(native['source_norm']['count']),123)
        average,count=nets.init_policy_ema(native)
        self.assertEqual(int(count),0)
        torch.testing.assert_close(average['dec'][0][0],native['dec'][0][0])
        for tensor in nets.tree_leaves(nets.actor_trainable(native)):
            tensor.requires_grad_()
        loss=source_actor_apply(native,history,goal).square().mean()
        loss.backward()
        self.assertGreater(float(native['enc']['convolutions.0.weight'].grad.abs().sum()),0)
        self.assertGreater(float(native['dec'][0][0].grad.abs().sum()),0)

    def test_native_variance_and_frozen_stored_std_have_distinct_owners(self):
        checkpoint=checkpoint_fixture()
        checkpoint['policy_normalizer_state_dict']['_std'].mul_(2)
        native=native_actor_from_official(checkpoint,'cpu')
        torch.testing.assert_close(native['source_norm']['variance'],checkpoint['policy_normalizer_state_dict']['_var'][0])
        official=OfficialSourcePolicy(checkpoint)
        torch.testing.assert_close(official.std,checkpoint['policy_normalizer_state_dict']['_std'])


if __name__=='__main__':
    unittest.main()
