"""Current official zero-update checkpoint inputs; fresh arrays, no run artifacts."""
import importlib.util
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Optional PyTorch backend is not installed')

import numpy as np
import torch

from nablamimic import torch_nets as nets
from nablamimic.torch_train import TorchTrainer, validate_initial_checkpoint


class InitialCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SimpleNamespace(goal_dim=686, hist_dim=930, critic_dim=825, nu=29)
        def mlp(sizes):
            return [(np.zeros((a, b), np.float32), np.zeros(b, np.float32)) for a, b in zip(sizes, sizes[1:])]
        actor = dict(enc=mlp([680, *nets.ACTOR_HIDDEN, nets.ACTOR_LATENT]),
                     dec=mlp([nets.ACTOR_LATENT+936, *nets.ACTOR_HIDDEN, 29]),
                     anchor_dec=np.zeros(6, np.float32))
        cls.checkpoint = dict(actor=actor, actor_ema=actor,
            critic=mlp([825, *nets.CRITIC_HIDDEN, 1]), cnorm=(np.zeros(825,np.float32), np.ones(825,np.float32)),
            st_a=(), st_c=(), st_r=(), bin_q=np.ones((2, 3),np.float32), iter=0, env_steps=0, wall_s=.01)

    def test_official_zero_metadata_is_sufficient(self):
        self.assertIs(validate_initial_checkpoint(self.checkpoint, self.env), self.checkpoint)
        without_average = {key:value for key,value in self.checkpoint.items() if key != 'actor_ema'}
        validate_initial_checkpoint(without_average, self.env)

    def test_nonzero_or_malformed_current_metadata_rejected(self):
        for changes in ({'iter':1}, {'env_steps':1}, {'iter':False}, {'wall_s':float('nan')},
                        {'unrecognized':0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_initial_checkpoint({**self.checkpoint, **changes}, self.env)
        missing = {key:value for key,value in self.checkpoint.items() if key != 'env_steps'}
        with self.assertRaises(ValueError):
            validate_initial_checkpoint(missing, self.env)

    def test_initial_weights_and_normalizer_contract(self):
        malformed = {**self.checkpoint, 'cnorm':(np.ones(825,np.float32), np.ones(825,np.float32))}
        with self.assertRaises(ValueError):
            validate_initial_checkpoint(malformed, self.env)
        malformed = {**self.checkpoint, 'actor':{**self.checkpoint['actor'], 'anchor_dec':np.zeros(5,np.float32)}}
        with self.assertRaises(ValueError):
            validate_initial_checkpoint(malformed, self.env)
        bad_critic = list(self.checkpoint['critic'])
        bad_critic[-1] = (np.zeros((3,1),np.float32), np.zeros(1,np.float32))
        with self.assertRaises(ValueError):
            validate_initial_checkpoint({**self.checkpoint,'critic':bad_critic}, self.env)
        bad_critic[-1] = (self.checkpoint['critic'][-1][0], np.full(1,np.nan,np.float32))
        with self.assertRaises(ValueError):
            validate_initial_checkpoint({**self.checkpoint,'critic':bad_critic}, self.env)

    def test_numbered_weights_expose_official_actor_keys(self):
        for decay in (0., .999):
            fake = SimpleNamespace(completed_updates=3, n=2, h=4,
                actor={'enc':[torch.ones(1)]}, critic=[torch.zeros(1)], ema={'enc':[torch.full((1,), .5)]},
                ema_count=torch.tensor(3 if decay else 0), cnorm=(torch.zeros(1),torch.ones(1)),
                sampler=SimpleNamespace(bin_q=torch.ones((2,1))),
                args=SimpleNamespace(actor_ema=decay,execution='compile-graph'))
            with self.subTest(decay=decay), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); (root/'ckpt').mkdir()
                path=TorchTrainer.save(fake,root,1.5,full=False)
                with open(path,'rb') as handle:checkpoint=pickle.load(handle)
                self.assertEqual((checkpoint['iter'],checkpoint['env_steps']),(3,24))
                self.assertEqual('actor_ema' in checkpoint,decay>0)
                self.assertIsInstance(checkpoint['actor']['enc'][0],np.ndarray)


if __name__ == '__main__':
    unittest.main()
