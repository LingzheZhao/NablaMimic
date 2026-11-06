"""Generated full state restores independently of one-time actor initialization."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nablamimic import torch_nets as nets
from nablamimic.torch_train import TorchTrainer


class TorchResumeInitializationTest(unittest.TestCase):
    def test_full_resume_ignores_initialization_and_restores_all_owners(self):
        trainer=TorchTrainer.__new__(TorchTrainer)
        trainer.device='cpu'
        trainer.env=SimpleNamespace(perception_contract={'sensor':'generated'},observation_contract={'width':3})
        trainer.args=SimpleNamespace(actor_checkpoint=None,actor_weights=None,official_actor_checkpoint=None,
            initial_checkpoint=None,resume='full.pt',out='fresh',iters=3000,seed=7,gamma=.99)
        owners={}
        for index,name in enumerate(('actor','critic','ema','actor_opt','critic_opt','cnorm','collector','replay')):
            value={'value':torch.zeros(2)}
            setattr(trainer,name,value)
            owners[name]=value
        trainer.ema_count=torch.tensor(0)
        trainer.outer_step=torch.tensor(0)
        trainer.replay_ptr=torch.zeros(1,dtype=torch.int64)
        trainer.sampler=SimpleNamespace(bin_q=torch.zeros(2),probs=torch.zeros(2))
        state=dict(format='nablamimic_torch_training_v1',perception=trainer.env.perception_contract,
            observation=trainer.env.observation_contract,resolved_config={**vars(trainer.args),
                'actor_checkpoint':'original_actor.pkl','actor_weights':'ema','out':'old','iters':1000},
            completed_updates=1000,total_collector_steps=2000,col_accumulator=.25,wall_s=17.,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.tensor([1],dtype=torch.uint8),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            ema_count=torch.tensor(1000),outer_step=torch.tensor(1000),replay_ptr=torch.tensor([3]),
            bin_q=torch.tensor([.2,.8]),probabilities=torch.tensor([.4,.6]))
        mapping={'actor_opt':'actor_optimizer','critic_opt':'critic_optimizer'}
        for index,(name,value) in enumerate(owners.items()):
            state[mapping.get(name,name)]={'value':torch.full((2,),index+1.)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.pt'
            torch.save(state,path)
            with patch('torch.cuda.set_rng_state') as cuda_restore:
                elapsed=trainer.restore(path)
            cuda_restore.assert_called_once()
            self.assertEqual(elapsed,17.)
            self.assertEqual(trainer.completed_updates,1000)
            self.assertEqual(trainer.total_collector_steps,2000)
            self.assertEqual(trainer.col_accumulator,.25)
            for name,value in owners.items():
                self.assertIs(getattr(trainer,name),value)
                torch.testing.assert_close(value['value'],state[mapping.get(name,name)]['value'])
            for name in ('ema_count','outer_step','replay_ptr'):
                torch.testing.assert_close(getattr(trainer,name),state[name])
            torch.testing.assert_close(trainer.sampler.bin_q,state['bin_q'])
            torch.testing.assert_close(trainer.sampler.probs,state['probabilities'])
            trainer.args.gamma=.98
            with self.assertRaisesRegex(ValueError,'gamma'):
                trainer.restore(path)


if __name__=='__main__':
    unittest.main()
