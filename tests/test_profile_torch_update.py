"""CPU checks of the profiling entry point against the real Torch trainer."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('Optional PyTorch backend is not installed')

import torch
from nablamimic import torch_train

spec = importlib.util.spec_from_file_location(
    'profile_torch_update', Path(__file__).resolve().parents[1] / 'scripts/profile_torch_update.py')
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


class Event:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def elapsed_time(self, end):
        return 0.0


class ProfileUpdateTest(unittest.TestCase):
    def test_entry_point_selects_third_update_on_real_trainer_class(self):
        calls = []
        trainer = object.__new__(torch_train.TorchTrainer)
        trainer.completed_updates = 0

        def update(instance):
            instance.completed_updates += 1
            calls.append(instance.completed_updates)
            return instance.completed_updates

        def main(arguments):
            self.assertEqual(arguments, ['--iters', '3'])
            for _ in range(3):
                trainer.update()

        with patch.object(torch_train.TorchTrainer, 'update', update), \
             patch.object(torch_train, 'main', main), \
             patch.object(profiler, 'profile_update', side_effect=lambda instance, fn, trace: fn(instance)) as selected:
            profiler.main(['--', '--iters', '3'])
            self.assertEqual(selected.call_count, 1)
            self.assertIs(torch_train.TorchTrainer.update, update)
        self.assertEqual(calls, [1, 2, 3])

    def test_real_update_call_surface_and_rng_are_preserved(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        with tempfile.TemporaryDirectory() as output:
            trainer = object.__new__(torch_train.TorchTrainer)
            trainer.args = SimpleNamespace(utd=.4, execution='compile-graph', regression_batch=2,
                action_target=[8], critic_epochs=5, critic_minibatches=4, out=output)
            trainer.completed_updates = 2
            trainer.col_accumulator = 0.0
            trainer.total_collector_steps = 0
            trainer.device, trainer.m = 'cpu', 8
            trainer.env = SimpleNamespace(max_reward=1., action_scale=torch.ones(2))
            trainer.actor_opt = {'count': torch.tensor(16)}
            trainer.critic_opt = {'count': torch.tensor(40)}
            trainer.ema_count = torch.tensor(2)
            trainer.cnorm = (torch.zeros(3), torch.ones(3))
            trainer.collector = {'qpos': torch.zeros(2, 36), 'qvel': torch.zeros(2, 35)}
            trainer.training_data = dict(r=torch.ones(2, 4), term=torch.zeros(2, 4, dtype=torch.bool),
                done=torch.zeros(2, 4, dtype=torch.bool), v2=torch.ones(2, 4),
                v2_raw=torch.ones(2, 4), window_value_raw=torch.ones(4), window_value_used=torch.ones(4),
                a_pi=torch.ones(2, 4, 2), hist=torch.ones(2, 4, 3), cobs_pre=torch.ones(2, 4, 3),
                terms={'track': torch.ones(2, 4)})
            trainer.target_data = dict(targets=torch.ones(2, 4, 2), weights=torch.ones(2, 4),
                                       at_clip=torch.ones(2))
            trainer.critic_targets = torch.ones(2, 4)
            trainer.window_inputs = lambda: ()
            trainer.collector_inputs = lambda: ()
            trainer.target_fn = lambda action, gradient: trainer.target_data
            trainer.td = lambda reward, term, done, value: value
            trainer.window = lambda: (torch.tensor(1.), torch.ones(2, 4, 2), torch.ones(2), trainer.training_data)
            trainer.collect = lambda: torch.tensor(0.)

            def optimizer(indices, state):
                state['count'].add_(1)
                return torch.tensor(1.), torch.tensor(1.), torch.tensor(1.)

            trainer.regression = lambda indices: optimizer(indices, trainer.actor_opt)
            trainer.critic_update = lambda indices: optimizer(indices, trainer.critic_opt)
            trainer.statistics = lambda: trainer.ema_count.add_(1)
            trainer.graphs = dict(window=trainer.window, collector=trainer.collect,
                regression=trainer.regression, critic=trainer.critic_update, statistics=trainer.statistics)
            originals = trainer.graphs.copy()
            engine = SimpleNamespace(torch_ffi=SimpleNamespace(read_diagnostics=lambda reset: {}))
            cuda_patches = dict(Event=Event, synchronize=lambda: None, memory_allocated=lambda: 0,
                memory_reserved=lambda: 0, max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
            with patch.multiple(torch.cuda, **cuda_patches), patch.dict('sys.modules', nablaengine=engine):
                seed = torch.get_rng_state()
                profiler.profile_update(trainer, torch_train.TorchTrainer.update, False)
                measured_rng = torch.get_rng_state()
                self.assertEqual(trainer.completed_updates, 3)
                self.assertEqual(trainer.graphs, originals)
                self.assertEqual(int(trainer.actor_opt['count']), 24)
                self.assertEqual(int(trainer.critic_opt['count']), 60)
                self.assertTrue((Path(output) / 'update-profile-000003.json').exists())
                torch.set_rng_state(seed)
                # These are the only random draws made by this CPU update fixture.
                for _ in range(8):
                    torch.randint(8, (2,))
                for _ in range(5):
                    torch.randperm(8)
                self.assertTrue(torch.equal(measured_rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
