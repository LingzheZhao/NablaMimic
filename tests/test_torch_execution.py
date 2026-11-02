"""CPU storage/layout contracts; real CUDA capture is tested separately."""
import unittest

import torch

from nablamimic.torch_execution import _clone_tree, _copy_inputs_, _detach_tree


class CapturedInputTreeTest(unittest.TestCase):
    def test_private_storage_keeps_shapes_grad_flags_and_aliases(self):
        shared = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)
        source = ({'actor_input': shared, 'same_input': shared}, [torch.ones(2), None], 4)
        private = _clone_tree(source)
        self.assertIs(private[0]['actor_input'], private[0]['same_input'])
        self.assertNotEqual(private[0]['actor_input'].data_ptr(), shared.data_ptr())
        self.assertTrue(private[0]['actor_input'].requires_grad)
        self.assertTrue(private[0]['actor_input'].is_leaf)
        replacement = torch.full((2, 3), 7., requires_grad=True)
        pointer = private[0]['actor_input'].data_ptr()
        _copy_inputs_(private, ({'actor_input': replacement, 'same_input': replacement},
                                [torch.full((2,), 3.), None], 4))
        self.assertEqual(private[0]['actor_input'].data_ptr(), pointer)
        torch.testing.assert_close(private[0]['actor_input'], replacement)
        torch.testing.assert_close(shared, torch.arange(6, dtype=torch.float32).reshape(2, 3))

    def test_late_mismatch_rejects_before_any_buffer_is_changed(self):
        private = _clone_tree((torch.ones(2), {'counter': torch.zeros((), dtype=torch.int32)}))
        wrong = (torch.full((2,), 9.), {'counter': torch.zeros((), dtype=torch.int64)})
        with self.assertRaisesRegex(ValueError, 'layout differs'):
            _copy_inputs_(private, wrong)
        torch.testing.assert_close(private[0], torch.ones(2))
        for wrong in ((torch.ones(3), {'counter': torch.zeros((), dtype=torch.int32)}),
                      (torch.ones(2), {'other': torch.zeros((), dtype=torch.int32)}),
                      (torch.ones(2, requires_grad=True), {'counter': torch.zeros((), dtype=torch.int32)})):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                _copy_inputs_(private, wrong)

    def test_alias_and_static_value_changes_are_rejected(self):
        shared = torch.ones(2)
        private = _clone_tree((shared, shared, 'fixed'))
        with self.assertRaisesRegex(ValueError, 'alias pattern'):
            _copy_inputs_(private, (torch.ones(2), torch.ones(2), 'fixed'))
        with self.assertRaisesRegex(ValueError, 'static value'):
            _copy_inputs_(private, (shared, shared, 'changed'))

    def test_detached_outputs_preserve_storage_without_autograd_graph(self):
        input_tensor = torch.ones(2, requires_grad=True)
        original = {'output': input_tensor * 3., 'counter': (torch.tensor(2),)}
        detached = _detach_tree(original)
        self.assertEqual(detached['output'].data_ptr(), original['output'].data_ptr())
        self.assertIsNone(detached['output'].grad_fn)
        self.assertFalse(detached['output'].requires_grad)
        torch.testing.assert_close(detached['output'], original['output'])


if __name__ == '__main__':
    unittest.main()
