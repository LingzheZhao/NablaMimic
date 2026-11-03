"""CPU checks for precision selection and the resolved training configuration."""

from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Optional PyTorch backend is not installed")

import torch

from nablamimic import torch_train
from nablamimic.params import read_params


class TrainingPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.precision = torch.get_float32_matmul_precision()
        self.cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.cudnn_tf32 = torch.backends.cudnn.allow_tf32

    def tearDown(self):
        torch.set_float32_matmul_precision(self.precision)
        torch.backends.cuda.matmul.allow_tf32 = self.cuda_tf32
        torch.backends.cudnn.allow_tf32 = self.cudnn_tf32

    def test_default_and_explicit_precision_map_to_backend_flags(self):
        parser = torch_train.build_parser()
        default = parser.parse_args(["--out", "unused"])
        self.assertEqual(default.matmul_precision, "nets_tf32")
        for mode in ("nets_tf32", "highest", "nets_tf32", "highest"):
            with self.subTest(mode=mode):
                selected = parser.parse_args(["--out", "unused", "--matmul-precision", mode])
                torch.backends.cudnn.allow_tf32 = True
                observed = torch_train.configure_matmul_precision(selected.matmul_precision)
                self.assertEqual(observed, torch_train.read_matmul_precision())
                self.assertEqual(observed["cuda_matmul_allow_tf32"], mode == "nets_tf32")
                self.assertFalse(observed["cudnn_allow_tf32"])
                self.assertEqual(observed["float32_matmul_precision"], torch.get_float32_matmul_precision())
                if mode == "highest":
                    self.assertEqual(observed["float32_matmul_precision"], "highest")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--out", "unused", "--matmul-precision", "medium"])

    def test_torch_config_uses_official_actor_ema_arguments(self):
        recipe = Path(torch_train.__file__).resolve().parents[2] / 'configs/ours_torch.args'
        parser = torch_train.build_parser()
        args = parser.parse_args(read_params(recipe) + ['--out', 'unused'])
        torch_train.validate_args(args)
        self.assertEqual(args.actor_ema, .999)
        self.assertTrue(args.actor_ema_warmup)
        self.assertEqual(args.lr_warmup_steps, 100)
        self.assertFalse(args.mask_after_done)
        self.assertEqual(args.execution, 'compile-graph')
        args = parser.parse_args(read_params(recipe) + ['--out', 'unused', '--actor-ema', '0'])
        torch_train.validate_args(args)
        self.assertEqual(args.actor_ema, 0)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--out', 'unused', '--policy-ema'])

    def test_resolved_fp32_config_overrides_fixed_tf32_metadata_before_model_loading(self):
        # Execute the real parser/config path, stopping before reference/model
        # preparation. Device capability and the unused native import are the
        # only stand-ins; precision setters/getters and config writing are real.
        class ReachedModelInputs(Exception):
            pass

        engine = ModuleType("nablaengine")
        engine.torch_ffi = object()
        recipe = Path(torch_train.__file__).resolve().parents[2] / "configs" / "ours_torch.args"
        torch_train.configure_matmul_precision("nets_tf32")
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(sys.modules, {"nablaengine": engine}),
            mock.patch.dict(os.environ, {"NABLA_SOURCE_REVISION": "precision-config-test"}),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(torch.cuda, "get_device_name", return_value="CPU configuration test"),
            mock.patch.object(torch, "set_num_threads"),
            mock.patch.object(torch, "manual_seed"),
            mock.patch.object(torch_train.np.random, "seed"),
            mock.patch.object(torch_train.random, "seed"),
            mock.patch("nablamimic.motion.all_clip_names", side_effect=ReachedModelInputs),
        ):
            with self.assertRaises(ReachedModelInputs):
                torch_train.main(["--config", str(recipe), "--out", directory,
                                  "--matmul-precision", "highest"])
            configuration = json.loads((Path(directory) / "config.json").read_text())
        self.assertEqual(configuration["matmul_precision"], "highest")
        self.assertEqual(configuration["observed_precision"], {
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        })


if __name__ == "__main__":
    unittest.main()
