"""An initialized training process can later resume without repeating initialization."""
import json
from pathlib import Path
import tempfile
import unittest

from nablamimic.params import resolve_params


class InitializeResumeTests(unittest.TestCase):
    def test_resume_preserves_provenance_and_does_not_initialize_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'recipe.args'
            config.write_text('--seed 42\n')
            locked = ['--seed', '42', '--initialize-from-ema', '/BONES/it_019000.pkl']
            (root / 'config.json').write_text(json.dumps(dict(argv=locked, config_args=['--seed', '42'])))
            argv, evidence = resolve_params(['--config', str(config), '--out', str(root), '--resume', 'latest'])
            self.assertNotIn('--initialize-from-ema', argv)
            self.assertEqual(evidence['argv'], locked)
            self.assertEqual(argv[-2:], ['--resume', 'latest'])


if __name__ == '__main__':
    unittest.main()
