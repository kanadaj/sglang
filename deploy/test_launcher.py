import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent

class LauncherTests(unittest.TestCase):
    def invoke(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'sglang'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / 'launch.py'), *args],
                env={**os.environ, 'PATH': tmp + os.pathsep + os.environ['PATH']},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_default_override_is_bundled_and_complete(self):
        argv = self.invoke('--model-path', '/model')
        self.assertEqual(argv[:3], ['serve', '--model-path', '/model'])
        override = json.loads(argv[argv.index('--json-model-override-args') + 1])
        self.assertEqual(override, json.loads((ROOT / 'yarn2-model-override.json').read_text()))
        self.assertEqual(override['text_config']['rope_parameters']['factor'], 2.0)
        self.assertIn('--context-length=524288', argv)

    def test_explicit_serving_arguments_take_precedence(self):
        for prefix in ([], ['serve'], ['sglang', 'serve']):
            for custom in ([ '--json-model-override-args', '{}', '--context-length', '262144'],
                           ['--json-model-override-args={}', '--context-length=262144']):
                with self.subTest(prefix=prefix, custom=custom):
                    self.assertEqual(self.invoke(*prefix, *custom), ['serve', *custom])

if __name__ == '__main__':
    unittest.main()
