"""CPU subprocess tests; fake only the final sglang executable."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent

class StandaloneTests(unittest.TestCase):
    def invoke(self, *args):
        self.assertTrue((ROOT / 'standalone.py').exists(), 'bundled standalone launcher missing')
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'sglang'
            fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps({"argv":sys.argv[1:],"env":dict(os.environ)}))\n')
            fake.chmod(0o755)
            result = subprocess.run([sys.executable, str(ROOT / 'standalone.py'), *args],
                env={**os.environ, 'PATH': tmp + os.pathsep + os.environ['PATH']},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_default_vision_profile(self):
        result = self.invoke()
        argv = result['argv']
        for arg in json.loads((ROOT / 'args.json').read_text()):
            self.assertIn(arg, argv)
        self.assertNotIn('--language-model-only', argv)
        self.assertIn('local-inference-lab/Qwen3.8-Flash-Next-NVFP4', argv)
        for key, value in json.loads((ROOT / 'environment.json').read_text()).items():
            self.assertEqual(result['env'][key], value)
        self.assertEqual(json.loads(argv[argv.index('--json-model-override-args') + 1]),
                         json.loads((ROOT / 'yarn2-model-override.json').read_text()))
        self.assertNotEqual(result['env'].get('HF_HUB_OFFLINE'), '1')
        self.assertNotEqual(result['env'].get('TRANSFORMERS_OFFLINE'), '1')

    def test_explicit_overrides_replace_defaults(self):
        for prefix in ([], ['serve'], ['sglang', 'serve']):
            for custom in (['--model-path', '/model', '--context-length', '262144'],
                           ['--model-path=/model', '--context-length=262144']):
                with self.subTest(prefix=prefix, custom=custom):
                    argv = self.invoke(*prefix, *custom)['argv']
                    self.assertNotIn('local-inference-lab/Qwen3.8-Flash-Next-NVFP4', argv)
                    self.assertNotIn('--context-length=524288', argv)
                    self.assertEqual(argv.count('serve'), 1)
                    start = argv.index(custom[0])
                    self.assertEqual(argv[start:start + len(custom)], custom)

if __name__ == '__main__':
    unittest.main()
