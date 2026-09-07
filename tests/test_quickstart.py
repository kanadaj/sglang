"""Exercise the copyable Bash argv without Docker or a serving process."""
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def quickstart_argv():
    text = (ROOT / 'docs/standalone.md').read_text()
    match = re.search(r'```bash\n(IMAGE=.*?)\n```', text, re.S)
    assert match is not None, 'Missing copyable Docker block'
    block = match.group(1)
    subprocess.run(['bash', '-n'], input=block, text=True, check=True)
    # Bash itself performs quoting/continuations. Replace only the docker command.
    stub = 'docker() { printf "%s\\0" "$@"; };\n'
    result = subprocess.run(['bash', '-c', stub + block], capture_output=True, check=True)
    words = result.stdout.decode().split('\0')[:-1]
    return words[words.index('run'):]


class QuickstartTests(unittest.TestCase):
    def test_builds_do_not_package_defaults_launchers(self):
        for filename in ['Dockerfile', 'Dockerfile.digest']:
            text = (ROOT / filename).read_text()
            self.assertNotIn('COPY deploy', text)
            self.assertIn('ENTRYPOINT ["python3", "-m", "sglang.launch_server"]', text)
        for filename in ['Dockerfile.standalone', 'deploy/standalone.py',
                         'deploy/test_standalone.py', 'deploy/launch.py',
                         'deploy/test_launcher.py']:
            self.assertFalse((ROOT / filename).exists(), filename)

    def test_external_profile_matches_deployed_settings(self):
        argv = quickstart_argv()
        self.assertIn('--entrypoint', argv)
        self.assertEqual(argv[argv.index('--entrypoint') + 1], 'python3')
        self.assertEqual(argv[argv.index('--gpus') + 1], '"device=0,1"')
        module = argv.index('-m')
        self.assertEqual(argv[module + 1], 'sglang.launch_server')
        self.assertIn('@sha256:99fef9b4927e7e7c0dbd185a6bfe55995cea78e0d5a3c53e4409afb91109dc52', argv[module - 1])
        args = argv[module + 2:]
        expected = json.loads((ROOT / 'deploy/args.json').read_text())
        for arg in expected:
            self.assertIn(arg, args)
        def options(tokens):
            result = {}
            i = 0
            while i < len(tokens):
                key, separator, value = tokens[i].partition('=')
                i += 1
                if not separator:
                    value = True
                    if i < len(tokens) and not tokens[i].startswith('--'):
                        value = tokens[i]
                        i += 1
                self.assertNotIn(key, result, f'duplicate option {key}')
                result[key] = value
            return result
        import shlex
        deployed = json.loads((ROOT / 'docs/deployed-manifest.json').read_text())
        deployed_args = options([word for line in deployed['backend_parameters']
                                 for word in shlex.split(line)])
        actual = options(args)
        for key in ['--model-path', '--served-model-name', '--host', '--port']:
            actual.pop(key)
        self.assertEqual(actual, deployed_args)
        override = json.loads(args[args.index('--json-model-override-args') + 1])
        self.assertEqual(override, json.loads((ROOT / 'deploy/yarn2-model-override.json').read_text()))
        for key, value in json.loads((ROOT / 'deploy/environment.json').read_text()).items():
            self.assertIn(f'{key}={value}', argv[:module])
        self.assertNotIn('--language-model-only', args)
        self.assertNotIn('/opt/qwen-yarn', ' '.join(argv))


if __name__ == '__main__':
    unittest.main()
