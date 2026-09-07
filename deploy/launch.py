#!/usr/bin/env python3
"""Default standalone launcher; GPUStack may override the entrypoint."""
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[:2] == ['sglang', 'serve']:
    args = args[2:]
elif args[:1] == ['serve']:
    args = args[1:]
if not any(a == '--json-model-override-args' or a.startswith('--json-model-override-args=') for a in args):
    override = json.loads(Path(__file__).with_name('yarn2-model-override.json').read_text())
    args += ['--json-model-override-args', json.dumps(override, separators=(',', ':'))]
if not any(a == '--context-length' or a.startswith('--context-length=') for a in args):
    args += ['--context-length=524288']
os.execvp('sglang', ['sglang', 'serve', *args])
