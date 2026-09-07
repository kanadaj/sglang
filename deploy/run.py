#!/usr/bin/env python3
"""Generate or execute an equivalent standalone TP2 Docker launch.

Defaults to printing JSON argv; only --execute starts a container.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--execute', action='store_true')
p.add_argument('--model-dir', required=True, type=Path)
p.add_argument('--gpus', default='0,1', help='Select an unused GPU pair explicitly')
p.add_argument('--port', type=int, default=30000)
p.add_argument('--model-name', default='qwen-vision')
p.add_argument('--image', default='kanadaj/sglang-qwen38fn-sm120-turbo:r22-tp2-vision-pad32-bias-20260907-v2@sha256:8cb9b598ba0be1bbd77924037a517d71e064ef72807b8b96b0fa5c78eeb2f3cc')
a = p.parse_args()
if not a.model_dir.is_dir():
    p.error('--model-dir must exist')
if not 1 <= a.port <= 65535:
    p.error('invalid port')
argv = ['docker', 'run', '--rm', '--gpus', '"device=' + a.gpus + '"', '--ipc=host',
        '-p', f'127.0.0.1:{a.port}:30000', '--mount', f'type=bind,src={a.model_dir.resolve()},dst=/model,readonly']
for key, value in json.loads((ROOT / 'environment.json').read_text()).items():
    argv += ['-e', key + '=' + value]
argv += ['--entrypoint', 'sglang', a.image, 'serve', '--model-path=/model',
         '--host=0.0.0.0', '--port=30000', '--served-model-name=' + a.model_name]
argv += json.loads((ROOT / 'args.json').read_text())
argv += ['--json-model-override-args', json.dumps(json.loads((ROOT / 'yarn2-model-override.json').read_text()), separators=(',', ':'))]
if a.execute:
    subprocess.run(argv, check=True)
else:
    print(json.dumps(argv, indent=2))
