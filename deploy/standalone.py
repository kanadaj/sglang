#!/usr/bin/env python3
"""Online-download TP2 vision profile; no checkout or external JSON required."""
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
for key, value in json.loads((root / 'environment.json').read_text()).items():
    os.environ.setdefault(key, value)
args = sys.argv[1:]
if args[:2] == ['sglang', 'serve']:
    args = args[2:]
elif args[:1] == ['serve']:
    args = args[1:]
explicit = {arg.split('=', 1)[0] for arg in args if arg.startswith('--')}
defaults = [
    '--model-path', 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4',
    '--served-model-name', 'qwen-vision', '--host', '0.0.0.0', '--port', '30000',
    *json.loads((root / 'args.json').read_text())]
selected = []
i = 0
while i < len(defaults):
    end = i + 1
    if '=' not in defaults[i] and end < len(defaults) and not defaults[end].startswith('--'):
        end += 1
    if defaults[i].split('=', 1)[0] not in explicit:
        selected.extend(defaults[i:end])
    i = end
sys.argv[1:] = [*selected, *args]
runpy.run_path(str(root / 'launch.py'), run_name='__main__')
