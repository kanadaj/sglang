#!/bin/bash
set -euo pipefail
export HOME=/tmp XDG_CACHE_HOME=/tmp/cache PYTHONDONTWRITEBYTECODE=1
for file in function_call/utils.py function_call/qwen3_coder_detector.py parser/reasoning_parser.py entrypoints/openai/serving_responses.py; do
 cp "/overlay/python/sglang/srt/$file" "/sgl-workspace/sglang/python/sglang/srt/$file"
done
python3 - "$@" <<'PY'
import pathlib
for p in pathlib.Path('/sgl-workspace/sglang/python/sglang').rglob('*.pyc'):
    p.unlink()
PY
exec python3 -m pytest -p no:cacheprovider "$@" -q --tb=short
