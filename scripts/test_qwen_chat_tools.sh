#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${QWEN_TOKENIZER_PATH:?Set an existing resolved tokenizer directory}"
IMAGE=${QWEN_CHAT_TOOLS_IMAGE:-kanadaj/sglang-qwen38fn-sm120-turbo:qwen-chat-tools-20261002-reviewed-candidate}
docker run --rm --runtime runc --pull never --network none --memory 8g --cpus 3 \
 --user 1000:1000 -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/cache \
 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
 -e PYTHONPATH=/candidate/tests -e QWEN_TOKENIZER_PATH=/tokenizer \
 -v "$ROOT:/candidate:ro" -v "$QWEN_TOKENIZER_PATH:/tokenizer:ro" \
 --entrypoint bash "$IMAGE" -euo pipefail -c '
 python3 -B /candidate/scripts/verify_qwen_chat_tools.py --tree /sgl-workspace/sglang
 python3 -m pytest -p no:cacheprovider -q --tb=short --disable-warnings \
 /candidate/validation/qwen-chat-tools /candidate/tests/runtime_hicache_pr19_embed_tools.py \
 /candidate/tests/test_qwen_chat_tools_packaging.py \
 /candidate/tests/test_qwen_chat_tools_negative_controls.py
 python3 -m pytest -p no:cacheprovider -q --tb=short --disable-warnings \
 /sgl-workspace/sglang/test/registered/unit/function_call/test_function_call_parser.py \
 /sgl-workspace/sglang/test/registered/unit/function_call/test_unknown_tool_name.py \
 -k "not TestDeepSeekV32Detector and not TestDeepSeekV4Detector"
 python3 -m pytest -p no:cacheprovider -q --tb=short --disable-warnings \
 /sgl-workspace/sglang/test/registered/unit/parser/test_reasoning_parser.py
 '
