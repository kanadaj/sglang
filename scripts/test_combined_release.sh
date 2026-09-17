#!/usr/bin/env bash
# No GPU access; immutable source tree, offline, unprivileged tests.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${COMBINED_IMAGE:?Set the image built with Dockerfile.combined-release}"
: "${QWEN_TOKENIZER_PATH:?Set an existing resolved tokenizer directory}"
python3 tests/test_combined_release.py
common=(--rm --pull never --runtime runc --network none --read-only --user 65534:65534
  --memory 8g --cpus 3 --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges
  --tmpfs /tmp:rw,exec,size=2g,mode=1777
  -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/tmp
  -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e SGLANG_DEVICE=cpu
  -e QWEN_HICACHE_TEST_DEVICE=cpu -v "$PWD:/repo:ro")
docker run "${common[@]}" --entrypoint python3 "$COMBINED_IMAGE" -B \
  /opt/qwen-combined-release/scripts/verify_combined_release.py --tree /sgl-workspace/sglang
docker run "${common[@]}" \
  -e PYTHONPATH=/repo/validation/hicache:/sgl-workspace/sglang/python \
  --entrypoint python3 "$COMBINED_IMAGE" -c \
  'from sglang.test.test_utils import maybe_stub_sgl_kernel; maybe_stub_sgl_kernel(); import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
  /repo/validation/cache-diagnostics/test_selective_diagnostics.py \
  /repo/validation/hicache/test_common_boundary.py /repo/validation/hicache/test_hicache_file_local.py \
  /repo/validation/hicache/test_hicache_qsa_local.py /repo/validation/hicache/test_hicache_ple_local.py \
  /repo/validation/hicache/test_hicache_load_order.py /repo/validation/hicache/test_qsa_short_extend.py \
  -q -p no:cacheprovider
docker run "${common[@]}" \
  -e QWEN_TOKENIZER_PATH=/tokenizer -e PYTHONPATH=/repo/tests:/sgl-workspace/sglang/python \
  -v "$QWEN_TOKENIZER_PATH:/tokenizer:ro" --entrypoint bash "$COMBINED_IMAGE" -c \
  'set -e; python3 /repo/tests/runtime_qwen_strict_tools.py -q; python3 /repo/tests/runtime_qwen_effort_alias.py -q; python3 /repo/validation/responses/runtime_stream_stability.py -q'
