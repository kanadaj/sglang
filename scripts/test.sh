#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${PYTHON:-python3}"
"$PYTHON" scripts/verify_source.py
"$PYTHON" -m unittest discover -s tests -v
