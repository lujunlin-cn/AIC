#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ASSET_ROOT="${1:-/root/autodl-tmp/videoora42_clone}"
RESEARCH_PYTHON="${RESEARCH_PYTHON:-$ASSET_ROOT/.venv/bin/python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
"$RESEARCH_PYTHON" -u scripts/run_spatial_research.py run \
  --asset-root "$ASSET_ROOT" --device cuda:0 --resume
