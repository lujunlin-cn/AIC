#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run only after a human/agent has checked nvidia-smi and selected an allowed
# physical GPU. The default is physical GPU 2 (logical cuda:0).
GPU_ID="${AIC_GPU_ID:-2}"
CONFIG="${AIC_CONFIG:-configs/A0_001.json}"
PYTHON_BIN="${AIC_PYTHON:-python}"
case ",1,2,4,5,6,7," in
  *",${GPU_ID},"*) ;;
  *) echo "Refusing disallowed physical GPU ${GPU_ID}; allowed: 1 2 4 5 6 7" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
# This host's timeout parser accepts seconds reliably; 42,600s = 11h50m.
exec timeout --signal=TERM --kill-after=3m 42600s \
  "${PYTHON_BIN}" -m aic.train --config "${CONFIG}"
