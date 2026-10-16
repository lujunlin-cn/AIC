#!/usr/bin/env bash
set -euo pipefail
task_root=/root/autodl-tmp/experiments/R29_BREAKTHROUGH
task_python=/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/qwen_env/bin/python
task_config=/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/R28_SPATIAL_PUBLIC_PILOT.json
task_previous=/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/public_pilot_cached/predictions.jsonl
cd "$task_root"
test ! -e spatial_continuation
"$task_python" r29_spatial_continue.py --config "$task_config" --previous "$task_previous" --output spatial_continuation --validate-only > spatial_preflight.json
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv > gpu_before.csv
# This user-authorized machine has one visible GPU. Refuse to touch another job.
task_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$task_gpu_pids" ]]; then
  printf '%s\n' "$task_gpu_pids"
  exit 73
fi
"$task_python" --version > python_version.txt
"$task_python" -m pip freeze > pip_freeze.txt
sha256sum r29_spatial_continue.py "$task_config" "$task_previous" > spatial_input_sha256.txt
set +e
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 20m "$task_python" r29_spatial_continue.py --config "$task_config" --previous "$task_previous" --output spatial_continuation > spatial_continuation.log 2>&1
task_exit=$?
set -e
printf '%s\n' "$task_exit" > spatial_exit_code.txt
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv > gpu_after.csv
exit "$task_exit"
