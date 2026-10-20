#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/root/miniconda3/bin/python
CODE=/root/AIC/scripts
RUN=/data/aic/experiments/DINOV3_SPATIAL_MATCHED_V2
CACHE=/data/aic/features/DINOV3_SPATIAL_MATCHED_V2
mkdir -p "$RUN" "$CACHE"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/exit_code.txt"' EXIT
date -u > "$RUN/start_utc.txt"
nvidia-smi > "$RUN/nvidia_start.txt"
"$PY" -m pip freeze > "$RUN/pip_freeze.txt"
sha256sum "$CODE"/dinov3_spatial_matched.py "$CODE"/dinov3_spatial_patch_probe.py "$CODE"/dinov3_runtime.py "$CODE"/r70_spatial_utility_pilot.py > "$RUN/code.sha256"
"$PY" "$CODE/dinov3_matched_smoke.py" /data/aic/experiments/T5_CROPHEAD_V3/keyframes_train/video29/0.png > "$RUN/smoke.json"
if [[ ! -f "$CACHE/complete.json" ]]; then
  timeout --signal=TERM --kill-after=1m 45m "$PY" "$CODE/dinov3_spatial_matched.py" extract --out "$CACHE" --batch 64 --workers 4 > "$RUN/extract.log" 2>&1
fi
for arm in dino v9; do
  timeout --signal=TERM --kill-after=1m 30m "$PY" "$CODE/dinov3_spatial_matched.py" train --arm "$arm" --cache "$CACHE" --out "$RUN" --steps 1200 --train-batch 32 > "$RUN/${arm}_train.log" 2>&1
done
date -u > "$RUN/end_utc.txt"
nvidia-smi > "$RUN/nvidia_end.txt"
