#!/bin/bash
set -u
base=/root/autodl-tmp/experiments/R44_RESIDUAL_001
out=/root/autodl-tmp/experiments/R46_HIGHFREQ_LOCAL_RESIDUAL_B3
mkdir -p "$out"
for s in 0 1 2 3; do
  a=$((s*107)); b=$(((s+1)*107)); if [ "$s" -eq 3 ]; then b=426; fi
  CUDA_VISIBLE_DEVICES="" timeout --signal=TERM --kill-after=2m 3h /root/autodl-tmp/aha_env/bin/python "$base/scripts/r46_highfreq_local_residual.py" \
    --labels "$base/project1_labels.csv" "$base/project2_labels.csv" \
    --public-video-root /root/autodl-tmp/annotation_batch_v1/videos \
    --video-root /root/autodl-tmp/vpb_artifacts/video_426 \
    --parent "$base/b3_allframes.jsonl" --output "$out/shard${s}.jsonl" \
    --start "$a" --end "$b" > "$out/shard${s}.log" 2>&1 &
done
wait
