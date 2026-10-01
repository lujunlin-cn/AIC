#!/bin/bash
set -a; . /data/aic/tools/ascend_teacher.env; set +a
cd /root/AIC
mkdir -p /data/aic/semifinal_20261001/teacher_points
exec python3 /root/AIC/scripts/v8_teacher_points.py \
  --index /data/aic/semifinal_20261001/intake/index.enriched.jsonl \
  --frames-dir /data/aic/semifinal_20261001/keyframes/keyframes \
  --output /data/aic/semifinal_20261001/teacher_points \
  --shard 0 --nshards 1 --tp 4 --max-seqs 32 --chunk 96
