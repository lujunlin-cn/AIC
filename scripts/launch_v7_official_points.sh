#!/bin/bash
# Usage: launch_v7_official_points.sh <head.pt> <outdir-tag>
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
HEAD=$1
OUT=$E/official_points_$2
for i in 0 1 2 3; do
  CARD=$((i+2))
  nohup $PY scripts/v7_s_official_points.py --head $HEAD --cards $CARD --shard $i --nshards 4 --output-dir $OUT > $OUT.shard$i.log 2>&1 &
done
sleep 1
echo launched shards to $OUT
