#!/bin/bash
# V7: FP32 smoke retry (NPU 3) + extraction shard 3 retry on NPU 5 (NPU6 invisible in aic-batch)
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
nohup $PY scripts/v7_e0_train_smoke.py --cards 3 --steps 60 --output $E/e0/train_smoke.json > $E/e0/train_smoke.log 2>&1 &
nohup $PY scripts/v7_s_extract_feats.py --cards 5 --shard 3 --nshards 4 --output-root $E/feats > $E/feats/extract_s3.log 2>&1 &
sleep 1
echo launched
