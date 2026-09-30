#!/bin/bash
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
nohup $PY scripts/v7_s_train_head.py --device npu --seed 0 --output-dir $E/s_head_s0 > $E/s_head_s0.log 2>&1 &
sleep 1
echo launched
