#!/bin/bash
# V7 E0: protocol check (NPU 2) + train smoke (NPU 3)
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7/e0
PY=/data/aic/tools/lfm_venv/bin/python
mkdir -p "$E"
nohup $PY scripts/v7_e0_protocol_check.py --cards 2 --output $E/protocol_check.json > $E/protocol_check.log 2>&1 &
nohup $PY scripts/v7_e0_train_smoke.py --cards 3 --steps 60 --output $E/train_smoke.json > $E/train_smoke.log 2>&1 &
sleep 1
echo launched
