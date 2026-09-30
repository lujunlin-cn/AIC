#!/bin/bash
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
ASCEND_RT_VISIBLE_DEVICES=4 nohup $PY scripts/v7_t_train_tcn.py --device npu --seed 0 --output-dir $E/t_tcn > $E/t_tcn_s0.log 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=3 nohup $PY scripts/v7_s_train_unfrozen.py --steps 600 --eval-every 100 --output-dir $E/s_unfrozen_s0 > $E/s_unfrozen_s0.log 2>&1 &
ASCEND_RT_VISIBLE_DEVICES=2 nohup $PY scripts/v7_s_train_head.py --device npu --seed 1 --output-dir $E/s_head_s1 > $E/s_head_s1.log 2>&1 &
sleep 1
echo launched
