#!/bin/bash
# V7 stage 2: unfrozen S (NPU 3) + T TCN seed0 (NPU 4) + S KD arm seed0 (NPU 5)
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
nohup $PY scripts/v7_s_train_unfrozen.py --cards 3 --steps 600 --eval-every 100 --output-dir $E/s_unfrozen_s0 > $E/s_unfrozen_s0.log 2>&1 &
nohup $PY scripts/v7_t_train_tcn.py --device npu --seed 0 --output-dir $E/t_tcn > $E/t_tcn_s0.log 2>&1 &
nohup $PY scripts/v7_s_train_head_kd.py --device npu --seed 0 --kd-weight 0.2 --output-dir $E/s_kd_s0 > $E/s_kd_s0.log 2>&1 &
sleep 1
echo launched
