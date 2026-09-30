#!/bin/bash
cd /root/AIC
PY=/data/aic/tools/lfm_venv/bin/python
OUT=/data/aic/experiments_910a/LFM_V8/arms
while [ ! -f /data/aic/experiments_910a/LFM_V8/samples/live_train.npz ]; do sleep 60; done
sleep 30
echo "[train] starting arms $(date)" >> /data/aic/experiments_910a/LFM_V8/chain.log

train_arm() {  # card name sources steps seed
  [ -f $OUT/$2_s$5/summary.json ] && { echo "[skip] $2_s$5 done"; return; }
  ASCEND_RT_VISIBLE_DEVICES=$1 $PY scripts/v8_s_train_multidata.py \
    --sources ${3//,/ } --steps $4 --seed $5 --output-dir $OUT/$2_s$5 > $OUT/$2_s$5.log 2>&1
}
kd_arm() {  # card weight seed
  [ -f $OUT/KD$2_s$3/summary.json ] && { echo "[skip] KD$2_s$3 done"; return; }
  ASCEND_RT_VISIBLE_DEVICES=$1 $PY scripts/v8_s_train_kd_pref.py \
    --kd-weight $2 --seed $3 --output-dir $OUT/KD$2_s$3 > $OUT/KD$2_s$3.log 2>&1
}

( train_arm 2 B1 rv_native 1200 0; train_arm 2 B3 rv_native,rv_rot,live_train 3000 0; kd_arm 2 0.3 0 ) &
( train_arm 3 B1 rv_native 1200 1; train_arm 3 B3 rv_native,rv_rot,live_train 3000 1; kd_arm 3 0.3 1 ) &
( train_arm 4 B2 rv_native,rv_rot 1200 0; train_arm 4 B4 live_train 3000 0; kd_arm 4 0.1 0 ) &
( train_arm 5 B2 rv_native,rv_rot 1200 1; train_arm 5 B4 live_train 3000 1; kd_arm 5 0.1 1 ) &
wait
echo "[train] all arms done $(date)" >> /data/aic/experiments_910a/LFM_V8/chain.log
