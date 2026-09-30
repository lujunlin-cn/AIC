#!/bin/bash
# V8 arm queue, memory-throttled (V8 fix).
#
# Root cause of the earlier host RAM exhaustion: the sample loader inflated
# each cached npz once per row (~15 GB x 25,878 accesses for rv_train) and kept
# a second full copy, so one arm peaked at ~30 GB (rv) to ~100 GB (B3).
# Four arms at once plus duplicate resume entries exhausted host RAM and the
# node rebooted repeatedly.
#
# Mitigations:
#   1. loaders inflate each array once and materialise rows lazily (scripts);
#   2. this queue runs at most 2 arms concurrently, split into two phases so
#      the two heaviest (B3/B4) are never paired with each other;
#   3. --wait-gb guards each phase against available host RAM;
#   4. per-arm summary.json skip makes the whole queue restart-safe.
cd /root/AIC
PY=/data/aic/tools/lfm_venv/bin/python
OUT=/data/aic/experiments_910a/LFM_V8/arms
WAIT_GB=${WAIT_GB:-40}

wait_ram() {  # block until >= WAIT_GB free, so a resume never stacks on a live arm
  while true; do
    FREE=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$FREE" -ge "$WAIT_GB" ] && break
    echo "[queue] waiting for RAM: ${FREE}GB free < ${WAIT_GB}GB" >> /data/aic/experiments_910a/LFM_V8/chain.log
    sleep 60
  done
}

train_arm() {  # card name sources steps seed
  [ -f $OUT/$2_s$5/summary.json ] && { echo "[skip] $2_s$5 done"; return; }
  wait_ram
  ASCEND_RT_VISIBLE_DEVICES=$1 $PY /root/AIC/scripts/v8_s_train_multidata.py \
    --sources ${3//,/ } --steps $4 --seed $5 --output-dir $OUT/$2_s$5 > $OUT/$2_s$5.log 2>&1
}
kd_arm() {  # card weight seed
  [ -f $OUT/KD$2_s$3/summary.json ] && { echo "[skip] KD$2_s$3 done"; return; }
  wait_ram
  ASCEND_RT_VISIBLE_DEVICES=$1 $PY /root/AIC/scripts/v8_s_train_kd_pref.py \
    --kd-weight $2 --seed $3 --output-dir $OUT/KD$2_s$3 > $OUT/KD$2_s$3.log 2>&1
}

# phase 1: RV controls (B1 replica vs B2 +rotation) - establishes the baseline
( train_arm 2 B1 rv_native 1200 0; kd_arm 2 0.3 0 ) &
( train_arm 3 B1 rv_native 1200 1; kd_arm 3 0.3 1 ) &
wait
echo "[train] phase 1 done $(date)" >> /data/aic/experiments_910a/LFM_V8/chain.log

# phase 2: multi-source arms (heaviest; two at a time, never B3+B4 together)
( train_arm 2 B3 rv_native,rv_rot,live_train 2000 0; train_arm 4 B2 rv_native,rv_rot 1200 0 ) &
( train_arm 3 B4 live_train 2000 1; kd_arm 4 0.1 1 ) &
wait
echo "[train] phase 2 done $(date)" >> /data/aic/experiments_910a/LFM_V8/chain.log

# phase 3: remaining seeds
( train_arm 2 B4 live_train 2000 0; kd_arm 2 0.1 0 ) &
( train_arm 3 B3 rv_native,rv_rot,live_train 2000 1; train_arm 4 B2 rv_native,rv_rot 1200 1 ) &
wait
echo "[train] all arms done $(date)" >> /data/aic/experiments_910a/LFM_V8/chain.log