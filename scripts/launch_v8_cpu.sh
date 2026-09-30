#!/bin/bash
# V8 arm queue on CPU - fallback while the NPU driver is unavailable
# (aclInit 507899 on 2026-09-30 after a host reboot).
#
# Rationale: the head is 1.5 M parameters and the features are pre-cached, so a
# 1200-2000 step arm costs ~10-20 min per core. The host has 192 cores and other
# tenants are active, so each arm is pinned to 2 threads and at most ARMS_CONC
# arms run at once.
#
# Restart-safe: an arm with summary.json is skipped. The resume entry point
# holds a flock, so duplicate queues cannot stack (that bug, plus the npz
# per-row inflation, is what exhausted host RAM earlier).
cd /root/AIC
PY=/data/aic/tools/lfm_venv/bin/python
OUT=/data/aic/experiments_910a/LFM_V8/arms
LOG=/data/aic/experiments_910a/LFM_V8
CONC=${CONC:-8}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

mkdir -p $OUT
wait_ram() {
  while true; do
    FREE=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    [ "$FREE" -ge 40 ] && break
    echo "[cpu-queue] waiting for RAM: ${FREE}GB" >> $LOG/chain.log
    sleep 60
  done
}

arm() {  # name script sources steps seed kdweight
  local name=$1 script=$2 sources=$3 steps=$4 seed=$5 kd=${6:-}
  [ -f $OUT/$name/summary.json ] && { echo "[skip] $name"; return; }
  wait_ram
  if [ "$script" = "kd" ]; then
    $PY -u /root/AIC/scripts/v8_s_train_kd_pref.py --device cpu \
      --kd-weight $kd --seed $seed --output-dir $OUT/$name > $OUT/$name.log 2>&1
  else
    $PY -u /root/AIC/scripts/v8_s_train_multidata.py --device cpu \
      --sources ${sources//,/ } --steps $steps --seed $seed --output-dir $OUT/$name > $OUT/$name.log 2>&1
  fi
  echo "[cpu-queue] $name done $(date)" >> $LOG/chain.log
}

# concurrency gate: run jobs in waves of CONC
JOBS=(
  "B1_s0|multi|rv_native|1200|0|"
  "B1_s1|multi|rv_native|1200|1|"
  "B2_s0|multi|rv_native,rv_rot|1200|0|"
  "B2_s1|multi|rv_native,rv_rot|1200|1|"
  "B3_s0|multi|rv_native,rv_rot,live_train|2000|0|"
  "B3_s1|multi|rv_native,rv_rot,live_train|2000|1|"
  "B4_s0|multi|live_train|2000|0|"
  "B4_s1|multi|live_train|2000|1|"
  "KD0.1_s0|kd||1200|0|0.1"
  "KD0.1_s1|kd||1200|1|0.1"
  "KD0.3_s0|kd||1200|0|0.3"
  "KD0.3_s1|kd||1200|1|0.3"
)
echo "[cpu-queue] start $(date) conc=$CONC" >> $LOG/chain.log
i=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r name script sources steps seed kd <<< "$spec"
  arm "$name" "$script" "$sources" "$steps" "$seed" "$kd" &
  i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then wait; fi
done
wait
echo "[cpu-queue] all arms done $(date)" >> $LOG/chain.log