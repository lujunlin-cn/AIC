#!/bin/bash
# V9 P1-4 arms.  Two questions, five arms:
#
#   Q1  does a KD-only PHD2 stream help at all?
#       A0_ctrl   B3 recipe, no PHD2
#       A1_raw    + PHD2 teacher points as-is, weight 0.5
#       A1_smooth + PHD2 teacher points smoothed over the fragment, weight 0.5
#   Q2  if it helps, is more of it better, and is the KD-v2 in-pool term
#       separable from the KD-only stream?
#       A2_smooth weight 1.0
#       A3_kdv2   in-pool KD term only (kd_weight 0.3), no PHD2
#
# A1_raw vs A1_smooth is the arm this round added: the teacher audit measured a
# p90 frame-to-frame point jump of 0.25-0.30 on 1 s keyframes, and V8's KD-v2
# gained only +0.003/+0.007 - consistent with the student averaging teacher
# jitter rather than teacher signal.
#
# CPU on purpose: the 32B teacher owns NPU cards 2-5 and a co-resident NPU
# process pushed its per-batch wall time from 12 s to 33 s.  The head is 1.5M
# params, so CPU is not the bottleneck.  Threads are pinned per arm.
set -u
PY=/data/aic/tools/lfm_venv/bin/python
V8=/data/aic/experiments_910a/LFM_V8/samples
RAW=/data/aic/experiments_910a/PHD2_FRAG_V1/samples_s0
SMO=/data/aic/experiments_910a/PHD2_FRAG_V1/samples_s1
OUT=/data/aic/experiments_910a/LFM_V9/arms
STEPS=${STEPS:-3000}
SEED=${SEED:-0}
mkdir -p "$OUT"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

# setsid puts each arm in its own session/process group.  Without it the arms
# were all killed together partway through step 200: the shell that started them
# was reaped and the whole process group went with it, with no OOM and 711 GB
# free.  Reaping the launcher must not reap the training.
run() {  # name pool-root extra-args...
  local name=$1 root=$2; shift 2
  setsid nohup "$PY" scripts/v9_s_train_kd.py \
      --samples-dir "$V8" --kd-samples-dir "$root" \
      --steps "$STEPS" --seed "$SEED" --device cpu \
      --output-dir "$OUT/$name" --save-per-video "$@" \
      > "$OUT/$name.log" 2>&1 < /dev/null &
  echo "launched $name pid=$!"
}

run A0_ctrl       "$V8" --kd-pools            --kd-pool-weight 0.0
run A1_raw        "$RAW" --kd-pools phd2_train --kd-pool-weight 0.5
run A1_smooth     "$SMO" --kd-pools phd2_train --kd-pool-weight 0.5
run A2_smooth_w10 "$SMO" --kd-pools phd2_train --kd-pool-weight 1.0
run A3_kdv2       "$V8" --kd-pools            --kd-pool-weight 0.0 --kd-weight 0.3
# Poll instead of `wait`: this launcher is expected to exit immediately so the
# caller is not blocked for the length of the training run.
for i in $(seq 1 "${POLL:-60}"); do
  n_done=$(grep -l SUMMARY "$OUT"/*.log 2>/dev/null | wc -l)
  [ "$n_done" -ge 5 ] && break
  sleep "${POLL_EVERY:-60}"
done
echo "SUMMARY_POLL done=$(grep -l SUMMARY "$OUT"/*.log 2>/dev/null | wc -l)/5"
for n in A0_ctrl A1_raw A1_smooth A2_smooth_w10 A3_kdv2; do
  printf '%-14s ' "$n"; grep SUMMARY "$OUT/$n.log" 2>/dev/null || tail -1 "$OUT/$n.log"
done