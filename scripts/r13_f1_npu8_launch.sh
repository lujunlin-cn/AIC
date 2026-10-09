#!/bin/bash
# R13 F1 NPU 8-arm launcher: waits for the 4B v3e line to wind down, then
# fires one training arm per card (config-level parallelism, no DDP).
# Arms: 3x seed (f3_head bce/official-dequant), dequant ablation, MSE
# (official original), depth-3, audio-channel, pairwise-loss control.
set -u
PY=/data/aic/venvs/qwen35/bin/python3.11
F1DIR=/data/aic/experiments_910a/LFM_V11/r13_f1
V3E=/data/aic/experiments_910a/LFM_V13/q35_zs_scores_12f_v3e
LOG=$F1DIR/npu8_launch.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 0) kill the CPU arms (superseded; frees 36+ cores for NPU feeding)
pids=$(ps -eo pid,cmd | grep r13_f1_rank | grep -v grep | grep "device cpu" | awk '{print $1}')
[ -n "$pids" ] && { kill -9 $pids 2>/dev/null; log "killed CPU arms: $pids"; sleep 3; }

# 1) wait for 4B v3e to finish (>=426 files and no v3e scorer processes)
while true; do
  N=$(ls "$V3E" 2>/dev/null | wc -l)
  R=$(pgrep -fc "q35_zs_scores_12f_v3e" 2>/dev/null || echo 0)
  log "waiting 4B v3e: files=$N running=$R"
  if [ "$N" -ge 426 ] && [ "$R" -eq 0 ]; then break; fi
  # safety: also proceed if only gap-fill remains and files stop growing
  sleep 180
done
log "4B v3e done, starting NPU smoke"

# 2) single-card smoke: subset 300 x 1 epoch, numeric health gate
$PY /root/AIC/scripts/r13_f1_rank.py --device npu --npu-id 0 \
  --loss bce --depth 5 --schedule warmcos --feat rgb --official-dequant \
  --epochs 1 --subset 300 --val-subset 100 --seed 0 --tag npu_smoke \
  > "$F1DIR/npu_smoke.log" 2>&1
SMOKE=$?
if [ $SMOKE -ne 0 ] || ! grep -q "F1_FINAL" "$F1DIR/npu_smoke.log"; then
  log "SMOKE FAILED exit=$SMOKE — abort, CPU arms must be relaunched"
  exit 1
fi
if grep -q "nan" "$F1DIR/npu_smoke.log"; then
  log "SMOKE NaN — abort"
  exit 1
fi
log "smoke PASS, launching 8 arms"

# 3) 8 arms, one per card
launch() {  # card tag extra...
  local card=$1 tag=$2; shift 2
  nohup $PY /root/AIC/scripts/r13_f1_rank.py --device npu --npu-id $card \
    --epochs 3 --val-subset 0 --seed 0 --tag $tag "$@" \
    > "$F1DIR/${tag}.log" 2>&1 &
  log "launched $tag on card $card: $*"
}
launch 0 f3_head_s0      --loss bce --depth 5 --schedule warmcos --feat rgb --official-dequant --seed 0
launch 1 f3_head_s1      --loss bce --depth 5 --schedule warmcos --feat rgb --official-dequant --seed 1
launch 2 f3_head_s2      --loss bce --depth 5 --schedule warmcos --feat rgb --official-dequant --seed 2
launch 3 f1_deq_p0b_s0   --loss bce --depth 5 --schedule warmcos --feat rgb
launch 4 f1_mse_s0       --loss mse --depth 5 --schedule warmcos --feat rgb --official-dequant
launch 5 f1_d3_s0        --loss bce --depth 3 --schedule warmcos --feat rgb --official-dequant
launch 6 f1_audio_s0     --loss bce --depth 5 --schedule warmcos --feat rgb+audio --official-dequant
launch 7 f1_pair_s0      --loss pairwise --depth 5 --schedule warmcos --feat rgb --official-dequant
log "8 arms launched; monitor $F1DIR/*.log"
