#!/bin/bash
# R12 auto-chain: wait for both P0-b finals -> freeze loss by verdict ->
# launch full GATT/VMAE-F dual arms (vmae + lfm) per side.
# Launched detached (setsid). Idempotent: skips arms whose GATT_FINAL exists.
set -u
P0B_DIR=/data/aic/experiments_910a/LFM_V11/r12_p0b
GATT_DIR=/data/aic/experiments_910a/LFM_V11/r12_gatt
LFM_POOL=/data/aic/experiments_910a/LFM_V11/r12_p0c/phd2_lfm_pooled
PY=/usr/local/python3.11.15/bin/python
LOG=$P0B_DIR/chain.log
mkdir -p "$GATT_DIR"

log() { echo "[chain $(date +%H:%M)] $*" >> "$LOG"; }

# 1) wait for both finals
while true; do
  P=$(grep -h P0B_FINAL "$P0B_DIR/pair_full_s0.log" 2>/dev/null | tail -1)
  M=$(grep -h P0B_FINAL "$P0B_DIR/mse_full_s0.log" 2>/dev/null | tail -1)
  [ -n "$P" ] && [ -n "$M" ] && break
  sleep 300
done
log "both finals present: $P | $M"

# 2) freeze loss: winner by test_map50; tie (<0.5) -> bce (TRINITY recipe)
LOSS=$($PY - "$P" "$M" <<'PYEOF'
import json, sys
p = json.loads(sys.argv[1].split(" ", 1)[1])
m = json.loads(sys.argv[2].split(" ", 1)[1])
diff = p["test_map50"] - m["test_map50"]
loss = "pairwise" if diff >= 0.5 else ("bce" if abs(diff) < 0.5 else "mse")
print(loss)
PYEOF
)
log "frozen loss=$LOSS"

# 3) wait for lfm pooled extraction to finish (32 shards or 8 cards, any layout)
while true; do
  N=$(ls "$LFM_POOL" 2>/dev/null | grep -c '\.npz$')
  RUNNING=$(ps aux | grep -c "[p]0c_phd2_lfm_pooled")
  [ "$RUNNING" -eq 0 ] && break
  log "waiting lfm pool: $N npz, $RUNNING procs"
  sleep 600
done
log "lfm pool ready: $(ls "$LFM_POOL" | grep -c '\.npz$') npz"

# 4) launch full arms: vmae + lfm x frozen loss (skip if already final)
launch() {  # feat seed
  local FEAT=$1 SEED=$2 TAG="${3}_${LOSS}_s$2"
  grep -q GATT_FINAL "$GATT_DIR/$TAG.jsonl" 2>/dev/null && { log "skip $TAG (final exists)"; return; }
  setsid nohup $PY /root/AIC/scripts/r12_gatt_vmaef_train.py \
    --feat "$FEAT" --loss "$LOSS" --epochs 50 --seed "$SEED" --tag "$TAG" \
    >> "$GATT_DIR/$TAG.log" 2>&1 &
  log "launched $TAG"
}
launch vmae 0 vmae
launch vmae 1 vmae
launch lfm 0 lfm
launch lfm 1 lfm
log "all arms launched; chain done"
