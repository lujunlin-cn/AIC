#!/bin/bash
# R12 auto-chain v2: 8-card NPU edition.
# Wait for 8 P0B_FINALs (pair_npu_s0..3 + mse_npu_s0..3) -> 4-seed mean verdict
# -> freeze loss -> launch GATT 8 arms on NPU (vmae/lfm x seed 0..3).
set -u
P0B_DIR=/data/aic/experiments_910a/LFM_V11/r12_p0b
GATT_DIR=/data/aic/experiments_910a/LFM_V11/r12_gatt
PY=/usr/local/python3.11.15/bin/python
LOG=$P0B_DIR/chain.log
mkdir -p "$GATT_DIR"
log() { echo "[chain $(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }

# 1) wait for all 8 finals
while true; do
  N=$(grep -h P0B_FINAL "$P0B_DIR"/{pair,mse}_npu_s[0-3].log 2>/dev/null | wc -l)
  [ "$N" -ge 8 ] && break
  log "waiting finals: $N/8"
  sleep 180
done
log "8/8 finals present"

# 2) 4-seed mean verdict; tie (<0.5) -> bce (TRINITY recipe)
LOSS=$($PY - <<PYEOF
import json, re, glob, statistics
finals = {"pair": [], "mse": []}
for fp in glob.glob("$P0B_DIR/{pair,mse}_npu_s[0-3].log"):
    for line in open(fp):
        if line.startswith("P0B_FINAL"):
            r = json.loads(line.split(" ", 1)[1])
            finals["pair" if "pair" in r["tag"] else "mse"].append(r["test_map50"])
pm = statistics.mean(finals["pair"])
mm = statistics.mean(finals["mse"])
diff = pm - mm
loss = "pairwise" if diff >= 0.5 else ("bce" if abs(diff) < 0.5 else "mse")
verdict = {"pair_mean": round(pm, 2), "mse_mean": round(mm, 2), "diff": round(diff, 2),
           "pair_std": round(statistics.pstdev(finals["pair"]), 2),
           "mse_std": round(statistics.pstdev(finals["mse"]), 2), "frozen_loss": loss}
open("$P0B_DIR/verdict.json", "w").write(json.dumps(verdict, indent=1))
print(loss)
PYEOF
)
log "frozen loss=$LOSS (verdict.json written)"

# 3) launch GATT 8 arms on NPU (vmae/lfm x seed 0..3), idempotent by GATT_FINAL
launch() {  # feat seed card
  local TAG="${1}_${LOSS}_npu_s$2"
  grep -q GATT_FINAL "$GATT_DIR/$TAG.jsonl" 2>/dev/null && { log "skip $TAG"; return; }
  setsid nohup bash -c "source /data/aic/tools/ascend_teacher.env; export ASCEND_RT_VISIBLE_DEVICES=$3; /usr/local/python3.11.15/bin/python /root/AIC/scripts/r12_gatt_vmaef_train.py --feat $1 --loss $LOSS --epochs 50 --seed $2 --device npu --tag $TAG" >> "$GATT_DIR/$TAG.log" 2>&1 &
  log "launched $TAG on card $3"
}
launch vmae 0 0
launch vmae 1 1
launch vmae 2 2
launch vmae 3 3
launch lfm 0 4
launch lfm 1 5
launch lfm 2 6
launch lfm 3 7
log "GATT 8 arms launched; chain done"
