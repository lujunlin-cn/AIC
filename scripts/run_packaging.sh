#!/bin/bash
# R12 packaging chain: wait for 8 GATT finals -> pick winner feature + best
# ckpt -> build the official submission package (single variable vs 34.88).
set -u
GATT_DIR=/data/aic/experiments_910a/LFM_V11/r12_gatt
P0B_DIR=/data/aic/experiments_910a/LFM_V11/r12_p0b
PY=/usr/local/python3.11.15/bin/python
LOG=$GATT_DIR/packaging.log
log() { echo "[pkg $(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }

LOSS=$($PY -c "import json; print(json.load(open('$P0B_DIR/verdict.json'))['frozen_loss'])" 2>/dev/null || echo pairwise)

# 1) wait for 8 finals
while true; do
  N=$(grep -h GATT_FINAL "$GATT_DIR"/{vmae,lfm}_${LOSS}_npu_s[0-3].log 2>/dev/null | wc -l)
  [ "$N" -ge 8 ] && break
  log "waiting GATT finals: $N/8 (loss=$LOSS)"
  sleep 180
done
log "8/8 GATT finals present (loss=$LOSS)"

# 2) winner feature by mean best_dev_ap; best seed within winner
read FEAT SEED CKPT <<< $($PY - <<PYEOF
import json, glob, statistics
best = {}
for fp in glob.glob("$GATT_DIR/*_${LOSS}_npu_s[0-3].jsonl"):
    tag = fp.split("/")[-1][:-6]
    feat = tag.split("_")[0]
    top = -1
    for line in open(fp):
        r = json.loads(line)
        if r.get("best_dev_ap") is not None:
            top = max(top, r["best_dev_ap"])
    best[tag] = (feat, top)
means = {}
for feat in ("vmae", "lfm"):
    vals = [v[1] for t, v in best.items() if v[0] == feat]
    means[feat] = statistics.mean(vals) if vals else -1
win = "lfm" if means["lfm"] >= means["vmae"] else "vmae"
top_tag, top_val = max(((t, v[1]) for t, v in best.items() if v[0] == win), key=lambda x: x[1])
seed = top_tag.rsplit("_s", 1)[1]
ckpt = "$GATT_DIR/" + top_tag + "_best.pt"
print(win, seed, ckpt)
PYEOF
)
log "winner feat=$FEAT (seed $SEED) ckpt=$CKPT"

# 3) build the package
SUB=R12_GATT_${FEAT^^}_SEMIFINAL
$PY /root/AIC/scripts/r12_gatt_package.py --ckpt "$CKPT" --feat "$FEAT" \
  --out-dir /data/aic/semifinal_20261001/submissions/$SUB --sub-id $SUB \
  >> "$LOG" 2>&1
log "package built: /data/aic/semifinal_20261001/submissions/$SUB/$SUB.zip"
rm -rf "$GATT_DIR/pkg_dryrun"
log "chain done"
