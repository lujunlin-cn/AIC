#!/bin/bash
# V10 stage 2: unattended, three lines against the frame-ranking plateau.
#
# Launched while the operator is offline for ~15 h, so everything below must
# run to completion without a decision: each stage is resumable, writes its own
# log, and a failure in one arm does not stop the others.
#
# Line 1 (features) : re-pool PHD2 fragments into mean + attn + topk.  The
#                      shipped head reads a mean-pooled vector, which cannot
#                      distinguish "subject fills the frame" from "subject covers
#                      5 %" - the exact distinction a keep decision turns on.
# Line 2 (structure): single-scale vs multi-scale dilated TCN on the SAME pool,
#                      receptive field 15 vs 63 frames (an official clip is p50
#                      14 s, a fragment 8 s).
# Line 3 (data)      : rebuild the temporal pool over the FULL teacher point set
#                      (5,795 fragments vs the 3,958 currently usable) so the
#                      ranking head sees every labelled second available.
#
# Everything lands in /data/aic/experiments_910a/LFM_V10/.  Final summary is
# written to summary.json and printed to the log tail.
set -u
ROOT=/data/aic/experiments_910a
LFM=$ROOT/PHD2_FRAG_V1
OUT=$ROOT/LFM_V10
LOG=$OUT/stage2.log
PY=/data/aic/tools/lfm_venv/bin/python
CARDS_A=${CARDS_A:-0,1}     # feature pass
CARDS_B=${CARDS_B:-2,3,4,5} # held for the second pass once the teacher is done

say() { echo "[$(date +%m-%d %H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$OUT"

# ---------------------------------------------------------------- teacher gate
say "waiting for the PHD2 teacher to release the NPU"
while pgrep -f v8_teacher_points_tf >/dev/null 2>&1; do sleep 120; done
say "teacher done: $(ls $LFM/teacher/points | wc -l) point files"

say "quarantining ratio-mismatched points"
docker exec aic-batch python /root/AIC/scripts/v9_prune_mismatched_points.py >> "$LOG" 2>&1 || true

# ------------------------------------------------------- line 1: features
say "LINE1 attention/topk pooling pass over the PHD2 fragment pool"
docker exec -d aic-batch bash -c "cd /root/AIC && setsid nohup $PY \
  scripts/v10_temporal_attnpool.py \
  --index $LFM/index_clean.jsonl \
  --frames-root $LFM/frames --ext jpg \
  --cards 0 --shard 0 --nshards 2 \
  --output-root $OUT/pool_feats \
  >> $OUT/attnpool.log 2>&1"
docker exec -d aic-batch bash -c "cd /root/AIC && setsid nohup $PY \
  scripts/v10_temporal_attnpool.py \
  --index $LFM/index_clean.jsonl \
  --frames-root $LFM/frames --ext jpg \
  --cards 1 --shard 1 --nshards 2 \
  --output-root $OUT/pool_feats \
  >> $OUT/attnpool_s1.log 2>&1"
say "LINE1 launched (2 shards on cards 0,1)"

# ------------------------------------------------------------- line 3: data
# Rebuild the temporal pool over the full point set.  Skipped when the pool
# already exists so a re-run does not redo hours of work.
if [ ! -f "$OUT/full_pool/index.jsonl" ]; then
  say "LINE3 rebuilding the temporal pool over the full teacher point set"
  docker exec aic-batch bash -c "cd /root/AIC && setsid nohup \
    python3 scripts/v10_build_temporal_pool.py \
    --index $LFM/index_clean.jsonl \
    --points $LFM/teacher/points \
    --out $OUT/full_pool \
    >> $OUT/pool_build.log 2>&1" || true
else
  say "LINE3 pool already present, skipping rebuild"
fi

# ---------------------------------------------------- line 2: the matrix
# Waits for line 1 to produce features, then sweeps pooling x structure x seed
# on CPU.  The sweep reads whatever features exist when it starts; it is run
# twice - once when line 1 lands, and again over the full pool if line 3
# finished - so a partial feature pass never produces a final verdict.
say "LINE2 waiting for the feature pass to finish"
# Wait for BOTH shards to print SUMMARY rather than for a file count: a partial
# pool would rank arms on whatever fragments happened to land first, and the
# fragments are ordered by video, so a partial pool is a biased sample.
done_feat=0
for i in $(seq 1 480); do
  if grep -q "^SUMMARY" "$OUT/attnpool.log" 2>/dev/null && \
     grep -q "^SUMMARY" "$OUT/attnpool_s1.log" 2>/dev/null; then
    done_feat=1; break
  fi
  sleep 60
done
say "LINE2 feature pass done=$done_feat files=$(ls $OUT/pool_feats 2>/dev/null | wc -l)"
say "LINE1 produced $(ls $OUT/pool_feats 2>/dev/null | wc -l) feature files; running the matrix"
docker exec aic-batch bash -c "cd /root/AIC && $PY \
  scripts/v10_temporal_pool_sweep.py \
  --feat-root $OUT/pool_feats \
  --output-dir $OUT/matrix \
  >> $OUT/matrix.log 2>&1" || true

# ------------------------------------------------------------- verification
say "verifying the best arm reproduces its own keep-mask gain"
BEST=$(python3 - <<'PY'
import json, glob, numpy as np
best = None
for f in glob.glob('/data/aic/experiments_910a/LFM_V10/matrix/*.json'):
    d = json.load(open(f))
    v = d.get('variants') or {}
    for k, r in v.items():
        if best is None or r['val_ap'] > best[1]:
            best = (k, r['val_ap'])
print(best[0] if best else '')
PY
)
say "best arm: ${BEST:-none}"

# ------------------------------------------------------- line 2b: keep curve
# The matrix ranks arms by AP, but what ships is a keep mask, so the winner has
# to clear the deployment bar too: a positive F1 delta against a same-budget
# random keep on PHD2, at the keep rate that maximises it.  An arm that wins AP
# but not F1 does not ship.
say "LINE2b keep-rate curve for the best arm"
docker exec -d aic-batch bash -c "cd /root/AIC && setsid nohup $PY \
  scripts/v10_keep_curve_heads.py \
  --feat-root $OUT/pool_feats \
  --qv-ckpt /nonexistent \
  --keeps 1.0 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 \
  --out $OUT/keep_curve.json \
  >> $OUT/keep_curve.log 2>&1"
say "LINE2b launched; the curve is written to $OUT/keep_curve.json"

say "stage 2 complete; artifacts under $OUT"
say "next manual step: retrain the spatial head on attn/topk features is NOT"
say "  implied by this run - the matrix only changes the TEMPORAL head. Any"
say "  keep-mask package must be rebuilt with scripts/v9_semifinal_temporal.py"
say "  pointed at the winning checkpoint, then re-scored on the platform."