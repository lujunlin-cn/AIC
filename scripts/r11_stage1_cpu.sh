#!/bin/bash
# R11 stage-1 CPU chain (910A host).  Run AFTER R10 chains have exited.
# Four independent CPU jobs in parallel (each single-process, bounded threads;
# 192-core budget is wide open since R10 is done):
#   S1  dump B3 score table -> Viterbi audit (R11 plan appendix A runbook)
#   O1  E3 event-role features -> scout readout (+V1 feature table, same npz)
#   V0  dense-view premise probe (fail-fast NOT_RUN if cache missing)
# Incremental/constant writes: each job writes its own JSON/npz tree; no
# cross-writes; no media decoding in parallel with anything else.
set -u
D=/data/aic/experiments_910a/LFM_V11
R=/root/AIC
mkdir -p $D/r11
echo "r11 stage1 start $(date)" > $D/r11/stage1.log

# --- job 1: S1 dump + audit (B3 ckpt path: confirm on host, else first match)
B3=$(ls /data/aic/experiments_910a/LFM_V8/*/best.pt 2>/dev/null | head -1)
if [ -n "$B3" ]; then
  ( python3 $R/scripts/r11_s1_dump_scores.py \
      --ckpt "$B3" --samples-dir /data/aic/experiments_910a/LFM_V8/samples \
      --manifest /data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl \
      --tags live_dev live_confirmation rv_dev rv_diag \
      --out $D/r11/s1_score_table.npz > $D/r11/s1_dump.log 2>&1 && \
    python3 -m aic.r11_trajectory --table $D/r11/s1_score_table.npz \
      --dev-pool live_dev --eval-pools live_confirmation rv_dev rv_diag \
      --lams 0 0.25 0.5 1 2 --out $D/r11/s1_audit.json > $D/r11/s1_audit.log 2>&1 ) &
else
  echo "S1: no B3 ckpt under LFM_V8/*/best.pt -> skipped" >> $D/r11/stage1.log
fi

# --- job 2: O1 scout + V1 feature table from cached E3 roles (if present)
E3=$(ls -d /data/aic/experiments_910a/*e3* /data/aic/experiments_910a/*E3* 2>/dev/null | head -1)
if [ -n "$E3" ]; then
  ( python3 $R/scripts/r11_o1_v1_features.py \
      --e3-dir "$E3" --out $D/r11_e3 > $D/r11/o1_scout.log 2>&1 ) &
else
  echo "O1/V1: no cached E3 role dir -> verifier arm stays queued (stage2)" >> $D/r11/stage1.log
fi

# --- job 3: V0 premise probe (fail-fast NOT_RUN is a valid record)
( python3 $R/scripts/r11_v0_dense_premise.py --out $D/r11/v0_premise.json > $D/r11/v0.log 2>&1 ) &

# --- job 4: S2c talking-head pre-audit (free, reads R10 AST-527 cache)
( python3 $R/scripts/r11_s2c_speech_audit.py --out $D/r11/s2c_speech_audit.json > $D/r11/s2c.log 2>&1 ) &

wait
echo "r11 stage1 done $(date)" >> $D/r11/stage1.log
echo "ARTIFACTS: $D/r11/s1_audit.json $D/r11_e3/scout_readout.json $D/r11/v0_premise.json"
