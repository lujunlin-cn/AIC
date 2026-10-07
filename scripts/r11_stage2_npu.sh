#!/bin/bash
# R11 stage-2 NPU chain (910A host).  Preconditions: stage-1 done; NPU free
# (R10 closed with 0 NPU spend).  Two gated arms; each aborts fast on the
# first failing gate and writes its verdict JSON before touching the next.
set -u
D=/data/aic/experiments_910a/LFM_V11
R=/root/AIC
mkdir -p $D/r11
echo "r11 stage2 start $(date)" > $D/r11/stage2.log

# --- arm A: S2a detector smoke (<=4 h stop-loss) ---------------------------
# GroundingDINO first (repo lineage), NPU parity gate on 10 sources; on any
# operator/import failure fall back to the SigLIP-proposal downgrade path
# and mark the arm accordingly.
( python3 $R/scripts/r11_s2a_detector_smoke.py \
    --out $D/r11/s2a_smoke.json > $D/r11/s2a_smoke.log 2>&1; \
  echo "s2a smoke exit=$? $(date)" >> $D/r11/stage2.log ) &

# --- arm B: V1 teacher event roles (only if stage-1 found no cached E3) ----
if [ ! -d "$(ls -d /data/aic/experiments_910a/*e3* /data/aic/experiments_910a/*E3* 2>/dev/null | head -1)" ]; then
  ( python3 $R/scripts/qwen_event_segments.py \
      --output /data/aic/experiments_910a/LFM_V11/r11_e3_roles \
      --model /data/aic/models/Qwen3-VL-32B --tp 4 \
      > $D/r11/v1_reinfer.log 2>&1; \
    echo "v1 re-infer exit=$? $(date)" >> $D/r11/stage2.log ) &
else
  echo "V1: cached E3 roles present -> re-inference skipped (0 NPU-h)" >> $D/r11/stage2.log
fi

wait
echo "r11 stage2 done $(date)" >> $D/r11/stage2.log
