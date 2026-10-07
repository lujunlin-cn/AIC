#!/bin/bash
# R10 ASR full-pool transcription chain: 8 CPU shards over ALL 3,917 frozen
# fragments (runs alongside the AST extraction; both CPU, threads stay
# within the 192-core budget).  Incremental json per fragment (resume-safe;
# pilot jsons are reused as-is).
set -u
D=/data/aic/experiments_910a/LFM_V11
echo "asr full start $(date)" > $D/r10_asr_full.log
for S in 0 1 2 3 4 5 6 7; do
  python3 /root/AIC/scripts/r10_asr_probe.py --shard $S --nshards 8 --limit 0 \
    > $D/r10_asr_full_s$S.log 2>&1 &
done
wait
echo "asr full done $(date)" >> $D/r10_asr_full.log
