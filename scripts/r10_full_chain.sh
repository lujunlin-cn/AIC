#!/bin/bash
# R10 full-pool AST extraction chain (8 CPU shards) - file-based to avoid
# the nested-quote trap (inline for-loops lost $S through ssh quoting).
set -u
D=/data/aic/experiments_910a/LFM_V11
echo "full extract start $(date)" > $D/r10_audio_full.log
for S in 0 1 2 3 4 5 6 7; do
  python3 /root/AIC/scripts/r10_audio_extract.py --shard $S --nshards 8 \
    --out $D/r10_audio > $D/r10_audio_full_s$S.log 2>&1 &
done
wait
echo "full extract done $(date)" >> $D/r10_audio_full.log
