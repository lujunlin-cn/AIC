#!/bin/bash
set -u
D=/data/aic/experiments_910a/LFM_V11
echo "asr chain start $(date)" > $D/r10_asr_chain.log
for S in 0 1 2 3; do
  python3 /root/AIC/scripts/r10_asr_probe.py --shard $S --nshards 4 --limit 0 > $D/r10_asr_s$S.log 2>&1 &
done
wait
echo "asr chain done $(date)" >> $D/r10_asr_chain.log
