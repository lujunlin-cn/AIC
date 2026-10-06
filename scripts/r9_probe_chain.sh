#!/bin/bash
# R9 media-probe chain: 8 CPU shards in parallel, then the signal screen.
# No NPU, no pgrep/pkill; incremental npz per frag.
set -u
D=/data/aic/experiments_910a/LFM_V11
cd /root/AIC
for S in 0 1 2 3 4 5 6 7; do
  python3 scripts/r9_media_probe.py --shard $S --nshards 8 \
    --out $D/r9_signal_probe > $D/r9_probe_s$S.log 2>&1 &
done
echo "probe launched 8 shards $(date)" > $D/r9_probe_chain.log
wait
echo "probe done $(date)" >> $D/r9_probe_chain.log
python3 scripts/r9_signal_screen.py > $D/r9_npu/signal_screen.log 2>&1 \
  && echo "screen DONE $(date)" >> $D/r9_probe_chain.log \
  || echo "screen FAILED $(date)" >> $D/r9_probe_chain.log
