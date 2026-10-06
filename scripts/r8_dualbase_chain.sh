#!/bin/bash
# R8 dual-base launch: two parallel CPU processes (bases A and B).
# Absolute script paths; no pgrep/pkill anywhere (self-match kills the
# session - 5th occurrence of this class of bug).
cd /root/AIC
D=/data/aic/experiments_910a/LFM_V11/r8_npu
nohup python3 /root/AIC/scripts/r8_temporal_dualbase.py --base A \
  --out $D/dualbase_A.json > $D/dualbase_A.log 2>&1 < /dev/null &
nohup python3 /root/AIC/scripts/r8_temporal_dualbase.py --base B \
  --out $D/dualbase_B.json > $D/dualbase_B.log 2>&1 < /dev/null &
echo "launched $(date)" > $D/dualbase_chain.log
wait
echo "both bases done $(date)" >> $D/dualbase_chain.log
