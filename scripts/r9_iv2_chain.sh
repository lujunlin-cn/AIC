#!/bin/bash
# R9 IV2 Stage1 slot-pool extraction launch chain: two shards in parallel on
# cards 4 and 5 (parallel-first rule: shard granularity / incremental npz per
# frag / 1 process per card / p0+p1 merge path known to the loader).
# No pgrep/pkill anywhere (self-match kill class, 5 prior occurrences).
set -u
P=/usr/local/python3.11.15/bin/python3
D=/data/aic/experiments_910a/LFM_V11
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /root/AIC
source /usr/local/Ascend/cann/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=4 $P scripts/r9_iv2_slot_feats.py \
  --shard 0 --nshards 2 --out $D/r9_iv2_slot \
  > $D/r9_iv2_slot_s0.log 2>&1 &
PID4=$!
ASCEND_RT_VISIBLE_DEVICES=5 $P scripts/r9_iv2_slot_feats.py \
  --shard 1 --nshards 2 --out $D/r9_iv2_slot \
  > $D/r9_iv2_slot_s1.log 2>&1 &
PID5=$!
echo "launched card4=$PID4 card5=$PID5 $(date)" > $D/r9_iv2_chain.log
wait $PID4 $PID5
echo "both shards done $(date)" >> $D/r9_iv2_chain.log
