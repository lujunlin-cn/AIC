#!/bin/bash
# R9 Stage2 full chain: two shards on logical cards 2/3 (physical 4/5), then
# the head trainer on the SAME readout keys (auto-retry with --skip-missing).
# Absolute paths; no pgrep/pkill.
set -u
P=/usr/local/python3.11.15/bin/python3
D=/data/aic/experiments_910a/LFM_V11
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /root/AIC
source /usr/local/Ascend/cann/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=2 $P scripts/r9_iv2s2_slot_feats.py \
  --shard 0 --nshards 2 --out $D/r9_iv2s2_slot \
  > $D/r9_iv2s2_s0.log 2>&1 &
PIDA=$!
ASCEND_RT_VISIBLE_DEVICES=3 $P scripts/r9_iv2s2_slot_feats.py \
  --shard 1 --nshards 2 --out $D/r9_iv2s2_slot \
  > $D/r9_iv2s2_s1.log 2>&1 &
PIDB=$!
echo "s2 launched $PIDA $PIDB $(date)" > $D/r9_s2_chain.log
wait $PIDA $PIDB
echo "s2 extraction done $(date)" >> $D/r9_s2_chain.log
mkdir -p $D/r9_npu
if $P scripts/r9_slot_head_train.py --feat-root $D/r9_iv2s2_slot \
     --readouts mean768 m1408_l m1408_m5 \
     --out $D/r9_npu/slot_head_s2.json > $D/r9_npu/slot_head_s2.log 2>&1; then
  echo "s2 head DONE $(date)" >> $D/r9_s2_chain.log
else
  $P scripts/r9_slot_head_train.py --feat-root $D/r9_iv2s2_slot \
       --readouts mean768 m1408_l m1408_m5 --skip-missing \
       --out $D/r9_npu/slot_head_s2.json >> $D/r9_npu/slot_head_s2.log 2>&1 \
    && echo "s2 head DONE (skip_missing) $(date)" >> $D/r9_s2_chain.log \
    || echo "s2 head FAILED twice $(date)" >> $D/r9_s2_chain.log
fi
