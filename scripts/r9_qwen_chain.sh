#!/bin/bash
# R9 Qwen visual-tower full chain: two shards on logical cards 0/1 (physical
# 2/3), then the head trainer on the qwen readout keys (auto-retry with
# --skip-missing).  Absolute paths; no pgrep/pkill.
set -u
P=/usr/local/python3.11.15/bin/python3
D=/data/aic/experiments_910a/LFM_V11
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /root/AIC
source /usr/local/Ascend/cann/set_env.sh

ASCEND_RT_VISIBLE_DEVICES=0 $P scripts/r9_qwen_slot_feats.py \
  --shard 0 --nshards 2 --out $D/r9_qwen_slot \
  > $D/r9_qwen_s0.log 2>&1 &
PIDA=$!
ASCEND_RT_VISIBLE_DEVICES=1 $P scripts/r9_qwen_slot_feats.py \
  --shard 1 --nshards 2 --out $D/r9_qwen_slot \
  > $D/r9_qwen_s1.log 2>&1 &
PIDB=$!
echo "qwen launched $PIDA $PIDB $(date)" > $D/r9_qwen_chain.log
wait $PIDA $PIDB
echo "qwen extraction done $(date)" >> $D/r9_qwen_chain.log
mkdir -p $D/r9_npu
if $P scripts/r9_slot_head_train.py --feat-root $D/r9_qwen_slot \
     --readouts qwen_l32 qwen_l24 \
     --out $D/r9_npu/slot_head_qwen.json > $D/r9_npu/slot_head_qwen.log 2>&1; then
  echo "qwen head DONE $(date)" >> $D/r9_qwen_chain.log
else
  $P scripts/r9_slot_head_train.py --feat-root $D/r9_qwen_slot \
       --readouts qwen_l32 qwen_l24 --skip-missing \
       --out $D/r9_npu/slot_head_qwen.json >> $D/r9_npu/slot_head_qwen.log 2>&1 \
    && echo "qwen head DONE (skip_missing) $(date)" >> $D/r9_qwen_chain.log \
    || echo "qwen head FAILED twice $(date)" >> $D/r9_qwen_chain.log
fi
