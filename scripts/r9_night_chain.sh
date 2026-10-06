#!/bin/bash
# R9 night chain (user sleeping, 10h window): wait for the two extraction
# shards -> head training (retry once with --skip-missing if any npz is
# missing).  Absolute paths; no pgrep/pkill; single process after the
# extraction chain exits.
D=/data/aic/experiments_910a/LFM_V11
P=/usr/local/python3.11.15/bin/python3
LOG=$D/r9_night_chain.log
echo "night chain started $(date)" > $LOG
while true; do
  if grep -q "DONE shard 0" $D/r9_iv2_slot_s0.log 2>/dev/null && \
     grep -q "DONE shard 1" $D/r9_iv2_slot_s1.log 2>/dev/null; then
    break
  fi
  sleep 180
done
echo "extraction shards done $(date)" >> $LOG
mkdir -p $D/r9_npu
cd /root/AIC
if $P scripts/r9_slot_head_train.py > $D/r9_npu/slot_head.log 2>&1; then
  echo "head training DONE $(date)" >> $LOG
else
  echo "head training retry with --skip-missing $(date)" >> $LOG
  $P scripts/r9_slot_head_train.py --skip-missing >> $D/r9_npu/slot_head.log 2>&1 \
    && echo "head training DONE (skip_missing) $(date)" >> $LOG \
    || echo "head training FAILED twice $(date)" >> $LOG
fi
