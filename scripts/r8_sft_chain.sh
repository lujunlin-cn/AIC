#!/bin/bash
# R8 VLM_SFT launch chain: frame prep (CPU, once) -> two preregistered lr
# configs IN PARALLEL on cards 2 and 3 (parallel-first rule).  The chain
# aborts before training if any frame is missing after prep.
set -u
P=/usr/local/python3.11.15/bin/python3
D=/data/aic/experiments_910a/LFM_V11/r8_npu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /root/AIC

$P scripts/r8_sft_prep.py > $D/sft_prep.log 2>&1
if ! grep -q "missing 0" $D/sft_prep.log; then
  echo "PREP_INCOMPLETE_ABORT: $(tail -2 $D/sft_prep.log)" > $D/sft_chain.log
  exit 1
fi
echo "prep done, launching two configs $(date)" > $D/sft_chain.log

source /usr/local/Ascend/cann/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=2 $P scripts/r8_vlm_sft_train.py --lr 5e-6 --card 2 \
  --out $D/vlm_sft_lr5e6.json --adapter-dir $D/sft_adapters/lr5e6 \
  > $D/vlm_sft_lr5e6.log 2>&1 &
PID2=$!
ASCEND_RT_VISIBLE_DEVICES=3 $P scripts/r8_vlm_sft_train.py --lr 1e-5 --card 3 \
  --out $D/vlm_sft_lr1e5.json --adapter-dir $D/sft_adapters/lr1e5 \
  > $D/vlm_sft_lr1e5.log 2>&1 &
PID3=$!
echo "launched card2=$PID2 card3=$PID3" >> $D/sft_chain.log
wait $PID2 $PID3
echo "both configs finished $(date)" >> $D/sft_chain.log
