#!/bin/bash
# R12 P1/P2/P3 launch (10-08).  8x 910B all-in.
#   P2   : keep 0.70 / 0.60 packs from the frozen 35.17 ckpt (zero training)
#   P1-UV: PHD2 user-vote labels, 2 arms (cards 0,1)
#   P3   : d6 / epochs150 / lfm+vmae concat, 1 arm each (cards 2,3,4)
#   P1-QV: labels -> 20-seg pilot (card 5); r12_qv_chain.sh takes over after.
set -u
set -a; source /data/aic/tools/ascend_teacher.env; set +a
PY=/usr/local/python3.11.15/bin/python
R=/root/AIC/scripts
G=/data/aic/experiments_910a/LFM_V11
SUB=/data/aic/semifinal_20261001/submissions
LOG=$G/r12_p1_launch.log
mkdir -p $G/r12_qv
log(){ echo "[p1 $(date +%m-%d\ %H:%M:%S)] $*" >> $LOG; }

CKPT=$G/r12_gatt/lfm_pairwise_npu_s1_best.pt

# ---- P2: two keep-depth packs (single card, minutes each) ----
log "P2 keep0.70 pack start"
ASCEND_RT_VISIBLE_DEVICES=6 $PY $R/r12_gatt_package.py --ckpt $CKPT --feat lfm \
  --keep-frac 0.70 --out-dir $SUB/R12_GATT_LFM_KEEP070SEMIFINAL \
  --sub-id R12_GATT_LFM_KEEP070SEMIFINAL \
  --parent "R12_GATT_LFM_SEMIFINAL (official 35.17)" >> $LOG 2>&1
log "P2 keep0.70 done rc=$?"
log "P2 keep0.60 pack start"
ASCEND_RT_VISIBLE_DEVICES=6 $PY $R/r12_gatt_package.py --ckpt $CKPT --feat lfm \
  --keep-frac 0.60 --out-dir $SUB/R12_GATT_LFM_KEEP060SEMIFINAL \
  --sub-id R12_GATT_LFM_KEEP060SEMIFINAL \
  --parent "R12_GATT_LFM_SEMIFINAL (official 35.17)" >> $LOG 2>&1
log "P2 keep0.60 done rc=$?"

# ---- P1-UV arms (cards 0,1) ----
for s in 0 1; do
  ASCEND_RT_VISIBLE_DEVICES=$s setsid nohup $PY $R/r12_gatt_vmaef_train.py \
    --feat lfm --loss pairwise --labels vote --seed $s --tag lfm_vote_npu_s$s --device npu \
    > $G/r12_gatt/lfm_vote_npu_s$s.log 2>&1 &
  log "UV arm s$s launched card $s"
done

# ---- P3 arms (cards 2,3,4) ----
ASCEND_RT_VISIBLE_DEVICES=2 setsid nohup $PY $R/r12_gatt_vmaef_train.py \
  --feat lfm --loss pairwise --n-layers 6 --seed 0 --tag lfm_pair_d6_npu_s0 --device npu \
  > $G/r12_gatt/lfm_pair_d6_npu_s0.log 2>&1 &
log "P3 d6 arm launched card 2"
ASCEND_RT_VISIBLE_DEVICES=3 setsid nohup $PY $R/r12_gatt_vmaef_train.py \
  --feat lfm --loss pairwise --epochs 150 --seed 0 --tag lfm_pair_e150_npu_s0 --device npu \
  > $G/r12_gatt/lfm_pair_e150_npu_s0.log 2>&1 &
log "P3 e150 arm launched card 3"
ASCEND_RT_VISIBLE_DEVICES=4 setsid nohup $PY $R/r12_gatt_vmaef_train.py \
  --feat concat --loss pairwise --seed 0 --tag lfm_concat_npu_s0 --device npu \
  > $G/r12_gatt/lfm_concat_npu_s0.log 2>&1 &
log "P3 concat arm launched card 4"

# ---- P1-QV: labels then 20-seg pilot (card 5) ----
$PY $R/r12_qv_build_labels.py >> $LOG 2>&1
log "QV labels built rc=$?"
ASCEND_RT_VISIBLE_DEVICES=5 setsid nohup $PY $R/r12_qv_extract.py \
  --cards 5 --nshards 8 --shard 5 --limit 20 \
  > $G/r12_qv/pilot_shard5.log 2>&1 &
log "QV pilot launched card 5 (shard 5/8, 20 segs)"
echo "LAUNCH_DONE"
