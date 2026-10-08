#!/bin/bash
# R12-QV chain: full 8-shard extraction -> 2-seed training, unattended.
#   cards 6,7 start shards 6,7 immediately;
#   card 5 continues shard 5 (after the pilot) ; cards 0-4 wait for the
#   five UV/P3 arms to finish, then take shards 0-4;
#   when all 8 shards are SHARD_DONE -> train qv_pair_npu_s0/s1 (cards 0,1).
set -u
set -a; source /data/aic/tools/ascend_teacher.env; set +a
PY=/usr/local/python3.11.15/bin/python
R=/root/AIC/scripts
G=/data/aic/experiments_910a/LFM_V11
Q=$G/r12_qv
LOG=$Q/chain.log
mkdir -p $Q
log(){ echo "[qvc $(date +%m-%d\ %H:%M:%S)] $*" >> $LOG; }

ARM_LOGS="$G/r12_gatt/lfm_vote_npu_s0.log $G/r12_gatt/lfm_vote_npu_s1.log $G/r12_gatt/lfm_pair_d6_npu_s0.log $G/r12_gatt/lfm_pair_e150_npu_s0.log $G/r12_gatt/lfm_concat_npu_s0.log"
arms_done(){ [ "$(grep -h GATT_FINAL $ARM_LOGS 2>/dev/null | wc -l)" -ge 5 ]; }

# cards 6,7: start now (free after the P2 packs)
for i in 6 7; do
  ASCEND_RT_VISIBLE_DEVICES=$i setsid nohup $PY $R/r12_qv_extract.py \
    --cards $i --nshards 8 --shard $i > $Q/full_shard$i.log 2>&1 &
  log "shard $i worker launched card $i (immediate)"
done

# card 5: continue shard 5 after the pilot exits
( while ! grep -q PILOT_DONE $Q/pilot_shard5.log 2>/dev/null; do sleep 60; done
  ASCEND_RT_VISIBLE_DEVICES=5 setsid nohup $PY $R/r12_qv_extract.py \
    --cards 5 --nshards 8 --shard 5 > $Q/full_shard5.log 2>&1 ) &
log "shard 5 continuation scheduled (after pilot)"

# cards 0-4: wait for the five arms
for i in 0 1 2 3 4; do
  ( while ! arms_done; do sleep 120; done
    ASCEND_RT_VISIBLE_DEVICES=$i setsid nohup $PY $R/r12_qv_extract.py \
      --cards $i --nshards 8 --shard $i > $Q/full_shard$i.log 2>&1 ) &
done
log "shards 0-4 scheduled (after arms)"

# wait all 8 shards
while true; do
  D=0; for i in 0 1 2 3 4 5 6 7; do grep -q SHARD_DONE $Q/full_shard$i.log 2>/dev/null && D=$((D+1)); done
  [ "$D" -ge 8 ] && break
  log "extract shards done: $D/8"
  sleep 600
done
log "extraction complete -> training 2 seeds (cards 0,1)"
for s in 0 1; do
  ASCEND_RT_VISIBLE_DEVICES=$s setsid nohup $PY $R/r12_qv_train.py \
    --seed $s --tag qv_pair_npu_s$s --device npu > $Q/qv_pair_npu_s$s.log 2>&1 &
done
log "QV_TRAIN_LAUNCHED"
