#!/bin/bash
# R12-QV chain v2: dynamic reinforcement scheduling (no shard ever has two
# live workers).  Pilot measured ~1,440 seg/h/worker -> ~45 min per shard.
#   now      : shards 5,6,7 on cards 5,6,7 (resume-safe)
#   after 5/6/7 SHARD_DONE : reinforcements shards 0,1,2 on cards 5,6,7
#   after vote s0/s1 + d6 + concat GATT_FINAL : shard 4 on card 4
#   after e150 GATT_FINAL : shard 3 on card 3
#   all 8 SHARD_DONE : train qv_pair_npu_s0/s1 on cards 0,1
set -u
set -a; source /data/aic/tools/ascend_teacher.env; set +a
PY=/usr/local/python3.11.15/bin/python
R=/root/AIC/scripts
G=/data/aic/experiments_910a/LFM_V11
Q=$G/r12_qv
LOG=$Q/chain2.log
log(){ echo "[qvc2 $(date +%m-%d\ %H:%M:%S)] $*" >> $LOG; }

EX(){ ASCEND_RT_VISIBLE_DEVICES=$1 setsid nohup $PY $R/r12_qv_extract.py \
  --cards $1 --nshards 8 --shard $2 >> $Q/full_shard$2.log 2>&1 & }

sd(){ grep -q SHARD_DONE $Q/full_shard$1.log 2>/dev/null; }
arm_final(){ grep -q GATT_FINAL $G/r12_gatt/$1.log 2>/dev/null; }

# phase 1
EX 5 5; EX 6 6; EX 7 7
log "phase1: shards 5,6,7 launched on cards 5,6,7"

# phase 2: reinforcements shards 0,1,2 after shards 5,6,7 finish
for i in 5 6 7; do
  ( while ! sd $i; do sleep 120; done
    EX $i $((i - 5))
    echo "[qvc2 $(date +%m-%d\ %H:%M:%S)] reinforcement shard $((i - 5)) launched card $i" >> $LOG ) &
done
log "phase2 scheduled: shards 0,1,2 after 5/6/7"

# phase 3: shard 4 after the four 50-epoch arms
( while ! ( arm_final lfm_vote_npu_s0 && arm_final lfm_vote_npu_s1 && \
            arm_final lfm_pair_d6_npu_s0 && arm_final lfm_concat_npu_s0 ); do sleep 180; done
  EX 4 4
  echo "[qvc2 $(date +%m-%d\ %H:%M:%S)] shard 4 launched card 4" >> $LOG ) &
log "phase3 scheduled: shard 4 after vote/d6/concat arms"

# phase 4: shard 3 after the e150 arm
( while ! arm_final lfm_pair_e150_npu_s0; do sleep 180; done
  EX 3 3
  echo "[qvc2 $(date +%m-%d\ %H:%M:%S)] shard 3 launched card 3" >> $LOG ) &
log "phase4 scheduled: shard 3 after e150 arm"

# phase 5: training
while true; do
  D=0; for i in 0 1 2 3 4 5 6 7; do sd $i && D=$((D + 1)); done
  [ "$D" -ge 8 ] && break
  log "shards done: $D/8"
  sleep 600
done
log "all shards complete -> training 2 seeds (cards 0,1)"
for s in 0 1; do
  ASCEND_RT_VISIBLE_DEVICES=$s setsid nohup $PY $R/r12_qv_train.py \
    --seed $s --tag qv_pair_npu_s$s --device npu > $Q/qv_pair_npu_s$s.log 2>&1 &
done
log "QV_TRAIN_LAUNCHED"
