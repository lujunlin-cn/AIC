#!/bin/bash
# V7: smoke grad-check rerun (NPU 3) + S head training (NPU 2) + TVSum 1fps extraction 2 shards (NPU 4,5)
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
nohup $PY scripts/v7_e0_train_smoke.py --cards 3 --steps 60 --output $E/e0/train_smoke.json > $E/e0/train_smoke.log 2>&1 &
nohup $PY scripts/v7_s_train_head.py --device npu --seed 0 --output-dir $E/s_head_s0 > $E/s_head_s0.log 2>&1 &
nohup $PY scripts/v7_t_extract_tvsum.py --cards 4 --shard 0 --nshards 2 --output $E/tfeats_tvsum > $E/textract_s0.log 2>&1 &
nohup $PY scripts/v7_t_extract_tvsum.py --cards 5 --shard 1 --nshards 2 --output $E/tfeats_tvsum > $E/textract_s1.log 2>&1 &
sleep 1
echo launched
