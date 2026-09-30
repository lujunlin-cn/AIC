#!/bin/bash
# V7: TVSum 1fps extraction 2 shards (NPU 4,5)
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
nohup $PY scripts/v7_t_extract_tvsum.py --cards 4 --shard 0 --nshards 2 --output $E/tfeats_tvsum > $E/textract_s0.log 2>&1 &
nohup $PY scripts/v7_t_extract_tvsum.py --cards 5 --shard 1 --nshards 2 --output $E/tfeats_tvsum > $E/textract_s1.log 2>&1 &
sleep 1
echo launched
