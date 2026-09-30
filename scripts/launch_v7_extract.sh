#!/bin/bash
# V7: re-run train smoke (NPU 3) + launch feature extraction 4 shards (NPU 2,4,5,6)
cd /root/AIC
set -a
. /data/aic/tools/ascend_teacher.env
set +a
E=/data/aic/experiments_910a/LFM_V7
PY=/data/aic/tools/lfm_venv/bin/python
mkdir -p "$E/feats"
nohup $PY scripts/v7_e0_train_smoke.py --cards 3 --steps 60 --output $E/e0/train_smoke.json > $E/e0/train_smoke.log 2>&1 &
i=0
for c in 2 4 5 6; do
  nohup $PY scripts/v7_s_extract_feats.py --cards $c --shard $i --nshards 4 --output-root $E/feats > $E/feats/extract_s$i.log 2>&1 &
  i=$((i+1))
done
sleep 1
echo launched
