#!/bin/bash
set -u
for s in 2 3 4 5 6 7 8; do
  a=$((s*50)); b=$(((s+1)*50)); if [ "$s" -eq 8 ]; then b=426; fi
  out=/root/autodl-tmp/r47_tl_shard${s}.json
  log=/root/autodl-tmp/r47_tl_shard${s}.log
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=2m 2h /root/autodl-tmp/aha_env/bin/python /root/autodl-tmp/r47_timelens2_official.py --model /root/autodl-tmp/evac/data/aic/pretrained/TimeLens2-8B --video-root /root/autodl-tmp/aic_video --output "$out" --start "$a" --end "$b" >"$log" 2>&1
  rc=$?
  echo "shard=$s rc=$rc" >> /root/autodl-tmp/r47_tl_shards.status
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
