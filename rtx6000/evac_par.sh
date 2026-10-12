#!/bin/bash
# R15 wave1 parallel driver on 910A (single-stream was ~8.5MB/s; fan out to
# lift aggregate throughput). Read-only on 910A sources; rsync --partial
# keeps every prior byte, killing/restarting is always safe.
set -u
export PORT=29735 KEY=/root/.ssh/evac_autodl
T=root@connect.westd.seetacloud.com
D=/root/autodl-tmp/evac
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no"

$SSH $T "mkdir -p $D/root $D/data/aic/experiments_910a $D/data/aic/pretrained $D/data/aic/external_datasets $D/data/aic/experiments"

rs() { # rs <tag> <src-abs> <dest-parent>
  rsync -a --info=progress2 --partial -e "$SSH" "$2" "$T:$3/" >> "/root/evac_$1.log" 2>&1
  echo "DONE $1 $(date +%T)" >> /root/evac_par.done
}

rs repo    "$(ls /data/evac_stage/AIC_repo_*.tgz 2>/dev/null | head -1)"      "$D/root/" &
rs v13     /data/aic/experiments_910a/LFM_V13                                 "$D/data/aic/experiments_910a/" &
rs v7      /data/aic/experiments_910a/LFM_V7                                  "$D/data/aic/experiments_910a/" &
rs v9      /data/aic/experiments_910a/LFM_V9                                  "$D/data/aic/experiments_910a/" &
rs v10     /data/aic/experiments_910a/LFM_V10                                 "$D/data/aic/experiments_910a/" &
rs v11     /data/aic/experiments_910a/LFM_V11                                 "$D/data/aic/experiments_910a/" &
rs oldexp  /data/aic/experiments                                              "$D/data/aic/" &
rs qvh10   /data/aic/experiments_910a/QVH_V10                                 "$D/data/aic/experiments_910a/" &
rs mdl4b   /data/aic/pretrained/Qwen3.5-4B                                    "$D/data/aic/pretrained/" &
rs mdltl   /data/aic/pretrained/TimeLens2-8B                                  "$D/data/aic/pretrained/" &
rs mdl27   /data/aic/pretrained/Qwen3.8-27B                                   "$D/data/aic/pretrained/" &
rs dsets   /data/aic/datasets                                                 "$D/data/aic/" &
rs smft    /data/aic/semifinal_test                                           "$D/data/aic/" &
rs smf2    /data/aic/semifinal_20261001                                       "$D/data/aic/" &
rs ofcl    /data/aic/official_test_20260926                                   "$D/data/aic/" &
rs wts     /data/aic/weights                                                  "$D/data/aic/" &
rs feats   /data/aic/features                                                 "$D/data/aic/" &
rs frames  /data/aic/experiments_910a/PHD2_FRAG_V1/frames                     "$D/data/aic/experiments_910a/PHD2_FRAG_V1/" &
rs conf    /data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1                    "$D/data/aic/experiments_910a/" &
rs lfm450  /data/aic/experiments_910a/LFM450_EVAL_V1                          "$D/data/aic/experiments_910a/" &

# PHD2_FRAG_V1 top-level label files (pool jsonl etc., 23 files)
find /data/aic/experiments_910a/PHD2_FRAG_V1 -maxdepth 1 -type f | sed 's#^/data/aic/##' > /data/evac_stage/frag_top_files.txt
rsync -a --partial -e "$SSH" --files-from=/data/evac_stage/frag_top_files.txt /data/aic/ "$T:$D/data/aic/" >> /root/evac_fragtop.log 2>&1
echo "DONE fragtop $(date +%T)" >> /root/evac_par.done

wait
echo "WAVE1_PAR_ALL_DONE $(date +%T)" >> /root/evac_par.done
