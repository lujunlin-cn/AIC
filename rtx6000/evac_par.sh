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
# rsync idempotent (--partial keeps prior bytes): re-running everything is
# cheap for finished streams (incremental skip). Two staggered batches of
# ~10 -- 19 concurrent ssh sessions once tripped AutoDL-side 255 cutoffs.
B1="mdl27 qvh10 oldexp v13 v11 mdltl v7 v9 v10 repo"
B2="dsets smft smf2 ofcl wts feats frames conf lfm450"

run_batch() { # run_batch <B1|B2>
  for t in $1; do
    case $t in
      mdl27)  rs "$t" /data/aic/pretrained/Qwen3.8-27B "$D/data/aic/pretrained/" & ;;
      qvh10)  rs "$t" /data/aic/experiments_910a/QVH_V10 "$D/data/aic/experiments_910a/" & ;;
      oldexp) rs "$t" /data/aic/experiments "$D/data/aic/" & ;;
      v13)    rs "$t" /data/aic/experiments_910a/LFM_V13 "$D/data/aic/experiments_910a/" & ;;
      v11)    rs "$t" /data/aic/experiments_910a/LFM_V11 "$D/data/aic/experiments_910a/" & ;;
      mdltl)  rs "$t" /data/aic/pretrained/TimeLens2-8B "$D/data/aic/pretrained/" & ;;
      v7)     rs "$t" /data/aic/experiments_910a/LFM_V7 "$D/data/aic/experiments_910a/" & ;;
      v9)     rs "$t" /data/aic/experiments_910a/LFM_V9 "$D/data/aic/experiments_910a/" & ;;
      v10)    rs "$t" /data/aic/experiments_910a/LFM_V10 "$D/data/aic/experiments_910a/" & ;;
      repo)   rs "$t" "$(ls /data/evac_stage/AIC_repo_*.tgz | head -1)" "$D/root/" & ;;
      dsets)  rs "$t" /data/aic/datasets "$D/data/aic/" & ;;
      smft)   rs "$t" /data/aic/semifinal_test "$D/data/aic/" & ;;
      smf2)   rs "$t" /data/aic/semifinal_20261001 "$D/data/aic/" & ;;
      ofcl)   rs "$t" /data/aic/official_test_20260926 "$D/data/aic/" & ;;
      wts)    rs "$t" /data/aic/weights "$D/data/aic/" & ;;
      feats)  rs "$t" /data/aic/features "$D/data/aic/" & ;;
      frames) rs "$t" /data/aic/experiments_910a/PHD2_FRAG_V1/frames "$D/data/aic/experiments_910a/PHD2_FRAG_V1/" & ;;
      conf)   rs "$t" /data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1 "$D/data/aic/experiments_910a/" & ;;
      lfm450) rs "$t" /data/aic/experiments_910a/LFM450_EVAL_V1 "$D/data/aic/experiments_910a/" & ;;
    esac
    sleep 3   # stagger: avoid concurrent ssh handshake storms
  done
  wait
}

run_batch "$B1"
# PHD2_FRAG_V1 top-level label files (pool jsonl etc., 23 files)
find /data/aic/experiments_910a/PHD2_FRAG_V1 -maxdepth 1 -type f | sed 's#^/data/aic/##' > /data/evac_stage/frag_top_files.txt
rsync -a --partial -e "$SSH" --files-from=/data/evac_stage/frag_top_files.txt /data/aic/ "$T:$D/data/aic/" >> /root/evac_fragtop.log 2>&1
echo "DONE fragtop $(date +%T)" >> /root/evac_par.done

run_batch "$B2"
echo "WAVE1_PAR_ALL_DONE $(date +%T)" >> /root/evac_par.done
