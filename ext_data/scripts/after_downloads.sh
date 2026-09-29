#!/usr/bin/env bash
# Wait for a background download session to finish, then run the follow-up ingests + refresh.
#   bash scripts/run_bg.sh post_batch4 $AIC_EXT_ROOT/_logs/post_batch4.log bash scripts/after_downloads.sh batch4
#   bash scripts/run_bg.sh post_mrhisum $AIC_EXT_ROOT/_logs/post_mrhisum.log bash scripts/after_downloads.sh mrhisum
# Refresh only updates the working registry; frozen releases in $AIC_EXT_ROOT/_releases are untouched.
set -uo pipefail
source /home/supie/AIC/ext_data/scripts/env.sh
cd /home/supie/AIC/ext_data
PY=/opt/miniconda3/envs/cv/bin/python
wait_exit() {  # $1 = log file written by run_bg.sh; waits for an EXIT line newer than $2 (epoch s)
  until [ -f "$1" ] && tail -1 "$1" | grep -q "EXIT rc=" && [ "$(stat -c %Y "$1")" -ge "${2:-0}" ]; do sleep 300; done
  grep "EXIT rc=" "$1" | tail -1
}
case "${1:-}" in
  batch2)
    wait_exit "$AIC_EXT_ROOT/_logs/batch2_fetch.log"
    $AIC_EXT_NICE $PY scripts/ingest_davsod.py --allow-partial
    $AIC_EXT_NICE $PY scripts/ingest_clipshots.py --stage all --workers 4
    $AIC_EXT_NICE $PY scripts/ingest_phd2.py --sample 40 --download 0
    VALIDATE_DATASETS="DAVSOD ClipShots PHD2" bash scripts/refresh_all.sh
    ;;
  batch4)
    start=$(date +%s)
    wait_exit "$AIC_EXT_ROOT/_logs/batch4_fetch.log" "$start"
    $AIC_EXT_NICE $PY scripts/ingest_davsod.py --allow-partial
    $AIC_EXT_NICE $PY scripts/ingest_clipshots.py --stage all --workers 4
    VALIDATE_DATASETS="DAVSOD ClipShots" bash scripts/refresh_all.sh
    ;;
  mrhisum)
    wait_exit "$AIC_EXT_ROOT/MrHiSum/logs/yt8m.log"
    $AIC_EXT_NICE $PY scripts/ingest_mrhisum.py --stage index
    VALIDATE_DATASETS="MrHiSum" bash scripts/refresh_all.sh
    ;;
  *) echo "usage: $0 batch2|batch4|mrhisum" >&2; exit 2 ;;
esac
