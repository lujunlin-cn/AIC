#!/usr/bin/env bash
# LaSOT full download, stage 1: fetch+extract every category zip, deleting the
# archive right after a successful extract so ~248 GB of archives never coexist.
# Indexing is one pass over all categories afterwards (ingest_lasot.py rewrites
# processed/*.jsonl from the categories it is given, so it must run once over
# the full set, not per category).
#   bash scripts/run_bg.sh lasot_full $AIC_EXT_ROOT/_logs/lasot_full.log bash scripts/lasot_full.sh
set -uo pipefail
source /home/supie/AIC/ext_data/scripts/env.sh
cd /home/supie/AIC/ext_data
PY=/opt/miniconda3/envs/cv/bin/python
ROOT=${AIC_EXT_ROOT}/LaSOT

cats=$(cat "$ROOT/annotations/training_set.txt" "$ROOT/annotations/testing_set.txt" \
       | tr ' ' '\n' | sed 's/-[0-9]*$//' | sort -u)

for cat in $cats; do
  if [ -f "$ROOT/raw/$cat/.extracted" ]; then
    echo "[$(date -Is)] SKIP $cat (already extracted)"
    rm -f "$ROOT/raw/archives/$cat.zip"   # keep disk free even if a previous run left it
    continue
  fi
  echo "[$(date -Is)] FETCH+EXTRACT $cat"
  # Only fetch+extract; do NOT index here (per-category indexing rewrites the
  # shared media.jsonl).  fetch+extract is the reusable part of ingest_lasot.
  if $PY scripts/ingest_lasot.py --categories "$cat" --fetch-only 2>>"$ROOT/../_logs/lasot_full_err.log"; then
    if [ -f "$ROOT/raw/$cat/.extracted" ]; then
      rm -f "$ROOT/raw/archives/$cat.zip" "$ROOT/raw/archives/$cat.zip.part"
      echo "[$(date -Is)] DONE $cat (zip removed)"
    else
      echo "[$(date -Is)] FETCHED $cat (extract marker missing; zip kept)"
    fi
  else
    echo "[$(date -Is)] FAIL $cat (zip/.part kept for resume)"
  fi
  sleep 2
done
echo "[$(date -Is)] ALL_FETCHED"
