#!/usr/bin/env bash
# Re-derive splits, merged registry and validation after any ingest step.
# Ingest scripts rewrite processed/media.jsonl (aic_split reset), so always run
# this afterwards.  Safe to re-run; raw data is never touched.
set -euo pipefail
source /home/supie/AIC/ext_data/scripts/env.sh
cd /home/supie/AIC/ext_data
$AIC_EXT_PY scripts/build_splits_and_ledger.py
$AIC_EXT_PY scripts/build_registry.py
if [ "${1:-}" != "--no-validate" ]; then
  $AIC_EXT_PY scripts/validate_datasets.py ${VALIDATE_DATASETS:+--datasets $VALIDATE_DATASETS}
fi
$AIC_EXT_PY scripts/make_report.py --out "$AIC_EXT_ROOT/_registry/registry_status.md"
