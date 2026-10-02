#!/bin/bash
# Chain that fires when the PHD2 teacher run finishes.  AMENDED 2026-10-02:
# the PHD2 KD stream is a CONFIRMED negative transfer on the official drop
# (the A1_raw KD head packaged as VISIONONLY scored 34.61 raw vs the no-KD B3
# head's 38.76 on the identical 234,159-prediction set), so the downstream KD
# pool rebuild + five-arm retrain are REMOVED - running them again would only
# reconfirm a rejected direction while holding the NPU the QVH temporal work
# needs.  What remains: quarantine bad points so the 5,795-clip teacher point
# set stays a clean reference asset, then stop.  The points are retained for
# any future non-KD use (e.g. as weak spatial prior, or diagnosis only).
#
# Usage: v9_pipeline_after_teacher.sh [--wait]
set -u
P=/data/aic/experiments_910a/PHD2_FRAG_V1
EXPECT=${EXPECT:-5795}
LOG=$P/pipeline.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ "${1:-}" = "--wait" ]; then
  say "waiting for $EXPECT teacher point files (currently $(ls $P/teacher/points | wc -l))"
  while [ "$(ls $P/teacher/points 2>/dev/null | wc -l)" -lt "$EXPECT" ]; do
    sleep 600
  done
  say "teacher complete: $(ls $P/teacher/points | wc -l) point files"
fi

say "quarantining any ratio-mismatched points (keeps the point set clean for later use)"
docker exec aic-batch python /root/AIC/scripts/v9_prune_mismatched_points.py | tee -a "$LOG"

say "SKIPPED: KD pool rebuild + five-arm retrain - PHD2 KD is a confirmed -4.15-raw"
say "negative transfer on the official drop (VISIONONLY/A1 head 34.61 vs B3 38.76)."
say "NPU freed for the QVHighlights temporal-axis work instead.  Pipeline ends here."
