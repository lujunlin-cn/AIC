#!/bin/bash
# V8 resume entry point - SINGLE INSTANCE enforced by flock.
#
# Why flock: an earlier watchdog re-entered this script every few minutes
# because it probed the wrapper name, which `exec` had already replaced with
# launch_v8_train.sh. Each re-entry stacked another queue on a live one; with
# the npz loader bug (per-row re-inflation, ~100 GB/arm) that exhausted host RAM
# and the node rebooted. The lock makes duplicate entry impossible regardless of
# how the caller behaves; callers can keep retrying safely.
set -u
LOCK=/tmp/v8_resume.lock
PY=/data/aic/tools/lfm_venv/bin/python
S=/data/aic/experiments_910a/LFM_V8/samples
LOG=/data/aic/experiments_910a/LFM_V8
cd /root/AIC

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[resume] another instance holds the lock; exiting" >> $LOG/resume.log
  exit 0
fi
echo "[resume] acquired lock $(date)" >> $LOG/resume.log

# rebuild any missing sample cache (npy sidecars preferred; the script writes
# memmap files when absent and skips existing ones)
NEED=""
for t in rv_train rv_dev rv_diag live_train live_dev live_confirmation; do
  [ -f $S/${t}_feat.npy ] || NEED="$NEED $t"
done
if [ -n "$NEED" ]; then
  echo "[resume] building sample caches:$NEED $(date)" >> $LOG/resume.log
  $PY /root/AIC/scripts/v8_build_samples.py --tags $NEED >> $LOG/build_samples.log 2>&1
fi

echo "[resume] entering arm queue $(date)" >> $LOG/chain.log
bash /root/AIC/scripts/launch_v8_train.sh
echo "[resume] queue returned $(date)" >> $LOG/chain.log