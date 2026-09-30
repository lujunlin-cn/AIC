#!/bin/bash
# Local watchdog v3 (V8).
#
# Two earlier bugs made this actively harmful, so read before reusing:
#  1. It re-entered v8_resume.sh whenever `pgrep -fc v8_resume.sh` was 0, but
#     v8_resume.sh execs launch_v8_train.sh, so the process name changed and the
#     check always read 0 -> a second (then third) queue stacked on a live one,
#     multiplying peak RAM until the node OOMed and rebooted.
#  2. Empty probe fields (failed ssh) compared unequal to "true" and triggered
#     spurious docker starts.
#
# v3: probe the TRAINING PROCESSES (stable name), never the wrapper; only act
# when the container is genuinely down or no training is running. Pure
# observation otherwise.
HOST=221.213.81.199; PORT=44000
for i in $(seq 1 40); do
  if timeout 10 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
    ST=$(ssh -p $PORT -o ConnectTimeout=15 root@$HOST \
      'docker inspect aic-batch --format "{{.State.Running}}"; docker exec aic-batch pgrep -c -f v8_s_train 2>/dev/null | head -1; ls /data/aic/experiments_910a/LFM_V8/arms/*/summary.json 2>/dev/null | wc -l' 2>/dev/null)
    RUN=$(printf '%s\n' "$ST" | sed -n 1p)
    NPROC=$(printf '%s\n' "$ST" | sed -n 2p)
    DONE=$(printf '%s\n' "$ST" | sed -n 3p)
    case "$RUN$NPROC$DONE" in *true*) ;; *) NPROC=0; DONE=0 ;; esac
    echo "$(date +%H:%M:%S) run=$RUN train_procs=${NPROC:-?} arms_done=${DONE:-?}"
    if [ "${DONE:-0}" = "12" ]; then echo ALL_ARMS_DONE; exit 0; fi
    if [ "$RUN" = "true" ]; then
      if [ "${NPROC:-0}" -lt 1 ] 2>/dev/null; then
        echo "$(date +%H:%M:%S) no training running - re-entering queue"
        ssh -p $PORT -o ConnectTimeout=15 root@$HOST 'docker exec aic-batch bash -c "cd /root/AIC; nohup bash /root/AIC/scripts/v8_resume.sh >> /data/aic/experiments_910a/LFM_V8/resume.log 2>&1 &"' 2>/dev/null
      fi
    else
      echo "$(date +%H:%M:%S) container down - restarting"
      ssh -p $PORT -o ConnectTimeout=15 root@$HOST 'docker start aic-batch >/dev/null 2>&1; sleep 10; docker exec aic-batch bash -c "cd /root/AIC; nohup bash /root/AIC/scripts/v8_resume.sh >> /data/aic/experiments_910a/LFM_V8/resume.log 2>&1 &"' 2>/dev/null
    fi
  else
    echo "$(date +%H:%M:%S) port closed"
  fi
  sleep 180
done
echo WATCHDOG_TIMEOUT