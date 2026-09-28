#!/usr/bin/env bash
# Run a command in its own detached tmux session with the ext_data env.
#   bash run_bg.sh <session-name> <log-file> <command...>
# The session stays open after the command exits so the exit code is visible.
set -euo pipefail
name=$1; logf=$2; shift 2
mkdir -p "$(dirname "$logf")"
cmd=$(printf '%q ' "$@")
if tmux has-session -t "$name" 2>/dev/null; then
  echo "session $name already running" >&2; exit 1
fi
tmux new-session -d -s "$name" "source /home/supie/AIC/ext_data/scripts/env.sh; cd /home/supie/AIC/ext_data; \
  echo \"[\$(date -Is)] START $cmd\" >> $logf; \$AIC_EXT_NICE $cmd >> $logf 2>&1; rc=\$?; \
  echo \"[\$(date -Is)] EXIT rc=\$rc\" >> $logf; sleep 600"
echo "started $name -> $logf"
