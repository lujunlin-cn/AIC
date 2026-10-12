#!/bin/bash
# R15 910A full-evacuation driver (RUN ON kmsp05 / 910A, before ssh cutoff).
#
# Decisions locked 2026-10-12: RTX node direct push; PHD2 is the only
# external dataset; teacher = Qwen3.8-27B only (plus 4B base + TimeLens2-8B).
# Feature caches (LFM_V8 261G, lfm_feats 60G, samples_*) and the Ascend
# stack (ascend910a 394G, docker, venvs) are NOT moved - recomputable or
# useless on CUDA.
#
# Usage:
#   bash r15_evac_910a.sh --plan                     # list waves + sizes
#   bash r15_evac_910a.sh --wave1 --to USER@HOST:/data/evac   # ~230G core
#   bash r15_evac_910a.sh --wave2 --to USER@HOST:/data/evac   # ~399G PHD2
#   bash r15_evac_910a.sh --audit  --to USER@HOST:/data/evac   # post-check
# Target layout mirrors 910A: DIR/root/... and DIR/data/aic/... - unpack or
# rsync straight into / on the RTX node for path-identical pools.
set -u

TO=${TO:-}
PORT=${PORT:-22}
KEY=${KEY:-}   # optional IdentityFile; ssh config aliases also work when PORT is empty
STAGE=/data/evac_stage
EVAC_TAG=evac-$(date +%Y%m%d)
ACTION=""

SSH_OPTS="-o StrictHostKeyChecking=no ${KEY:+-i $KEY}"
# accept both `--to X` and env TO=X
while [ $# -gt 0 ]; do
  case "$1" in
    --plan|--wave1|--wave2|--audit) ACTION=${1#--} ;;
    --to) TO=$2; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done

# wave1: everything the RL/post-training line plus historical verdicts need
W1_REPO_TAR=$STAGE/AIC_repo_$EVAC_TAG.tgz
W1_DATA=(
  experiments_910a/LFM_V13
  experiments_910a/LFM_V11
  experiments_910a/LFM_V7
  experiments_910a/LFM_V9
  experiments_910a/LFM_V10
  experiments_910a/LFM450_EVAL_V1
  experiments_910a/PHD2_FRAG_CONFIRM_V1
  experiments_910a/PHD2_FRAG_V1/frames
  experiments              # old-line artifacts V5/V4/T5... verdicts+ckpts
  datasets
  semifinal_test
  semifinal_20261001
  official_test_20260926
  weights
  features
  pretrained/Qwen3.5-4B
  pretrained/TimeLens2-8B
  pretrained/Qwen3.8-27B
)
# wave1 extras: top-level files of PHD2_FRAG_V1 (pool labels jsonl/md) via
# files-from, since the big subdirs (lfm_feats/samples_*) are NOT moved
W1_FRAG_TOP=$(find /data/aic/experiments_910a/PHD2_FRAG_V1 -maxdepth 1 -type f)

plan() {
  echo "== wave1 (core, ~230G) =="
  echo "  repo tar: /root/AIC (all branches incl. evac-910a-20261012, r8-fullinfo, teacher-v4 ahead94)"
  for d in "${W1_DATA[@]}"; do printf '  /data/aic/%s\n' "$d"; done
  echo "  + PHD2_FRAG_V1 top-level files ($(echo "$W1_FRAG_TOP" | wc -l) files)"
  echo "== wave2 (external, ~399G) =="
  echo "  /data/aic/external_datasets/PHD2"
  echo
  echo "== measured sizes =="
  du -sh /root/AIC "$STAGE" 2>/dev/null | head -2
  (for d in "${W1_DATA[@]}"; do du -sh "/data/aic/$d" 2>/dev/null; done) | sort -rh
  du -sh /data/aic/external_datasets/PHD2 2>/dev/null
}

rsync_to() {  # rsync_to <src-abs-path> <rel-dest-under-TO-parent>
  local src=$1 reldir=$2
  ssh -p "$PORT" $SSH_OPTS "$TO_HOST" \
    "mkdir -p '$TO_DIR/$reldir'" || return 1
  rsync -a --info=progress2 --partial -e "ssh -p $PORT $SSH_OPTS" \
    "$src" "$TO_HOST:$TO_DIR/$reldir/"
}

parse_to() {  # split USER@HOST:/dir into TO_HOST / TO_DIR
  case "$TO" in
    *:*) TO_HOST=${TO%%:*}; TO_DIR=${TO#*:} ;;
    *) echo "need --to USER@HOST:/dir"; exit 1 ;;
  esac
}

repo_tar() {
  mkdir -p "$STAGE"
  if [ ! -s "$W1_REPO_TAR" ]; then
    echo "packing /root/AIC -> $W1_REPO_TAR"
    tar czf "$W1_REPO_TAR" -C /root AIC
  fi
}

frag_top_files() {
  echo "shipping PHD2_FRAG_V1 top-level label files"
  printf '%s\n' "$W1_FRAG_TOP" | sed 's#^/data/aic/##' \
    > "$STAGE/frag_top_files.txt"
  rsync -a --info=progress2 --partial \
    --files-from="$STAGE/frag_top_files.txt" \
    -e "ssh -p $PORT $SSH_OPTS" \
    /data/aic/ "$TO_HOST:$TO_DIR/data/aic/"
}

wave1() {
  parse_to; repo_tar
  ssh -p "$PORT" $SSH_OPTS "$TO_HOST" "mkdir -p '$TO_DIR/root'"
  rsync_to "$W1_REPO_TAR" root/
  for d in "${W1_DATA[@]}"; do
    parent=$(dirname "$d")   # "weights" -> ".", "pretrained/X" -> "pretrained"
    ssh -p "$PORT" $SSH_OPTS "$TO_HOST" \
      "mkdir -p '$TO_DIR/data/aic/$parent'"
    rsync -a --info=progress2 --partial \
      -e "ssh -p $PORT $SSH_OPTS" \
      "/data/aic/$d" "$TO_HOST:$TO_DIR/data/aic/$parent/"
  done
  frag_top_files
  echo "WAVE1_DONE. Re-run later to resume incremental (rsync --partial)."
}

wave2() {
  parse_to
  ssh -p "$PORT" $SSH_OPTS "$TO_HOST" \
    "mkdir -p '$TO_DIR/data/aic/external_datasets'"
  rsync -a --info=progress2 --partial \
    -e "ssh -p $PORT $SSH_OPTS" \
    /data/aic/external_datasets/PHD2 \
    "$TO_HOST:$TO_DIR/data/aic/external_datasets/"
  echo "WAVE2_DONE (PHD2)."
}

audit() {
  parse_to
  echo "== local (910A) =="
  du -sh /root/AIC 2>/dev/null
  (for d in "${W1_DATA[@]}"; do du -sh "/data/aic/$d" 2>/dev/null; done
   du -sh /data/aic/external_datasets/PHD2 2>/dev/null) | sort -rh
  echo "== remote ($TO_HOST) =="
  ssh -p "$PORT" $SSH_OPTS "$TO_HOST" \
    "du -sh $TO_DIR/root $TO_DIR/data/aic/* 2>/dev/null | sort -rh" | head -30
  echo "compare line-by-line; any missing dir -> re-run its wave (incremental)."
}

case "$ACTION" in
  plan) plan ;;
  wave1) wave1 ;;
  wave2) wave2 ;;
  audit) audit ;;
  *) grep '^#' "$0" | head -20; exit 1 ;;
esac
