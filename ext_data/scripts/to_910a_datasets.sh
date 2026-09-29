#!/bin/bash
# 把远程A上已做好的数据集推送到 910A:/data/aic（tmux to_910a 运行）
# 排除：PHD2/raw（仍在下载）、*.part/*.tmp（半成品）、_logs/_tmp
# 传完自动对账（文件数+字节），结果写 _logs/to_910a_datasets.log
set -u
KEY="$HOME/.ssh/id_aic_transfer"
SSHBASE="ssh -i $KEY -p 44000 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
LOG=/data/aic/external_datasets/_logs/to_910a_datasets.log
mkdir -p "$(dirname "$LOG")"

run() {
  local name=$1; shift
  for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "[$(date -Is)] rsync $name pass $i" >> "$LOG"
    if "$@" >> "$LOG" 2>&1; then
      echo "[$(date -Is)] rsync $name CLEAN" >> "$LOG"
      return 0
    fi
    echo "[$(date -Is)] rsync $name pass $i nonzero, retry in 60s" >> "$LOG"
    sleep 60
  done
  echo "[$(date -Is)] rsync $name GAVE_UP after 10 passes" >> "$LOG"
  return 1
}

remote() { ssh -i "$KEY" -p 44000 -o BatchMode=yes root@221.213.81.199 "$@"; }

echo "[$(date -Is)] START to_910a_datasets" >> "$LOG"
ok=1

run external_datasets rsync -a --partial --append-verify \
    --exclude /PHD2/raw/ --exclude /_logs/ --exclude /_tmp/ \
    --exclude "*.part" --exclude "*.tmp" \
    -e "$SSHBASE" \
    /data/aic/external_datasets/ root@221.213.81.199:/data/aic/external_datasets/ || ok=0

run aic_datasets rsync -a --partial --append-verify -e "$SSHBASE" \
    /data/aic/datasets/ root@221.213.81.199:/data/aic/datasets/ || ok=0

echo "[$(date -Is)] VERIFY begin" >> "$LOG"
SRC=/data/aic/external_datasets
for d in YouTubeHighlights DHF1K DAVSOD LIVE_YT_VC MrHiSum LaSOT GAICD SA_V ClipShots RetargetVid _releases _registry _tools; do
  s_files=$(find "$SRC/$d" -type f ! -name "*.part" ! -name "*.tmp" | wc -l)
  d_files=$(remote "find /data/aic/external_datasets/$d -type f 2>/dev/null | wc -l")
  s_bytes=$(du -sb --exclude="*.part" --exclude="*.tmp" "$SRC/$d" | cut -f1)
  d_bytes=$(remote "du -sb /data/aic/external_datasets/$d 2>/dev/null | cut -f1")
  if [ "$s_files" = "$d_files" ] && [ "$s_bytes" = "$d_bytes" ]; then st=OK; else st=MISMATCH; ok=0; fi
  echo "VERIFY $d files=$s_files/$d_files bytes=$s_bytes/$d_bytes $st" >> "$LOG"
done

# PHD2：源端排除仍在增长的 raw/ 后对账
s_files=$(find "$SRC/PHD2" -type f ! -path "*/raw/*" ! -name "*.part" ! -name "*.tmp" | wc -l)
s_bytes=$(find "$SRC/PHD2" -mindepth 1 -maxdepth 1 ! -name raw -exec du -sb {} + | awk '{s+=$1} END {print s+0}')
d_files=$(remote "find /data/aic/external_datasets/PHD2 -type f 2>/dev/null | wc -l")
d_bytes=$(remote "du -sb /data/aic/external_datasets/PHD2 2>/dev/null | cut -f1")
if [ "$s_files" = "$d_files" ] && [ "$s_bytes" = "$d_bytes" ]; then st=OK; else st=MISMATCH; ok=0; fi
echo "VERIFY PHD2(excl raw) files=$s_files/$d_files bytes=$s_bytes/$d_bytes $st" >> "$LOG"

for d in /data/aic/datasets/*; do
  n=$(basename "$d")
  s_files=$(find "$d" -type f | wc -l)
  d_files=$(remote "find /data/aic/datasets/$n -type f 2>/dev/null | wc -l")
  s_bytes=$(du -sb "$d" | cut -f1)
  d_bytes=$(remote "du -sb /data/aic/datasets/$n 2>/dev/null | cut -f1")
  if [ "$s_files" = "$d_files" ] && [ "$s_bytes" = "$d_bytes" ]; then st=OK; else st=MISMATCH; ok=0; fi
  echo "VERIFY datasets/$n files=$s_files/$d_files bytes=$s_bytes/$d_bytes $st" >> "$LOG"
done

if [ "$ok" = 1 ]; then
  echo "[$(date -Is)] ALL DONE OK" >> "$LOG"
else
  echo "[$(date -Is)] DONE WITH ERRORS" >> "$LOG"
fi
