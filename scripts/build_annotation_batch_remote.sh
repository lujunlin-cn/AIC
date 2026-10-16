#!/usr/bin/env bash
set -euo pipefail

# Run on the AutoDL host. The TSV is intentionally separate from the source
# records so this batch can be reproduced without a Python environment there.

ROOT="${1:-/root/autodl-tmp/annotation_batch_v1}"
DATA="/root/autodl-tmp/evac/data/aic/datasets/QVH_NATIVE_CANDIDATES/source"
TSV="${2:-/tmp/ANNOTATION_BATCH_V1.tsv}"

mkdir -p "$ROOT/videos" "$ROOT/frames" "$ROOT/meta"
test -s "$TSV"
test -x "$(command -v ffmpeg)"

while IFS=$'\t' read -r vid start end; do
  [ -n "$vid" ] || continue
  src="$(find "$DATA" -type f -name "$vid.mp4" -print -quit)"
  if [ -z "$src" ]; then
    echo "missing source: $vid" >&2
    exit 2
  fi
  out="$ROOT/videos/$vid.mp4"
  frame_dir="$ROOT/frames/$vid"
  mkdir -p "$frame_dir"
  echo "[$vid] $start-$end from $src"
  ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$start" -i "$src" -t 30 \
    -vf "scale='min(960,iw)':-2" -c:v libx264 -preset veryfast -crf 28 \
    -c:a aac -b:a 64k -ac 1 -ar 16000 "$out"
  rm -f "$frame_dir"/*.jpg
  ffmpeg -nostdin -hide_banner -loglevel error -y -ss "$start" -i "$src" -t 30 \
    -vf "fps=1,scale='min(960,iw)':-2" -frames:v 30 -q:v 3 \
    -start_number 0 "$frame_dir/%02d.jpg"
  n="$(find "$frame_dir" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
  [ "$n" -eq 30 ] || { echo "[$vid] expected 30 frames, got $n" >&2; exit 3; }
  ffprobe -v error -show_entries format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels \
    -of default=nw=1 "$out" > "$ROOT/meta/$vid.ffprobe.txt"
done < "$TSV"

find "$ROOT/videos" "$ROOT/frames" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/checksums.sha256"
du -sh "$ROOT"
echo "READY $ROOT"
