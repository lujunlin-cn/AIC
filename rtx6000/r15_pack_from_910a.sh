#!/bin/bash
# R15 data/model bundle builder. RUN ON THE 910A BOX, not on the RTX node.
# Produces one tarball whose paths unpack to the SAME /data/aic layout the
# pool jsonl files carry in their frames_dir fields -- extract with:
#   tar xzf rtx6000_bundle_<date>.tar.gz -C /      # on the RTX node
#   cd /data && md5sum -c rtx6000_bundle_<date>.md5
#
# LoRA: the adapter to use as RL init is the user's call; set LORA_DIRS to
# a space-separated list of ckpt dirs under $LFM (default: ckpt_060, the
# R13-decided RL init candidate).
set -euo pipefail

PRE=/data/aic/pretrained
LFM=/data/aic/experiments_910a/LFM_V13
FRAG_FRAMES=/data/aic/experiments_910a/PHD2_FRAG_V1/frames
OUT=${OUT:-/data/aic/rtx6000_bundles}
LORA_DIRS=${LORA_DIRS:-"$LFM/q35_sft_out/ckpt_060"}
STAMP=$(date +%Y%m%d)
BUNDLE=$OUT/rtx6000_bundle_$STAMP

mkdir -p "$BUNDLE"
LIST=$BUNDLE/bundle_filelist.txt

add() {  # add <path>  (records relpath under /, i.e. "data/aic/...")
  local p=$1
  test -e "$p" || { echo "MISSING: $p" | tee -a "$BUNDLE/WARNINGS.txt"; return 1; }
  echo "${p#/}" >> "$LIST"
}

: > "$LIST"

echo "== base model =="
add $PRE/Qwen3.5-4B/config.json
find $PRE/Qwen3.5-4B -maxdepth 1 -name '*.safetensors' | while read -r f; do add "$f"; done
add $PRE/Qwen3.5-4B/tokenizer_config.json

echo "== pools / labels =="
for f in grpo_pool_v2.jsonl frag_labels_posfirst.jsonl holdout_grpo.jsonl; do
  test -s $LFM/$f && add $LFM/$f || echo "note: $LFM/$f absent (ok if superseded)" | tee -a "$BUNDLE/WARNINGS.txt"
done

echo "== LoRA inits (user picks) =="
for d in $LORA_DIRS; do
  test -s "$d/adapter_config.json" || { echo "MISSING adapter: $d"; exit 1; }
  find "$d" -maxdepth 1 -type f | while read -r f; do add "$f"; done
done
echo "packed LoRA dirs: $LORA_DIRS" > "$BUNDLE/LORA_MANIFEST.txt"

echo "== frames referenced by grpo_pool_v2.jsonl (filtered, not the full tree) =="
python3 - "$LFM/grpo_pool_v2.jsonl" >> "$LIST" <<'EOF'
import json, os, sys
vids = sorted({json.loads(l)["vid"] for l in open(sys.argv[1])})
n = 0
for v in vids:
    d = f"data/aic/experiments_910a/PHD2_FRAG_V1/frames/{v}"
    if not os.path.isdir("/" + d):
        print(f"WARN frames dir missing: {d}", file=sys.stderr)
        continue
    print(d)
    n += 1
print(f"frame dirs: {n}/{len(vids)}", file=sys.stderr)
EOF

echo "== mining artifacts (if present) =="
test -d $LFM/grpo_mining && find $LFM/grpo_mining -type f | while read -r f; do add "$f"; done || true

echo "== tarball =="
# entries are "data/aic/..." relative to / -> extract with -C / to rebuild
# the exact /data/aic layout
cd /
tar czf "$BUNDLE.tar.gz" -T "$LIST" 2> "$BUNDLE/tar_warnings.txt" || true
md5sum $(cat "$LIST") > "$BUNDLE.md5"
ls -lh "$BUNDLE.tar.gz"
echo "DONE: $BUNDLE.tar.gz (+ .md5). Transfer, then on the RTX node:"
echo "  tar xzf $(basename $BUNDLE).tar.gz -C /"
echo "  cd / && md5sum -c $(basename $BUNDLE).md5 | grep -v ': OK' || true"
