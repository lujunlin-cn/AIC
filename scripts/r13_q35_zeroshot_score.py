#!/usr/bin/env python3
"""R13 T3: Qwen3.5-4B zero-shot frame scoring over the 426 semifinal videos.

Sharded by video_id mod N_SHARDS; one worker per NPU card. Per video:
uniformly sample N frames (N = clip(duration, 16, 64)), one generate call,
parse a JSON array of per-frame scores, save <vid>.json for the packager.
Resumable: skips videos whose score file already exists.

Worker usage: r13_q35_zeroshot_score.py --shard 0 --n-shards 8
"""
import argparse
import json
import os
import re
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys  # noqa: E402

sys.path.insert(0, "/data/aic/scripts")
import r13_qwen35_safe_chunk  # noqa: E402

r13_qwen35_safe_chunk.apply()

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import cv2  # noqa: E402

torch.npu.set_compile_mode(jit_compile=False)

INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
OUT_DIR = "/data/aic/experiments_910a/LFM_V13/q35_zs_scores"

MODEL = "/data/aic/pretrained/Qwen3.5-4B"

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--n-shards", type=int, default=8)
args = ap.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
vids = []
for line in open(INTAKE):
    r = json.loads(line)
    vids.append((r["video_id"], r["video_path"], float(r["duration"])))
mine = [v for v in vids if int(v[0]) % args.n_shards == args.shard]

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(f"npu:0")
model.eval()
print(f"[shard {args.shard}] {len(mine)} videos, loaded on npu:0", flush=True)

t_start = time.time()
done = 0
for vid, path, dur in mine:
    out_path = os.path.join(OUT_DIR, f"{vid}.json")
    if os.path.exists(out_path):
        done += 1
        continue
    try:
        n_frames = int(max(16, min(64, round(dur))))
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            print(f"[shard {args.shard}] {vid} no frames, skip", flush=True)
            continue
        idxs = [int(round(i * (total - 1) / max(n_frames - 1, 1))) for i in range(n_frames)]
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < 4:
            print(f"[shard {args.shard}] {vid} only {len(frames)} frames, skip", flush=True)
            continue

        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": (
                f"These are {len(frames)} frames sampled in order from one video. "
                "Score each frame 0.00-1.00 for how likely it appears in a "
                "user-made highlight GIF of this video. Reply with ONLY a JSON "
                "array of " + str(len(frames)) + " numbers, nothing else.")},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        inputs = proc(text=[text], videos=[frames], return_tensors="pt").to("npu:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        raw = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        m = re.search(r"\[[\s\S]*?\]", raw)
        scores = json.loads(m.group(0)) if m else None
        if not isinstance(scores, list) or len(scores) != len(frames):
            # keep raw for manual recovery, save nulls to keep the packager simple
            scores = None
        with open(out_path, "w") as f:
            json.dump({"video_id": vid, "frame_indices": idxs,
                       "n_frames": len(frames), "scores": scores,
                       "raw": raw[:2000]}, f, ensure_ascii=False)
        done += 1
        if done % 10 == 0:
            el = time.time() - t_start
            print(f"[shard {args.shard}] {done}/{len(mine)} elapsed={el/60:.1f}m "
                  f"({el/done:.1f}s/video)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[shard {args.shard}] {vid} ERROR {type(e).__name__}: {str(e)[:200]}",
              flush=True)

print(f"[shard {args.shard}] ALL_DONE {done}/{len(mine)}", flush=True)
