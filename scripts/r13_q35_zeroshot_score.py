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


def parse_scores(raw, n):
    """JSON array anywhere in raw; else last-n floats (auto /100 if 0-100 scale)."""
    import re as _re

    best = None
    for mch in _re.finditer(r"\[[^\[\]]*\]", raw):
        try:
            arr = json.loads(mch.group(0))
            if (isinstance(arr, list) and len(arr) == n
                    and all(isinstance(x, (int, float)) for x in arr)):
                best = arr
                break
        except Exception:
            pass
    if best is not None:
        return best
    fl = _re.findall(r"-?\d+\.\d+|-?\d+", raw)
    if len(fl) >= n:
        vals = [float(x) for x in fl[-n:]]
        mx = max(vals) if vals else 1.0
        if mx > 1.5:
            vals = [v / 100.0 for v in vals]
        return vals
    return None

MODEL = "/data/aic/pretrained/Qwen3.5-4B"

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--n-shards", type=int, default=8)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--out-dir", default=OUT_DIR)
ap.add_argument("--device-map", default=None,
                help="e.g. 'auto' with --max-memory for cross-card sharding")
ap.add_argument("--max-memory", default=None,
                help="comma list, e.g. '28GiB,28GiB'")
ap.add_argument("--device", default="npu:0")
ap.add_argument("--max-frames", type=int, default=64)
ap.add_argument("--no-think", action="store_true")
args = ap.parse_args()

_dev_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
if not args.device_map:
    torch.npu.set_device(_dev_idx)
    assert torch.npu.device_count() > 0, "no NPU visible"
    print(f"[shard {args.shard}] using npu:{_dev_idx} "
          f"(visible={torch.npu.device_count()})", flush=True)

os.makedirs(args.out_dir, exist_ok=True)
vids = []
for line in open(INTAKE):
    r = json.loads(line)
    vids.append((r["video_id"], r["video_path"], float(r["duration"])))
mine = [v for v in vids if int(v[0]) % args.n_shards == args.shard]

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(args.model, use_fast=False)
if args.device_map:
    mm_lim = {}
    for i, m in enumerate(args.max_memory.split(",")):
        mm_lim[i] = m
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager",
        device_map=args.device_map, max_memory=mm_lim)
else:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager").to(args.device)
model.eval()
print(f"[shard {args.shard}] {len(mine)} videos, model={args.model}", flush=True)

t_start = time.time()
done = 0
for vid, path, dur in mine:
    out_path = os.path.join(args.out_dir, f"{vid}.json")
    if os.path.exists(out_path):
        try:
            if json.load(open(out_path)).get("scores") is not None:
                done += 1
                continue
        except Exception:
            pass
    try:
        n_frames = int(max(16, min(args.max_frames, round(dur))))
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
                                        add_generation_prompt=True)
        # NOTE: 4B degrades to all-zero scores without its thinking phase, so
        # default keeps thinking; --no-think is for the 27B fast lane (to be
        # validated on its first outputs before trusting the run)
        if args.no_think:
            text = text.replace("<think>\n", "<think>\n\n</think>\n\n", 1)
        inputs = proc(text=[text], videos=[frames], return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=(384 if args.no_think else 768), do_sample=False,
                             eos_token_id=[proc.tokenizer.eos_token_id])
        raw = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        scores = parse_scores(raw, len(frames))
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
