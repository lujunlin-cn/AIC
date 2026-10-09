#!/usr/bin/env python3
"""R13 A2-mini: TVSum 20-video zero-shot certification for one model.

Scores 12 uniformly-sampled frames per video, compares against TVSum
gtscore (annotator-mean) at the sampled positions:
  - mAP@rho50 (Mr.HiSum paper convention: precision of predicted top-50%
    against GT top-50% set, macro-averaged)
  - Spearman rank correlation
Random baseline for mAP@rho50 is ~0.50 by construction.

Usage:
  r13_tvsum_certify.py --model /data/aic/pretrained/Qwen3.5-27B --no-think \
      --max-frames 12 --tag 27b_nothink
  r13_tvsum_certify.py --model /data/aic/pretrained/Qwen3.5-4B --max-frames 16 \
      --tag 4b_think
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/data/aic/scripts")
import r13_qwen35_safe_chunk  # noqa: E402

r13_qwen35_safe_chunk.apply()

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

torch.npu.set_compile_mode(jit_compile=False)

TVSUM_DIR = "/data/aic/external_datasets/TVSum/raw/ydata-tvsum50-v1_1/video"
GT_H5 = "/data/aic/external_datasets/TVSum/raw/eccv16_dataset_tvsum_google_pool5.h5"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--no-think", action="store_true")
ap.add_argument("--max-frames", type=int, default=12)
ap.add_argument("--n-videos", type=int, default=20)
ap.add_argument("--device", default="npu:0")
args = ap.parse_args()

import torch  # noqa: E402,F811
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def map_at_rho(pred, gt, rho=0.5):
    n = len(pred)
    k = max(1, int(round(n * rho)))
    p = set(np.argsort(-np.asarray(pred))[:k].tolist())
    g = set(np.argsort(-np.asarray(gt))[:k].tolist())
    return len(p & g) / k


vids = sorted(f for f in os.listdir(TVSUM_DIR) if f.endswith(".mp4"))[:args.n_videos]
gt_h5 = h5py.File(GT_H5, "r")
gt_map = {k: gt_h5[k]["gtscore"][()] for k in gt_h5.keys()}

proc = AutoProcessor.from_pretrained(args.model, use_fast=False)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.model, torch_dtype=torch.float16, attn_implementation="eager").to(args.device)
model.eval()

maps, spears, n_ok = [], [], 0
t0 = time.time()
for vi, fname in enumerate(vids):
    vid = fname[:-4]
    if vid not in gt_map:
        continue
    path = os.path.join(TVSUM_DIR, fname)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        continue
    gtscore = gt_map[vid]
    gt_T = len(gtscore)
    n_frames = int(max(8, min(args.max_frames, max(8, round(total / 30.0)))))
    n_frames = min(n_frames, gt_T)
    idxs = [min(int(round(i * (total - 1) / max(n_frames - 1, 1))), total - 1)
            for i in range(n_frames)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 4:
        continue
    gtv = [float(gtscore[min(i, gt_T - 1)]) for i in idxs]

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
    if args.no_think:
        text = text.replace("<think>\n", "<think>\n\n</think>\n\n", 1)
    inputs = proc(text=[text], videos=[frames], return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(**inputs,
                             max_new_tokens=(384 if args.no_think else 768),
                             do_sample=False,
                             eos_token_id=[proc.tokenizer.eos_token_id])
    raw = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    import re as _re
    best = None
    for mch in _re.finditer(r"\[[^\[\]]*\]", raw):
        try:
            arr = json.loads(mch.group(0))
            if isinstance(arr, list) and len(arr) == len(frames):
                best = arr
                break
        except Exception:
            pass
    if best is None:
        fl = _re.findall(r"-?\d+\.\d+|-?\d+", raw)
        if len(fl) >= len(frames):
            best = [float(x) for x in fl[-len(frames):]]
    if best is None:
        print(f"[{args.tag}] {vid} PARSE_FAIL", flush=True)
        continue
    n_ok += 1
    maps.append(map_at_rho(best, gtv))
    spears.append(spearman(best, gtv))
    print(f"[{args.tag}] {vid} map50={maps[-1]:.3f} rho={spears[-1]:.3f} "
          f"({vi+1}/{len(vids)})", flush=True)

res = {"tag": args.tag, "model": args.model, "n_videos": n_ok,
       "map50_mean": round(float(np.mean(maps)) * 100, 2),
       "spearman_mean": round(float(np.mean(spears)), 4),
       "elapsed_min": round((time.time() - t0) / 60, 1)}
os.makedirs("/data/aic/experiments_910a/LFM_V13/tvsum_cert", exist_ok=True)
with open(f"/data/aic/experiments_910a/LFM_V13/tvsum_cert/{args.tag}.json", "w") as f:
    json.dump({**res, "per_video": dict(zip([v[:-4] for v in vids], zip(maps, spears))},
              f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
print("CERT_DONE")
