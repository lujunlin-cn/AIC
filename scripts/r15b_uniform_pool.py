#!/usr/bin/env python3
"""R15b 均匀切段池构建（判决实验 E1.1 步骤 1-2）。

单变量设计：与 r13_phd2_sft_pool.py 完全相同——
  时长 T 从 426 集 bootstrap、4~60s 截断、12 点相对位置、
  hard[i]=1 iff ∃窗 w: w0-0.03<=pos<=w1+0.03、段最短 4s。
唯一改动：段位置 c 在 [0, dur-T] 上**均匀采样**，不再取高光窗中心。

数据源改为本地反解的 windows.jsonl（克隆机无 PHD2 annotations），
时长用 cv2 元数据读，不用 .info.json。

用法:
  r15b_uniform_pool.py --windows <windows.jsonl> --video-dir <dir>
                      --intake <426.jsonl> --out <pool.jsonl>
                      [--per-video 4] [--seed 13]
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np

CLIP_S, CLIP_E = 4.0, 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--intake", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-dur", type=float, default=600.0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    wins_by_vid = {}
    for line in open(args.windows):
        r = json.loads(line)
        wins_by_vid[r["video_id"]] = [tuple(w) for w in r["windows_abs"]
                                      if w[1] > w[0]]
    d426 = [float(json.loads(l)["duration"]) for l in open(args.intake)]

    vids = []
    for vid, wins in wins_by_vid.items():
        if not wins:
            continue
        p = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        dur = (nf / fps) if (fps > 1e-6 and nf > 1) else 0.0
        if dur <= 0 or dur > args.max_dur:
            continue
        vids.append((vid, p, dur, wins))
    print(f"videos with recoverable duration: {len(vids)}/{len(wins_by_vid)}",
          flush=True)

    segs = []
    n_zero = 0
    offs = []
    for vid, path, dur, wins in sorted(vids):
        for _ in range(args.per_video):
            T = float(np.clip(rng.choice(d426), CLIP_S, CLIP_E))
            span = max(0.0, dur - T)
            if span < 0.5:
                continue
            s0 = rng.uniform(0.0, span)
            s1 = s0 + T
            c = (s0 + s1) / 2.0
            offs.append(min(abs(c - (a + b) / 2.0) for a, b in wins) / T)
            rel = [[round((a - s0) / T, 4), round((b - s0) / T, 4)]
                   for a, b in wins if b > s0 and a < s1]
            pos = np.arange(12) / 11.0
            hard = [int(any(w0 - 0.03 <= p <= w1 + 0.03 for w0, w1 in rel))
                    for p in pos]
            if sum(hard) == 0:
                n_zero += 1
            segs.append({
                "video_id": vid, "path": path,
                "seg_t0": round(s0, 3), "seg_t1": round(s1, 3),
                "duration": round(T, 3), "windows_rel": rel,
                "gif_hard": hard,
                "seg_center_off": round(offs[-1], 4),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for s in segs:
            f.write(json.dumps(s) + "\n")

    hard = np.array([s["gif_hard"] for s in segs])
    nz = hard.sum(1) > 0
    print(f"segments={len(segs)}  non-all-zero={int(nz.sum())} "
          f"({nz.mean():.3f})")
    print(f"seg_center_off<0.05: {np.mean(np.array(offs) < 0.05):.4f} "
          f"(原池 0.9345)")
    print(f"label profile: {np.round(hard.mean(0), 3).tolist()}")
    print("POOL_BUILT")


if __name__ == "__main__":
    main()
