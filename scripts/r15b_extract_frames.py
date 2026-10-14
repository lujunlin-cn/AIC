#!/usr/bin/env python3
"""R15b：从 PHD2 mp4 抽取均匀切段池的 12 帧，存为 [N,12,288,288,3] uint8 memmap。

每个视频顺序解码一次，收集其所有段所需的帧（避免重复 seek）。
并行进程数 = 视频数（128 核机器上 203 视频一次跑完）。
用法: r15b_extract_frames.py <pool.jsonl> <out.npy> [nproc]
"""
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

SIZE = 288
POOL = sys.argv[1]
OUT = sys.argv[2]
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 64

rows = [json.loads(l) for l in open(POOL)]
N = len(rows)
by_vid = {}
for i, r in enumerate(rows):
    by_vid.setdefault(r["video_id"], []).append(i)


def work(vid):
    """返回 {seg_index: [12,288,288,3] uint8}；纯顺序解码，不做任何 seek
    （cv2 CAP_PROP_POS_MSEC/POS_FRAMES 在这些 h264 上慢到不可用，10-13 实测）。"""
    idxs = by_vid[vid]
    path = rows[idxs[0]]["path"]
    want = {}
    for i in idxs:
        r = rows[i]
        want[i] = [r["seg_t0"] + (r["seg_t1"] - r["seg_t0"]) * (k + 0.5) / 12.0
                   for k in range(12)]
    out = {i: np.zeros((12, SIZE, SIZE, 3), np.uint8) for i in idxs}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return vid, {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-6:
        cap.release()
        return vid, {}
    # 目标时间 -> 目标帧号，按帧号排序后顺序抓
    flat = sorted((int(round(t * fps)), i, k)
                  for i, ts in want.items() for k, t in enumerate(ts))
    ptr, fi = 0, 0
    while ptr < len(flat):
        ok, im = cap.read()
        if not ok:
            break
        t_frame = fi / fps
        while ptr < len(flat) and flat[ptr][0] <= fi:
            _, i, k = flat[ptr]
            if im.shape[:2] != (SIZE, SIZE):
                im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            out[i][k] = im
            ptr += 1
        fi += 1
    cap.release()
    for i in idxs:
        miss = out[i].sum(axis=(1, 2, 3)) == 0
        if miss.any() and not miss.all():
            out[i][miss] = out[i][np.where(~miss)[0][0]]
    return vid, out


if __name__ == "__main__":
    mm = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.uint8,
                                   shape=(N, 12, SIZE, SIZE, 3))
    done = 0
    n_empty = 0
    with Pool(NPROC) as p:
        for vid, o in p.imap_unordered(work, sorted(by_vid), chunksize=1):
            for i, arr in o.items():
                mm[i] = arr
                if not arr.any():
                    n_empty += 1
            done += 1
            if done % 40 == 0:
                print(f"videos {done}/{len(by_vid)}", flush=True)
    mm.flush()
    del mm
    print(f"FRAMES_DONE {OUT} segs={N} empty={n_empty} "
          f"size={os.path.getsize(OUT) / 1e9:.2f}GB")
