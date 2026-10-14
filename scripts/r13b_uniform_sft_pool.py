#!/usr/bin/env python3
# r13b：均匀切段池构建器 —— 判决「L≈1.05 是否由切段器居中偏置造成」
#
# 【根因】scripts/r13_phd2_sft_pool.py:81-83 用 `c=(t0+t1)/2`（高光窗中心）
#   定位切段，`s0=c-T/2`。实测 93.5% 的段 |c-0.5|<0.05，标签位置剖面是完美钟形，
#   固定位置曲线 holdout spearman 0.750 > 我们最好的 ckpt_030 0.539。标签信息
#   上限 0.878，位置占 0.750，内容只剩 0.128。部署到 426 集整轴推理时位置先验
#   不存在 → 可迁移内容信号≈0 → L≈1.05。
#
# 【单变量设计】本文件与 r13_phd2_sft_pool.py 逐字相同的部分：
#   - 视频源遍历 / 跨用户一致性过滤 iou_m / merge_windows / 时长 T 从 426 集
#     bootstrap / 4s~60s 截断 / 抽帧 / teacher 打分 / label 组装规则
# 唯一改动：
#   - 段位置 `c` 不再取高光窗中心，改为在 [0, dur-T] 上**均匀采样**（seed 可控）
#   - windows_rel 仍按同一公式换算（窗落在段外则该段可能全零——这是预期，
#     它正是"位置先验失效"的检验对象）
#
# 用法: r13b_uniform_sft_pool.py --out <pool.jsonl> [--seed 13] [--per-video 1]
#                              [--root /data/aic/external_datasets/PHD2]
#                              [--intake /data/aic/semifinal_20261001/intake/index.enriched.jsonl]
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

ROOT_DEFAULT = "/data/aic/external_datasets/PHD2"
# 与 r13_phd2_sft_pool.py 对齐
CLIP_S, CLIP_E = 4.0, 60.0


def iou(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / max(
        1e-9, (a[1] - a[0]) + (b[1] - b[0]) - max(0.0, min(a[1], b[1]) - max(a[0], b[0])))


def merge_windows(wins, gap=2.0):
    """与原实现一致：按 gap 合并相邻窗。"""
    if not wins:
        return []
    wins = sorted(wins, key=lambda w: w[0])
    out = [list(wins[0])]
    for t0, t1 in wins[1:]:
        if t0 <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    return [tuple(w) for w in out]


def sample_uniform_seg(rng, dur, T, windows_abs):
    """在 [0, dur-T] 上均匀采段位置；偏离高光窗中心的程度记为 seg_center_off。

    关键：这里**不**用任何窗信息选位置。窗只在标签阶段用。
    """
    span = max(0.0, dur - T)
    s0 = rng.uniform(0.0, span) if span > 0 else 0.0
    s1 = s0 + T
    # 段中心与最近高光窗中心的距离（归一到段长），用于诊断分布
    c = (s0 + s1) / 2.0
    off = min((abs(c - (a + b) / 2.0) for a, b in windows_abs), default=float("nan"))
    return s0, s1, off / max(T, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--intake", default="/data/aic/semifinal_20261001/intake/"
                                         "index.enriched.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--per-video", type=int, default=1,
                    help="每个视频采几段（默认 1，与原实现同量级）")
    ap.add_argument("--max-seg-len", type=float, default=CLIP_E)
    args = ap.parse_args()
    ROOT = args.root
    rng = random.Random(args.seed)
    random.seed(args.seed)

    sel_tr = json.load(open(f"{ROOT}/annotations/selections/train.json"))
    sel_te = set(json.load(open(f"{ROOT}/annotations/selections/test.json")))
    d426 = [float(json.loads(l)["duration"]) for l in open(args.intake)]

    segs = []
    offs = []
    n_all_zero = 0
    for inf in sorted(glob.glob(f"{ROOT}/raw/youtube/*.info.json")):
        vid = os.path.basename(inf)[: -len(".info.json")]
        if vid in sel_te:                     # 与原实现同：排除 test
            continue
        sel = sel_tr.get(vid)
        if not sel:
            continue
        try:
            dur = float(json.load(open(inf)).get("duration") or 0)
        except Exception:
            continue
        if dur <= 0 or dur > 600:
            continue
        users = list(sel.values())
        n_users = len(users)
        wins_u = [[w for w in u if w.get("t1", 0) > w.get("t0", 0)] for u in users]
        wins_u = [w for w in wins_u if w]
        if not wins_u:
            continue
        iou_m = 0.0
        if n_users >= 2:
            ious = [max(iou((a["t0"], a["t1"]), (b["t0"], b["t1"]))
                        for a in wa for b in wb)
                    for i, wa in enumerate(wins_u)
                    for j, wb in enumerate(wins_u) if i < j]
            iou_m = float(np.mean(ious)) if ious else 0.0
        merged = merge_windows([w for u in wins_u for w in u])
        if not merged:
            continue
        for _ in range(args.per_video):
            T = float(np.clip(rng.choice(d426), CLIP_S, args.max_seg_len))
            s0, s1, off = sample_uniform_seg(rng, dur, T, merged)
            if s1 - s0 < CLIP_S:
                continue
            rel = [[round((a - s0) / (s1 - s0), 4), round((b - s0) / (s1 - s0), 4)]
                   for a, b in [(t0, t1) for t0, t1 in merged
                                if t1 > s0 and t0 < s1]]
            hard = None
            if rel:
                pos = np.arange(12) / 11.0
                hard = [int(any(w0 - 0.03 <= p <= w1 + 0.03 for w0, w1 in rel))
                        for p in pos]
                if sum(hard) == 0:
                    n_all_zero += 1
            if hard is None:
                continue
            offs.append(off)
            segs.append({
                "video_id": vid,
                "path": f"{ROOT}/raw/youtube/{vid}.mp4",
                "seg_t0": round(s0, 2), "seg_t1": round(s1, 2),
                "duration": round(s1 - s0, 2), "n_users": n_users,
                "iou_mean": iou_m, "windows_rel": rel,
                "gif_hard": hard, "seg_center_off": round(off, 4),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for s in segs:
            f.write(json.dumps(s) + "\n")

    # ---- 判决统计：位置先验基线塌没塌 ----
    prof = np.zeros(12)
    for s in segs:
        for i, h in enumerate(s["gif_hard"]):
            prof[i] += h
    prof /= max(1, len(segs))
    bell = np.array([0.16, 0.22, 0.30, 0.43, 0.64, 0.95, 0.94,
                     0.64, 0.42, 0.28, 0.21, 0.15])  # 原池实测剖面
    flat = np.full(12, 1.0 / 12)
    flat_pos = np.arange(12) / 11.0

    def corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    nz = [s for s in segs if sum(s["gif_hard"]) > 0]
    bell_r = [corr(bell, s["gif_hard"]) for s in nz]
    flat_r = [corr(flat, s["gif_hard"]) for s in nz]
    ramp_r = [corr(flat_pos, s["gif_hard"]) for s in nz]
    nz_frac = len(nz) / max(1, len(segs))
    print(f"segments={len(segs)}  non-all-zero={len(nz)} ({nz_frac:.3f})")
    print(f"seg_center_off: p50={np.nanmedian(offs):.4f} "
          f"(原池 93.5% 的段 <0.05)")
    print(f"label profile: {np.round(prof, 3).tolist()}")
    print("---- 位置先验基线（判决量，与原池 0.697 同口径）----")
    print(f"  fixed bell curve : {np.nanmean(bell_r):.4f}   (原池 0.6970)")
    print(f"  flat constant   : {np.nanmean(flat_r):.4f}")
    print(f"  linear ramp     : {np.nanmean(ramp_r):.4f}   (原池 0.5936)")
    print("VERDICT_HINT: bell_r < 0.30 且模型仍 >0.4 => 内容信号被切段器掩盖，"
          "L 可救；否则内容信号本身不存在。")
    print("POOL_DONE")


if __name__ == "__main__":
    main()
