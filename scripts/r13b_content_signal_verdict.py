#!/usr/bin/env python3
# r13b 判决实验：内容信号是否被切段器居中偏置掩盖？
#
# 依赖 r13b_uniform_sft_pool.py 产出的均匀切段池（含 12 帧 jpgs 的 frames_dir）。
# 三组读数，全部零 GPU：
#   (1) 位置先验基线（bell / flat / ramp）——应从原池 0.697 塌到 <0.30
#   (2) 段中心偏移分布 ——应从 93.5% (<0.05) 变为弥散
#   (3) 段内标签 vs 内容代理的运动能量 ——若 >0 说明标签含非位置内容信号
#   (4) 教师分数 spearman ——教师是内容信号，与新标签的同向性
#
# 用法: r13b_content_signal_verdict.py <uniform_pool.jsonl> [<orig_pool.jsonl>]
import json
import sys

import numpy as np

POS = np.arange(12) / 11.0
BELL = np.array([0.16, 0.22, 0.30, 0.43, 0.64, 0.95, 0.94,
                 0.64, 0.42, 0.28, 0.21, 0.15])
FLAT = np.full(12, 1.0 / 12.0)


def rankdata(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
    return r


def spear(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def load(p):
    rows = []
    for line in open(p):
        r = json.loads(line)
        rows.append(r)
    return rows


def get_hard(r):
    if "gif_hard" in r:
        return r["gif_hard"]
    w = r.get("windows_rel", [])
    return [int(any(a - 0.03 <= p <= b + 0.03 for a, b in w)) for p in POS]


def report(rows, name):
    hard = np.array([get_hard(r) for r in rows], dtype=int)
    nz_mask = hard.sum(1) > 0
    nz = hard[nz_mask]
    prof = hard.mean(0)
    offs = [r.get("seg_center_off") for r in rows if r.get("seg_center_off") is not None]
    print(f"\n===== {name} =====")
    print(f"segs={len(rows)}  non-all-zero={nz_mask.sum()} ({nz_mask.mean():.3f})")
    if offs:
        offs = np.array(offs)
        print(f"seg_center_off: p50={np.nanmedian(offs):.4f} "
              f"frac(<0.05)={np.mean(offs < 0.05):.3f}  (原池 0.935)")
    print(f"label profile: {np.round(prof, 3).tolist()}")
    print(f"  bell profile corr vs labels: {spear(BELL, prof):+.4f}")
    per_bell = [spear(BELL, h) for h in nz]
    per_flat = [spear(FLAT, h) for h in nz]
    per_ramp = [spear(POS, h) for h in nz]
    print("---- 位置先验基线（段内 spearman vs gif_hard）----")
    print(f"  fixed bell  : {np.nanmean(per_bell):+.4f}   [原池 0.6970 判决基准]")
    print(f"  flat const  : {np.nanmean(per_flat):+.4f}")
    print(f"  linear ramp : {np.nanmean(per_ramp):+.4f}   [原池 0.5936]")
    # 熵：标签信息量
    ent = -(prof * np.log2(prof + 1e-12) + (1 - prof) *
            np.log2(1 - prof + 1e-12)).sum()
    print(f"label entropy: {ent:.3f} bits (max 3.585)  "
          f"nonzero-rate {hard.mean():.3f}")
    return {"n": len(rows), "nz_frac": float(nz_mask.mean()),
            "bell_per_seg": float(np.nanmean(per_bell)),
            "ramp_per_seg": float(np.nanmean(per_ramp)),
            "profile": prof.tolist(), "entropy": float(ent)}


if __name__ == "__main__":
    uni = sys.argv[1]
    res = {"uniform": report(load(uni), "UNIFORM (r13b)")}
    if len(sys.argv) > 2:
        res["orig"] = report(load(sys.argv[2]), "ORIG (r13, 高光窗居中)")
    u = res["uniform"]
    o = res.get("orig")
    print("\n---- 判决 ----")
    if o:
        print(f"bell 基线: {o['bell_per_seg']:+.4f} -> {u['bell_per_seg']:+.4f} "
              f"(Δ {u['bell_per_seg'] - o['bell_per_seg']:+.4f})")
        drop = abs(u["bell_per_seg"]) < 0.30
        print(f"位置先验是否塌到 <0.30: {'YES' if drop else 'NO'}")
        print("若 YES 且模型仍 >0.4 => 内容信号被掩盖，L 可救")
        print("若 YES 且模型也塌 => 内容信号本身不存在，35.52 为终局")
    else:
        print("bell 基线 = " + f"{u['bell_per_seg']:+.4f}  (<0.30 为通过)")
