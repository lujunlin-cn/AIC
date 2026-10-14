#!/usr/bin/env python3
"""从已知官方分 + 实测帧数反解排序提升倍数 L = rho(p)/p。

依据赛事规则 01_赛事规则与评分标准.md:147-158（规则原文，非 eval_official.py）：
    P_i = S_IoU,i / N_pred,i ;  R_i = S_IoU,i / N_gt,i
    F_i = 2*P_i*R_i/(P_i+R_i) = 2*S_IoU,i / (N_pred,i + N_gt,i)
    F_video = (1/V) * sum_i F_i                      （macro）
k_size 已取消，Score = F1 * 100，不做任何惩罚换算。

记 N = 视频总帧数，d_i = GT 帧占比，p_i = 预测帧占比，
    rho_i(p) = |pred ∩ gt| / N_gt,i   （给定 p 下的召回）
    Rbar_i   = 匹配帧的平均空间 IoU
则 S_IoU,i = rho_i(p) * d_i * N * Rbar_i，于是
    F_i(p) = 2*Rbar_i*rho_i(p)*d_i / (p_i + d_i)
全帧包 p=1 时 rho≡1：
    F_i(1) = 2*Rbar_i*d_i / (1 + d_i)
两包共享同一空间框 => Rbar_i 相同，相除即可消去它：
    rho = (F_keep/F_full) * (p + d)/(1 + d)
    L   = rho / p
"""
import glob
import json
import os
import sys

import numpy as np

# 已知官方分（Score = F1*100，k_size 已取消）
KNOWN = {
    "v8b3": 34.75,   # 全帧预测 p=1.000
    "v8t": 34.88,    # keep 0.80
    "pkgA": 35.17,   # R12_GATT_LFM 族
    "pkgB": 35.52,   # VPB 融合
    "pkgC": 35.52,   # run10 VPB 注入（与 pkgB 同分，按同一分组处理）
}
ROOT = "/tmp/run7_pkg"


def load(name):
    p = os.path.join(ROOT, name, "predictions.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[str(r["video_id"])] = {int(x["frame"]) for x in r["predictions"]}
    return out


def stats(pkgs):
    """返回 {包名: (per-video n_pred dict, 总帧数 dict)}；总帧数用全帧包为基准。"""
    return {n: load(n) for n in pkgs}


def solve(base, keep, d_grid):
    """base/keep 为包名。d 为标量假设下的解析解（先给标量，再给区间）。"""
    fb, fk = load(base), load(keep)
    vids = sorted(set(fb) & set(fk))
    n_full = {v: len(fb[v]) for v in vids}
    n_keep = {v: len(fk[v]) for v in vids}
    # p 的取值：全帧包 p=1；keep 包 p = n_keep/n_full
    p_v = {v: n_keep[v] / n_full[v] for v in vids}
    p_mean = float(np.mean([p_v[v] for v in vids]))
    F_b, F_k = KNOWN[base] / 100.0, KNOWN[keep] / 100.0
    print(f"\n=== {keep} vs {base}  (官方 {KNOWN[keep]:.2f} vs {KNOWN[base]:.2f}) ===")
    print(f"视频数 {len(vids)} | 全帧包总帧 {sum(n_full.values())} | "
          f"keep 包总帧 {sum(n_keep.values())} | p_mean={p_mean:.4f}")
    print(f"{'d假设':>8} | {'rho':>8} | {'L=rho/p':>9} | {'随机基线rho':>11}")
    print("-" * 48)
    rows = []
    for d in d_grid:
        rho = (F_k / F_b) * (p_mean + d) / (1.0 + d)
        L = rho / p_mean
        rows.append((d, rho, L))
        print(f"{d:8.3f} | {rho:8.4f} | {L:9.4f} | {p_mean:11.4f}")
    return p_mean, rows


def target_ratio_check():
    """targetRatioWH 规则 96 行：必须与样本指定目标比例一致。
    bh 由 bw*rh/rw 导出并同时施加于 pred/ref，比例错则两侧同错。"""
    bad = 0
    tot = 0
    dist = {}
    for n in ("pkgA", "pkgB", "v8b3"):
        p = os.path.join(ROOT, n, "predictions.jsonl")
        if not os.path.exists(p):
            continue
        cnt = {}
        for line in open(p):
            r = json.loads(line)
            key = tuple(r.get("targetRatioWH", []))
            cnt[key] = cnt.get(key, 0) + 1
        print(f"\n{n} targetRatioWH 分布: "
              f"{ {f'{k[0]}:{k[1]}': v for k, v in cnt.items()} }")
        tot += sum(cnt.values())
        bad += cnt.get((9, 16), 0) + cnt.get((9.0, 16.0), 0)
    print(f"竖屏(9:16)占比合计 {bad}/{tot} = {bad / max(1, tot):.4f}")
    return bad, tot


if __name__ == "__main__":
    d_grid = [0.15, 0.20, 0.21, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    for keep in ("v8t", "pkgA", "pkgB", "pkgC"):
        try:
            solve("v8b3", keep, d_grid)
        except Exception as e:
            print(f"{keep}: {e}")
    print("\n" + "=" * 60)
    target_ratio_check()
    print("SOLVE_DONE")
