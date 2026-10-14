#!/usr/bin/env python3
"""零成本统计：426 集里有多少视频上排序器"无判别力"（回退型自适应的前提）。

不依赖任何 GT，只看每个视频内部的分数分布形状。无判别力 = 排序与时间位置
无关，此时降 keep 必亏（丢召回不换精度），应回退到 p=1.0。

三个互补的指标（全部视频内计算，跨视频可比）：
  entropy   : 分数归一化后的分布熵 / ln(n)。越低 => 分数越集中在少数帧。
  mono_rho  : 分数与时间索引的 spearman。|rho|→1 => 纯单调曲线（钟形或斜坡），
              这正是 SFT 模板化坍缩的形态，等于没有逐视频判别力。
  n_eff     : 参与实际排序的有效帧数 = 1/sum(p_i^2)，占 n_frames 的比例。
              极低 => 分数几乎是常数。

判据（三者任一命中即标为"低判别力"，保守门）：
  entropy < E_T 或 |mono_rho| > M_T 或 n_eff/N < K_T

用法: r15c_rankability_audit.py <scores_dir> [<scores_dir2> ...]
"""
import glob
import json
import os
import sys

import numpy as np


INT = {}


def load_intake(path):
    """intake 索引里的真实帧数 + 采样跨度（覆盖整轴 vs 只覆盖子窗口）。"""
    global INT
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            INT[str(r["video_id"])] = (int(r.get("n_frames") or r.get("frame_count") or 0),
                                       int(r.get("n_points") or r.get("n_sample") or 0))
    return len(INT)


def rankdata(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sa = a[o]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def stats_per_video(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        vid = str(r.get("video_id", os.path.basename(p)[:-5]))
        # 排序决策真正发生在采样点上（packager 再广播到全帧），
        # 所以在稀疏分数上度量判别力才是对的尺度。
        s = np.asarray(r["scores"], dtype=float)
        n = len(s)
        if n < 8:
            continue
        span = max(1, len(r.get("frame_indices", [])) - 1)
        true_n = INT.get(str(r.get("video_id")), (0, 0))[0] or 0
        w = s - s.min()
        tot = w.sum()
        if tot <= 1e-12:                      # 常数分
            out[vid] = dict(n=n, entropy=0.0, mono_rho=0.0, n_eff=1.0,
                            n_eff_frac=1.0 / n, const=True,
                            true_n=INT.get(str(r.get("video_id")), (0, 0))[0] or 0,
                            span=span)
            continue
        q = w / tot
        ent = float(-(q * np.log(q + 1e-12)).sum() / np.log(n))
        n_eff = float(1.0 / (q ** 2).sum())
        idx = np.arange(n, dtype=float)
        out[vid] = dict(n=n, entropy=ent, mono_rho=spearman(s, idx),
                        n_eff=n_eff, n_eff_frac=n_eff / n, const=False,
                        true_n=true_n, span=span)
    return out


def report(d, E_T, M_T, K_T):
    st = stats_per_video(d)
    vids = sorted(st)
    ent = np.array([st[v]["entropy"] for v in vids])
    rho = np.array([abs(st[v]["mono_rho"]) for v in vids])
    nef = np.array([st[v]["n_eff_frac"] for v in vids])
    ncs = np.array([st[v]["n"] for v in vids], dtype=float)
    tn = np.array([st[v]["true_n"] for v in vids], dtype=float)
    cov = np.array([st[v]["span"] / max(1.0, st[v]["true_n"]) for v in vids
                    if st[v]["true_n"] > 0])
    print(f"\n================ {os.path.basename(d)} ================")
    print(f"视频数 {len(vids)}  采样点 p50={np.median(ncs):.0f}  "
          f"真实帧数 p50={np.median(tn):.0f}")
    print(f"采样跨度/视频长度: p05={np.percentile(cov,5):.3f} "
          f"中位={np.median(cov):.3f} p95={np.percentile(cov,95):.3f}")
    def q(a, name):
        print(f"  {name:<12} p05={np.percentile(a,5):.4f} p25={np.percentile(a,25):.4f} "
              f"中位={np.median(a):.4f} p75={np.percentile(a,75):.4f} "
              f"p95={np.percentile(a,95):.4f}")
    q(ent, "entropy")
    q(rho, "|mono_rho|")
    q(nef, "n_eff_frac")
    h_e = ent < E_T
    h_m = rho > M_T
    h_k = nef < K_T
    h = h_e | h_m | h_k
    print(f"\n  低判别力命中：entropy<{E_T}: {h_e.sum():>3} | "
          f"|mono_rho|>{M_T}: {h_m.sum():>3} | n_eff_frac<{K_T}: {h_k.sum():>3}")
    print(f"  任一命中（并集）: {h.sum()}/{len(vids)} = {h.mean():.3f}")
    print(f"  三者全不命中（安全，保持 p=0.8534）: {(~h).sum()}/{len(vids)}")
    # 单调方向的细分
    sgn = np.array([np.sign(st[v]["mono_rho"]) for v in vids])
    print(f"\n  |mono_rho|>0.9 的 {int((rho>0.9).sum())} 个里，"
          f"上升(尾部高) {int(((rho>0.9)&(sgn>0)).sum())}，"
          f"下降(头部高) {int(((rho>0.9)&(sgn<0)).sum())}")
    return st, h


if __name__ == "__main__":
    # 阈值取分位数 + 绝对门：entropy 与 n_eff_frac 用 10 分位，mono_rho 用 0.90 绝对门
    intake = os.environ.get("INTAKE", "")
    if intake:
        print(f"intake loaded: {load_intake(intake)} videos")
    E_T, M_T, K_T = 0.90, 0.90, 0.25
    allst = {}
    for d in sys.argv[1:]:
        st, h = report(d, E_T, M_T, K_T)
        allst[d] = (st, h, sorted(st))
    if len(sys.argv) > 1:
        d0 = sys.argv[1]
        st, h, vids = allst[d0]
        ent = np.array([st[v]["entropy"] for v in vids])
        nef = np.array([st[v]["n_eff_frac"] for v in vids])
        print(f"\n>>> 自适应候选规模：entropy 的 10 分位 = {np.percentile(ent,10):.4f}，"
              f"n_eff_frac 的 10 分位 = {np.percentile(nef,10):.4f}")
        print(f">>> 若只按 entropy<10分位 回退：{int((ent<E_T).sum())} 个视频")
    print("AUDIT_DONE")