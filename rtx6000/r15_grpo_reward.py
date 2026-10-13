#!/usr/bin/env python3
"""R13 GRPO reward module v3 (2026-10-10, per reports/r13/rl_research_verdict.md).

v3 replaces the v2 ranking-proxy main term with the official-caliber analog:

  r_raw = 0.7 * R_F + 0.2 * R_num + 0.1 * format
  r     = shape_reward(r_raw)          # TempSamp-R1 non-linear shaping

  R_F   = 2 * sum(IoU_frame[i in topk]) / (k + n_gt)
          (official F_i = 2*S_IoU/(N_pred+N_gt) projected to the segment;
           gif GT has no boxes -> IoU_frame defaults to 1.0 = pure frame-
           selection projection. Corner cases per official rule:
           n_gt==0 and k==0 -> 1.0 ; n_gt==0 xor k==0 -> 0.0)
  R_num = exp(-|k - n_gt| / min(n_gt,3))      (TempR1 TAL count reward;
          n_gt==0: 1.0 iff k==0 -> empty-segment incentive built in,
          guards the all-low-score hack on the 85% n_gt>0 pool)
  topk  = {i : score_i >= tau}, tau=0.5

shaping (TempSamp-R1 eq.4): tau=0.8, a1=0.01, a2=1.0 — log-compress above
the threshold, exponentially expand below it; our R_raw lives in 0.1..1.0
so most groups sit on the expanding branch, which is exactly the fix for
sparse intra-group signal.

v2 (AP@rho/Spearman/range) is kept below as legacy for A/B re-runs.
"""
import json
import math

import numpy as np

W_RF, W_RN, W_FMT = 0.7, 0.2, 0.1
TAU_SEL = 0.5
SHAPE_TAU, SHAPE_A1, SHAPE_A2 = 0.8, 0.01, 1.0


def parse_scores(raw, n):
    """Extract an n-length score list from raw text. Returns
    (scores_or_None, fmt_ok). Constant outputs count as parsed (fmt=1):
    the R_F/R_num terms punish them through selection geometry."""
    if raw is None:
        return None, False
    import re
    best = None
    for m in re.finditer(r"\[[^\[\]]*\]", str(raw)):
        try:
            arr = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(arr, list) and len(arr) == n and all(
                isinstance(x, (int, float)) for x in arr):
            best = arr
            break
    if best is None:
        return None, False
    vals = [float(x) for x in best]
    mx = max(vals) if vals else 1.0
    if mx > 1.5:
        vals = [v / 100.0 for v in vals]
    vals = [min(max(v, 0.0), 1.0) for v in vals]
    return vals, True


def rf_score(sel, n_gt, iou_frame=None, gif_hard=None):
    """Official F_i segment analog. sel: bool per frame; iou_frame: per-frame
    IoU of the frozen composition box vs GT (None -> 1.0: FRAG gif GT has no
    boxes, so R_F degenerates to the pure frame-selection projection)."""
    k = int(sum(1 for b in sel if b))
    if n_gt == 0 and k == 0:
        return 1.0
    if n_gt == 0 or k == 0:
        return 0.0
    # Official semantics: only selected frames that HIT a GT frame contribute
    # IoU; non-GT selections only inflate N_pred (the denominator lever).
    if iou_frame is None:
        s = float(sum(1 for i, b in enumerate(sel) if b and gif_hard[i]))
    else:
        iou = np.asarray(iou_frame, dtype=float)
        s = float(sum(iou[i] for i, b in enumerate(sel) if b and gif_hard[i]))
    return 2.0 * s / (k + n_gt)


def rnum_score(k, n_gt):
    """TempR1 TAL count reward, empty-segment incentive built in."""
    if n_gt == 0:
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-abs(k - n_gt) / float(max(1, min(n_gt, 3)))))


def shape_reward(r, tau=SHAPE_TAU, a1=SHAPE_A1, a2=SHAPE_A2):
    """TempSamp-R1 eq.4 non-linear shaping: compress above tau (log),
    expand below tau (exp) -> amplifies gaps among suboptimal rollouts."""
    if r >= tau:
        return tau + a1 * math.log((r - tau) + 1.0)
    return tau - (math.exp(a2 * (tau - r)) - 1.0) / a2


def reward_v3(scores, gif_hard, iou_frame=None, fmt_ok=True, tau=TAU_SEL):
    """Full v3 reward. scores: list or None (unparsable). Returns
    (shaped_R, raw_R, parts)."""
    n_gt = int(sum(1 for b in gif_hard if b))
    if scores is None:
        sel = [False] * len(gif_hard)
    else:
        sel = [s >= tau for s in scores]
    k = int(sum(1 for b in sel if b))
    rf = rf_score(sel, n_gt, iou_frame, gif_hard=gif_hard)
    rn = rnum_score(k, n_gt)
    fm = 1.0 if fmt_ok else 0.0
    raw = W_RF * rf + W_RN * rn + W_FMT * fm
    parts = {"rf": round(rf, 4), "rnum": round(rn, 4), "fmt": fm,
             "k": k, "n_gt": n_gt}
    return shape_reward(raw), raw, parts


# ---------------- legacy v2 (kept for A/B re-runs) ----------------
W_RANK, W_CALIB, W_RANGE, W_FORMAT = 0.65, 0.15, 0.15, 0.05


def rank_normalize(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    if n <= 1:
        return np.zeros(n)
    return (ranks - 1.0) / (n - 1.0)


def spearman(a, b):
    ra, rb = rank_normalize(a), rank_normalize(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den < 1e-12:
        return 0.0
    return float((ra * rb).sum() / den)


def reward_v2(s, g, rng_full=None):
    """Legacy v2 scalar (rank main term) for A/B; not used by GRPO v3."""
    s = np.asarray(s, dtype=float)
    g = np.asarray(g, dtype=float)
    cb = spearman(s, g)
    rg = 0.0 if rng_full is None else float(
        np.clip((rng_full - 0.15) / 0.45, 0.0, 1.0))
    return 0.85 * cb + 0.15 * rg


if __name__ == "__main__":
    # sanity: synthetic group, n_gt=1 dominant FRAG case + empty segment
    gt = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    perfect = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    loose = [0.6, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    wrong = [0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    alllow = [0.1] * 12
    parsed, fmt = parse_scores(json.dumps(perfect), 12)
    assert fmt and abs(parsed[1] - 0.9) < 1e-9
    r_p, raw_p, p_p = reward_v3(perfect, gt)
    r_l, raw_l, p_l = reward_v3(loose, gt)
    r_w, raw_w, p_w = reward_v3(wrong, gt)
    r_e, raw_e, p_e = reward_v3(alllow, gt)
    r_n, raw_n, p_n = reward_v3(None, gt)
    print(f"perfect raw={raw_p:.3f} shaped={r_p:.3f} {p_p}")
    print(f"loose   raw={raw_l:.3f} shaped={r_l:.3f} {p_l}")
    print(f"wrong   raw={raw_w:.3f} shaped={r_w:.3f} {p_w}")
    print(f"alllow  raw={raw_e:.3f} shaped={r_e:.3f} {p_e}")
    print(f"noparse raw={raw_n:.3f} shaped={r_n:.3f} {p_n}")
    assert raw_p > raw_l > raw_n, "ordering violated"
    assert raw_e < 0.35, "all-low on n_gt=1 must be punished hard"
    gt0 = [0] * 12
    r0, raw0, p0 = reward_v3(alllow, gt0)
    print(f"empty seg alllow raw={raw0:.3f} {p0}")
    assert raw0 > 0.95, "empty seg + empty topk must be ~full marks"
    r0b, raw0b, p0b = reward_v3(perfect, gt0)
    assert raw0b < 0.2, "empty seg + selections must be punished"
    print("v3 asserts passed")


# ---------------- v4 (2026-10-11): temporal-Wasserstein frame reward ----------------
# TimeLens2 (arXiv 2607.17423) Table 8/9: overlap rewards give zero signal to
# all-zero groups (our smoke: 56% zero_adv); a 1D Wasserstein term on the
# normalized timeline breaks ties and hands near-miss errors a
# distance-proportional gradient (their rescue: 75.8% of zero groups).
W_RF4, W_RTW, W_RN4, W_FMT4 = 0.55, 0.15, 0.20, 0.10


def cdf_w1(scores, gif_hard, eps=1e-6):
    """Discrete 1-Wasserstein between the normalized score mass and the
    gif_hard mass on the segment timeline [0,1] (equal bins):
        W1 = mean_i |CDF_pred(i) - CDF_gt(i)|.
    Both masses get +eps smoothing so an all-zero prediction stays defined
    (it becomes uniform mass -> mid reward: 'uncertain' beats 'wrong')."""
    n = len(gif_hard)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0) + eps
    p = p / p.sum()
    g = np.asarray(gif_hard, dtype=float) + eps
    g = g / g.sum()
    return float(np.abs(np.cumsum(p) - np.cumsum(g)).mean())


def rtw_score(scores, gif_hard, smooth=0.15):
    """R_TW = exp(-W1 / (gt_duration_fraction + smooth)); empty segment uses
    a uniform reference mass, which punishes concentrated high scores there
    (consistent with R_F/R_num empty-segment semantics)."""
    n_gt = int(sum(1 for b in gif_hard if b))
    n = len(gif_hard)
    if scores is None:
        return 0.0  # unparsable: no mass information, fmt term handles it
    if n_gt == 0:
        g = np.full(n, 1.0 / n)
    else:
        g = np.asarray(gif_hard, dtype=float)
        g = g / g.sum()
    w1 = cdf_w1(scores, list(g))
    return float(math.exp(-w1 / (n_gt / float(n) + smooth)))


def reward_v4(scores, gif_hard, iou_frame=None, fmt_ok=True, tau=TAU_SEL):
    """v4 = 0.55*R_F + 0.15*R_TW + 0.2*R_num + 0.1*fmt, then shaping.
    Same signature as reward_v3 for drop-in use."""
    n_gt = int(sum(1 for b in gif_hard if b))
    if scores is None:
        sel = [False] * len(gif_hard)
    else:
        sel = [s >= tau for s in scores]
    k = int(sum(1 for b in sel if b))
    rf = rf_score(sel, n_gt, iou_frame, gif_hard=gif_hard)
    rtw = rtw_score(scores, gif_hard)
    rn = rnum_score(k, n_gt)
    fm = 1.0 if fmt_ok else 0.0
    raw = W_RF4 * rf + W_RTW * rtw + W_RN4 * rn + W_FMT4 * fm
    parts = {"rf": round(rf, 4), "rtw": round(rtw, 4), "rnum": round(rn, 4),
             "fmt": fm, "k": k, "n_gt": n_gt}
    return shape_reward(raw), raw, parts


if __name__ == "__v4_tests__":
    pass


# ------------- v5 (2026-10-13): pairwise ranking reward -------------
# Point-estimate rewards (R_F topk / RTW) can be partially harvested by an
# input-agnostic prior ordering (the "template hill"). A rank-correlation
# reward is still harvestable by the prior, so it MUST be paired with the
# counterfactual de-prioritization baseline (orarl_advantage cf_r=...).
W_PAIR_SP, W_PAIR_FMT = 0.9, 0.1


def reward_v6(scores, target, gif_hard, iou_frame=None, fmt_ok=True,
              tau=TAU_SEL):
    """v6 = 0.5*R_F + 0.3*R_TW + 0.2*pair_core (2026-10-13).

    keep-set 模拟(R_F, 官方 F_i analog) + 时间容差(W1) + 段内排序(spearman
    vs fused target)。官方裁决(run7 35.04 < 35.44)后引入: pair-only 把
    弧线先验当敌人清零(detmpl 换 spearman 负净值); v6 保留先验的 keep
    结构, 只在其上优化排序与命中。iou_frame 缺失时 R_F 退化为纯命中
    (rf_score 设计内)。"""
    sel = ([s >= tau for s in scores] if scores is not None
           else [False] * len(gif_hard))
    k = int(sum(1 for b in sel if b))
    n_gt = int(sum(1 for b in gif_hard if b))
    rf = rf_score(sel, n_gt, iou_frame, gif_hard=gif_hard)
    rtw = rtw_score(scores, gif_hard)
    if scores is not None and fmt_ok and len(scores) == len(target):
        pr = W_PAIR_SP * spearman(np.asarray(scores, float),
                                  np.asarray(target, float))
    else:
        pr = 0.0
    raw = 0.5 * rf + 0.3 * rtw + 0.2 * pr
    return shape_reward(raw), raw, {"rf": round(rf, 4), "rtw": round(rtw, 4),
                                    "pair": round(pr, 4), "k": k,
                                    "n_gt": n_gt}


def pairwise_reward(scores, target, fmt_ok=True):
    """v5: r = 0.9*spearman(scores, target) + 0.1*fmt. Unparsable -> -0.1."""
    if scores is None or not fmt_ok:
        return -0.1
    sp = spearman(np.asarray(scores, float), np.asarray(target, float))
    return W_PAIR_SP * float(sp) + W_PAIR_FMT * 1.0
