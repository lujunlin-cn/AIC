"""Versioned ranking and TVSum author-style summary benchmarks; never AIC F_video."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import kendalltau, rankdata

RANKING_PROTOCOL = "TVSUM_RANKING_V2"
SUMMARY_PROTOCOL = "TVSUM_SUMMARY_V1_FIXED"
AUTHOR_REVISION = "7bf3fb8ca032f5f4b42e22ba41f5e2490b819f96"

def _pair(pred, target):
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 1 or p.shape != t.shape or not len(p):
        raise ValueError("prediction and target must be matching nonempty vectors")
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("scores must be finite")
    return p, t

def spearman(pred, target):
    p, t = _pair(pred, target)
    p, t = rankdata(p), rankdata(t)
    p -= p.mean(); t -= t.mean()
    den = np.linalg.norm(p) * np.linalg.norm(t)
    return float(p @ t / den) if den else None

def kendall_tau(pred, target):
    p, t = _pair(pred, target)
    v = float(kendalltau(p, t, variant="b").statistic)
    return v if math.isfinite(v) else None

def ndcg(pred, target, k=None):
    p, t = _pair(pred, target)
    if (t < 0).any(): raise ValueError("NDCG requires nonnegative relevance")
    k = len(p) if k is None else min(int(k), len(p))
    if k <= 0: raise ValueError("NDCG k must be positive")
    discount = np.zeros(len(p)); discount[:k] = 1 / np.log2(np.arange(2, k + 2))
    order = np.argsort(-p, kind="stable")
    groups = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1, len(p)]
    dcg = sum(float(t[order[a:b]].mean() * discount[a:b].sum())
              for a, b in zip(groups[:-1], groups[1:]))
    ideal = float(np.sort(t)[::-1] @ discount)
    return dcg / ideal if ideal > 0 else None

def average_precision(pred, truth):
    p, y = _pair(pred, truth)
    if not np.isin(y, [0, 1]).all(): raise ValueError("AP requires binary target")
    if y.sum() == 0: return None
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(y[order])[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / y.sum() * tp / (ends + 1)))

def ranking_report(pred, target):
    p, t = _pair(pred, target); k = max(1, int(math.floor(len(t) * .15)))
    order = np.argsort(-p, kind="stable")[:k]
    return {"spearman": spearman(p,t), "kendall_tau_b": kendall_tau(p,t),
            "ndcg": ndcg(p,t), "ndcg_at_15pct": ndcg(p,t,k),
            "ap_fixed_gt_0p5": average_precision(p,t >= .5),
            "top15_mean_relevance": float(t[order].mean())}

def fixed_segments(nframes, length=60):
    """Zero-based half-open author's gt_seg; remainder merges into last shot."""
    if nframes <= 0 or length <= 0: raise ValueError("positive sizes required")
    boundaries = np.arange(length, nframes + 1, length)[:-1]
    return np.column_stack([np.r_[0,boundaries], np.r_[boundaries,nframes]])

def knapsack(weights, values, capacity):
    """0/1 DP; ties exclude later items exactly as author's backtracking."""
    w, v = np.asarray(weights), np.asarray(values, dtype=float)
    if w.ndim != 1 or w.shape != v.shape or capacity < 0: raise ValueError("invalid shapes")
    if np.any(w <= 0) or np.any(w != np.floor(w)) or not np.isfinite(v).all():
        raise ValueError("positive integer weights and finite values required")
    w=w.astype(int); table=np.zeros((len(w)+1, capacity+1))
    for i,(wi,vi) in enumerate(zip(w,v),1):
        table[i]=table[i-1]
        if wi <= capacity:
            table[i,wi:]=np.maximum(table[i-1,wi:],table[i-1,:capacity+1-wi]+vi)
    selected=np.zeros(len(w),bool); remaining=capacity
    for i in range(len(w),0,-1):
        if table[i,remaining] > table[i-1,remaining]:
            selected[i-1]=True; remaining-=w[i-1]
    return selected

def summary_mask(scores, segments=None, budget=.15):
    scores,_=_pair(scores,scores)
    if not 0 < budget <= 1: raise ValueError("budget in (0,1] required")
    seg=fixed_segments(len(scores)) if segments is None else np.asarray(segments)
    if (seg.ndim != 2 or seg.shape[1] != 2 or seg[0,0] != 0 or seg[-1,1] != len(scores)
        or np.any(seg[:,1] <= seg[:,0]) or np.any(seg[:-1,1] != seg[1:,0])):
        raise ValueError("segments must partition original frames")
    means=np.array([scores[a:b].mean() for a,b in seg])
    selected=knapsack(seg[:,1]-seg[:,0],means,int(budget*len(scores)))
    out=np.zeros(len(scores),bool)
    for (a,b),keep in zip(seg,selected):
        if keep: out[a:b]=True
    # Match author fallback, including ceil budget on constant short sequences.
    if not out.any(): out=scores > np.quantile(scores,1-budget,method="hazen")
    if not out.any(): out[np.argsort(-scores,kind="stable")[:math.ceil(budget*len(scores))]]=True
    return out

def human_summaries(user_anno):
    a=np.asarray(user_anno,dtype=float)
    if a.ndim != 2 or a.shape[0] != 20 or not np.isfinite(a).all():
        raise ValueError("TVSum annotations must be finite [20,nframes]")
    return np.stack([summary_mask(row) for row in a])

def summary_report(frame_scores,human_masks):
    p,_=_pair(frame_scores,frame_scores); h=np.asarray(human_masks,bool)
    if h.shape != (20,len(p)): raise ValueError("human summary shape mismatch")
    span=float(np.ptp(p)); p=(p-p.min())/span if span else np.zeros_like(p)
    selected=summary_mask(p); den=h.sum(axis=1)+selected.sum()
    f1=np.divide(2*(h & selected).sum(axis=1),den,out=np.zeros(20),where=den!=0)
    return {"summary_f1":float(f1.mean()),"summary_rater_f1":f1.tolist(),
            "summary_prediction_rate":float(selected.mean())}
