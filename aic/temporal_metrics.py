"""Threshold-independent temporal ranking and TVSum-style summary metrics.

These metrics are for TVSum summary-importance proxy research only.  They do
not implement the AIC joint temporal/spatial F_video.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return out


def spearman(pred: Sequence[float], target: Sequence[float]) -> float:
    p, t = _rank(np.asarray(pred, float)), _rank(np.asarray(target, float))
    p -= p.mean(); t -= t.mean()
    d = float(np.sqrt(np.dot(p, p) * np.dot(t, t)))
    return float(np.dot(p, t) / d) if d else 0.0


def kendall_tau(pred: Sequence[float], target: Sequence[float]) -> float:
    p, t = np.asarray(pred, float), np.asarray(target, float)
    concordant = discordant = 0
    for i in range(len(p)):
        dp = p[i + 1:] - p[i]
        dt = t[i + 1:] - t[i]
        concordant += int(np.sum(dp * dt > 0))
        discordant += int(np.sum(dp * dt < 0))
    den = concordant + discordant
    return float((concordant - discordant) / den) if den else 0.0


def ndcg(pred: Sequence[float], target: Sequence[float], k: int | None = None) -> float:
    p, t = np.asarray(pred, float), np.asarray(target, float)
    if k is None: k = len(p)
    k = max(1, min(int(k), len(p)))
    order = np.argsort(-p, kind="mergesort")[:k]
    ideal = np.argsort(-t, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(np.maximum(t[order], 0.0) * discounts))
    idcg = float(np.sum(np.maximum(t[ideal], 0.0) * discounts))
    return dcg / idcg if idcg else 0.0


def budget_f1(pred_mask: Sequence[bool], target: Sequence[float], budget: float = .15) -> float:
    """Deterministic frame-budget F1 proxy; summary evaluator remains separate."""
    p = np.asarray(pred_mask, bool)
    y = np.asarray(target, float)
    n = len(y); keep = max(1, int(round(n * budget)))
    chosen = np.zeros(n, bool)
    chosen[np.argsort(-p.astype(float), kind="mergesort")[:keep]] = True
    truth = y >= 0.5
    tp = int(np.sum(chosen & truth)); fp = int(np.sum(chosen & ~truth)); fn = int(np.sum(~chosen & truth))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def ranking_report(pred: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    p, t = np.asarray(pred, float), np.asarray(target, float)
    k = max(1, int(round(len(t) * .15)))
    return {"spearman": spearman(p, t), "kendall_tau": kendall_tau(p, t),
            "ndcg": ndcg(p, t), "ndcg_at_15pct": ndcg(p, t, k),
            "top15_ap": float(np.mean(t[np.argsort(-p, kind="mergesort")[:k]]))}
