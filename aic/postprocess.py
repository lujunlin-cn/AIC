"""Parameter-free temporal score post-processing.

The functions in this module operate on one video's dense temporal scores and
do not contain trainable parameters.  They are deliberately pure NumPy so the
same policy can be used by cached validation and raw-video inference.  Any
threshold or window size is an experiment parameter and must be selected on
TRAIN/DEV; this module never looks at labels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PostprocessConfig:
    """Versioned zero-parameter temporal policy.

    ``threshold`` applies after score normalization and smoothing.  For
    hysteresis it is the high threshold; ``low_threshold`` is used only when
    supplied.  Durations are measured in score timesteps (sampled frames),
    not seconds, so callers can record the sampling rate next to this config.
    """

    threshold: float = 0.4
    normalization: str = "none"
    smoothing: str = "none"
    smoothing_window: int = 1
    gaussian_sigma: float = 1.0
    ema_alpha: float = 0.5
    low_threshold: float | None = None
    gap: int = 0
    min_duration: int = 1
    max_duration: int | None = None
    shot_reset: bool = False

    def __post_init__(self) -> None:
        for name in ("threshold", "gaussian_sigma", "ema_alpha"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.low_threshold is not None and not math.isfinite(float(self.low_threshold)):
            raise ValueError("low_threshold must be finite")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be positive")
        if self.gap < 0 or self.min_duration < 1:
            raise ValueError("gap must be nonnegative and min_duration positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_scores(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def normalize_scores(scores: Sequence[float], mode: str = "none") -> np.ndarray:
    """Normalize a score sequence without using labels.

    ``zscore`` and ``robust`` return an unbounded score, while ``rank`` and
    ``percentile`` return [0, 1] values.  Constant sequences map to 0.5 for
    rank/percentile and zero for centered normalizations.
    """
    x = _validate_scores(scores)
    if mode in ("none", "identity"):
        return x.copy()
    if not x.size:
        return x.copy()
    if mode == "zscore":
        std = float(x.std())
        return (x - float(x.mean())) / std if std > 1e-12 else np.zeros_like(x)
    if mode in ("robust", "median_mad"):
        median = float(np.median(x)); mad = float(np.median(np.abs(x - median)))
        return (x - median) / (1.4826 * mad) if mad > 1e-12 else np.zeros_like(x)
    if mode in ("rank", "percentile"):
        if len(x) == 1 or float(x.max()) == float(x.min()):
            return np.full_like(x, 0.5)
        # Stable average ranks. Ties receive the same percentile.
        order = np.argsort(x, kind="mergesort")
        out = np.empty_like(x)
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and sorted_x[j] == sorted_x[i]:
                j += 1
            out[order[i:j]] = ((i + j - 1) / 2.0) / (len(x) - 1)
            i = j
        return out
    raise ValueError(f"unknown normalization={mode!r}")


def _segment_slices(n: int, boundaries: Iterable[int] | None) -> list[tuple[int, int]]:
    if not n:
        return []
    cuts = sorted({int(x) for x in (boundaries or []) if 0 < int(x) < n})
    edges = [0, *cuts, n]
    return list(zip(edges[:-1], edges[1:]))


def _smooth_one(x: np.ndarray, method: str, window: int, sigma: float, alpha: float) -> np.ndarray:
    if method in ("none", "identity") or len(x) < 2:
        return x.copy()
    if window < 1 or (window % 2 == 0 and method in ("median", "gaussian")):
        raise ValueError("smoothing_window must be positive and odd for median/gaussian")
    if method == "median":
        radius = window // 2
        padded = np.pad(x, (radius, radius), mode="edge")
        return np.stack([np.median(padded[i:i + window]) for i in range(len(x))])
    if method in ("gaussian", "normal"):
        if sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        radius = window // 2
        grid = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (grid / sigma) ** 2); kernel /= kernel.sum()
        padded = np.pad(x, (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")
    if method == "ema":
        if not 0 < alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        out = np.empty_like(x); out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out
    raise ValueError(f"unknown smoothing={method!r}")


def smooth_scores(scores: Sequence[float], method: str = "none", window: int = 1,
                  sigma: float = 1.0, alpha: float = 0.5,
                  shot_boundaries: Iterable[int] | None = None) -> np.ndarray:
    """Smooth each shot independently when boundaries are provided."""
    x = _validate_scores(scores); out = np.empty_like(x)
    for start, stop in _segment_slices(len(x), shot_boundaries):
        out[start:stop] = _smooth_one(x[start:stop], method, window, sigma, alpha)
    return out


def hysteresis_select(scores: Sequence[float], high: float, low: float | None = None,
                      shot_boundaries: Iterable[int] | None = None) -> np.ndarray:
    """Select contiguous regions with independent start and continuation cuts."""
    x = _validate_scores(scores)
    if low is None:
        low = high
    if low > high:
        raise ValueError("low threshold cannot exceed high threshold")
    selected = np.zeros(len(x), dtype=bool)
    for start, stop in _segment_slices(len(x), shot_boundaries):
        active = False
        for i in range(start, stop):
            if not active and x[i] >= high:
                active = True
            elif active and x[i] < low:
                active = False
            selected[i] = active
    return selected


def _fill_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 0 or not mask.any():
        return mask.copy()
    out = mask.copy(); n = len(out); i = 0
    while i < n:
        if out[i]: i += 1; continue
        start = i
        while i < n and not out[i]: i += 1
        left = start > 0 and out[start - 1]; right = i < n and out[i]
        if left and right and i - start <= gap:
            out[start:i] = True
    return out


def _filter_runs(mask: np.ndarray, min_duration: int, max_duration: int | None) -> np.ndarray:
    out = mask.copy(); n = len(out); i = 0
    while i < n:
        if not out[i]: i += 1; continue
        start = i
        while i < n and out[i]: i += 1
        stop = i; length = stop - start
        if length < max(1, min_duration):
            out[start:stop] = False
        elif max_duration is not None and max_duration > 0 and length > max_duration:
            # Keep deterministic chunks. The first chunk preserves the start
            # of the detected event; subsequent chunks are contiguous.
            out[start + max_duration:stop] = False
    return out


def select_temporal(scores: Sequence[float], config: PostprocessConfig | None = None,
                    *, shot_boundaries: Iterable[int] | None = None) -> np.ndarray:
    """Apply normalization, smoothing, thresholding and run filters."""
    cfg = config or PostprocessConfig()
    x = normalize_scores(scores, cfg.normalization)
    boundaries = shot_boundaries if cfg.shot_reset else None
    x = smooth_scores(x, cfg.smoothing, cfg.smoothing_window,
                      cfg.gaussian_sigma, cfg.ema_alpha, boundaries)
    mask = hysteresis_select(x, cfg.threshold, cfg.low_threshold, boundaries)
    if cfg.gap > 0:
        if cfg.shot_reset:
            pieces = []
            for s, e in _segment_slices(len(mask), boundaries):
                pieces.append(_fill_gaps(mask[s:e], cfg.gap))
            mask = np.concatenate(pieces) if pieces else mask
        else:
            mask = _fill_gaps(mask, cfg.gap)
    return _filter_runs(mask, cfg.min_duration, cfg.max_duration)


__all__ = ["PostprocessConfig", "normalize_scores", "smooth_scores",
           "hysteresis_select", "select_temporal"]
