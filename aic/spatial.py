"""Weight-free legal crop candidates and shot-aware path smoothing."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .contract import ContractError, center_crop, parse_ratio


def legal_widths(width: int, height: int, target_ratio: Sequence[float],
                 fractions: Sequence[float] = (1.0, .85, .7, .55)) -> list[float]:
    rw, rh = parse_ratio(target_ratio)
    maximum = min(float(width), float(height) * rw / rh)
    if maximum <= 0:
        raise ContractError("no positive legal crop width")
    values = sorted({max(1.0, min(maximum, maximum * float(f))) for f in fractions}, reverse=True)
    return values


def place_crop(width: int, height: int, target_ratio: Sequence[float],
               center_x: float, center_y: float, crop_width: float | None = None) -> list[float]:
    rw, rh = parse_ratio(target_ratio)
    if not all(math.isfinite(float(v)) for v in (center_x, center_y)):
        raise ContractError("crop center must be finite")
    w = legal_widths(width, height, target_ratio, (1.0,))[0] if crop_width is None else float(crop_width)
    if not math.isfinite(w) or w <= 0 or w > legal_widths(width, height, target_ratio, (1.0,))[0] + 1e-6:
        raise ContractError("crop width is outside the legal domain")
    h = w * rh / rw
    x = min(max(float(center_x) - w / 2, 0.0), float(width) - w)
    y = min(max(float(center_y) - h / 2, 0.0), float(height) - h)
    return [x, y, w]


def gradient_saliency_center(rgb: np.ndarray) -> tuple[float, float, float]:
    """Return normalized gradient-weighted center and a confidence proxy."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("Expected RGB HWC image")
    gray = rgb.astype(np.float32).mean(axis=-1)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    saliency = np.sqrt(gx * gx + gy * gy) + 1e-6
    mass = float(saliency.sum())
    yy, xx = np.mgrid[:gray.shape[0], :gray.shape[1]]
    cx = float((saliency * xx).sum() / mass / max(1, gray.shape[1] - 1))
    cy = float((saliency * yy).sum() / mass / max(1, gray.shape[0] - 1))
    confidence = float(np.clip(saliency.std() / (saliency.mean() + 1e-6), 0, 10) / 10)
    return cx, cy, confidence


def smooth_centers(centers: Iterable[Sequence[float]], alpha: float = .25,
                   reset: Iterable[bool] | None = None) -> np.ndarray:
    """EMA in normalized coordinates; reset at shot cuts and invalid samples."""
    if not 0 < alpha <= 1 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite in (0,1]")
    values = np.asarray(list(centers), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("centers must be [T,2]")
    resets = np.zeros(len(values), dtype=bool) if reset is None else np.asarray(list(reset), dtype=bool)
    if len(resets) != len(values):
        raise ValueError("reset length mismatch")
    result = np.zeros_like(values)
    state = values[0].copy() if len(values) else np.zeros(2, dtype=np.float32)
    for i, value in enumerate(values):
        if i == 0 or resets[i] or not np.isfinite(value).all():
            state = value if np.isfinite(value).all() else np.array([.5, .5], dtype=np.float32)
        else:
            state = (1 - alpha) * state + alpha * value
        result[i] = np.clip(state, 0, 1)
    return result


def saliency_crop(rgb: np.ndarray, target_ratio: Sequence[float], alpha: float = .25) -> list[float]:
    """Single-frame spatial candidate; caller handles temporal association."""
    height, width = rgb.shape[:2]
    cx, cy, _ = gradient_saliency_center(rgb)
    # Keep the largest legal window for a stable, conservative first candidate.
    return place_crop(width, height, target_ratio, cx * width, cy * height)


def subject_crop(rgb: np.ndarray, target_ratio: Sequence[float],
                 confidence_floor: float = .08) -> list[float]:
    """Return a conservative subject-centred crop candidate.

    No detector is bundled in the <=100 MB baseline.  This policy uses the
    gradient saliency centre as a subject proxy and blends it toward the
    geometric centre when saliency confidence is weak.  Keeping this policy
    explicit makes raw-video inference reproducible and prevents callers from
    silently treating the old fixed centre crop as a learned subject box.
    """
    height, width = rgb.shape[:2]
    cx, cy, confidence = gradient_saliency_center(rgb)
    weight = float(np.clip((confidence - confidence_floor) /
                           max(1e-6, 1.0 - confidence_floor), 0.0, 1.0))
    cx = .5 + weight * (cx - .5)
    cy = .5 + weight * (cy - .5)
    return place_crop(width, height, target_ratio, cx * width, cy * height)
