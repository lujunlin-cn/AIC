"""Deterministic fixed-center scale geometry for the AIC [x, y, w] contract."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _image_and_ratio(
    image_width: int, image_height: int, target_ratio: Sequence[int]
) -> tuple[int, int, int, int]:
    if (not isinstance(image_width, int) or isinstance(image_width, bool) or image_width <= 0
            or not isinstance(image_height, int) or isinstance(image_height, bool) or image_height <= 0):
        raise ValueError("image dimensions must be positive integers")
    if (len(target_ratio) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                                      for v in target_ratio)):
        raise ValueError("target ratio must be two positive integers")
    return image_width, image_height, target_ratio[0], target_ratio[1]


def maximum_window(
    image_width: int, image_height: int, target_ratio: Sequence[int]
) -> tuple[float, float]:
    """Return the largest exact-ratio window that fits the image."""
    width, height, ratio_w, ratio_h = _image_and_ratio(image_width, image_height, target_ratio)
    crop_width = min(float(width), height * ratio_w / ratio_h)
    return crop_width, crop_width * ratio_h / ratio_w


def scale_crop_from_center(
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
    scale: float,
) -> tuple[int, int, int, int]:
    """Project a fixed pixel center and scale to integer ``(x, y, w, h)``.

    AIC submissions carry only ``[x, y, w]``; evaluation derives the exact
    height from ``target_ratio``. The returned integer height is for diagnostics.
    At image boundaries, the legal-window clamp can translate the crop center.
    """
    width, height, ratio_w, ratio_h = _image_and_ratio(image_width, image_height, target_ratio)
    if (len(center_xy) != 2 or not all(_finite_number(v) for v in center_xy)
            or not 0 <= center_xy[0] <= width or not 0 <= center_xy[1] <= height):
        raise ValueError("center_xy must be finite pixel coordinates inside the image")
    if not _finite_number(scale) or not 0 < scale <= 1:
        raise ValueError("scale must be finite and in (0, 1]")

    maximum_width, _ = maximum_window(width, height, target_ratio)
    crop_width = max(1, math.floor(maximum_width * scale + 1e-9))
    crop_height = crop_width * ratio_h / ratio_w
    if crop_height > height + 1e-9:
        raise ValueError("scale projection exceeds image height")

    x = round(center_xy[0] - crop_width / 2)
    y = round(center_xy[1] - crop_height / 2)
    x = min(max(x, 0), width - crop_width)
    y = min(max(y, 0), math.floor(height - crop_height + 1e-9))
    return int(x), int(y), int(crop_width), int(round(crop_height))


def deployed_box_xyxy(
    crop_xywh: Sequence[float], target_ratio: Sequence[int]
) -> tuple[float, float, float, float]:
    """Return the exact evaluator rectangle implied by AIC ``[x, y, w]``."""
    if len(crop_xywh) != 3 or not all(_finite_number(v) for v in crop_xywh):
        raise ValueError("AIC crop must be finite [x, y, w]")
    if len(target_ratio) != 2 or any(not _finite_number(v) or v <= 0 for v in target_ratio):
        raise ValueError("target ratio must contain two positive values")
    x, y, width = crop_xywh
    if width <= 0:
        raise ValueError("crop width must be positive")
    height = width * target_ratio[1] / target_ratio[0]
    return float(x), float(y), float(x + width), float(y + height)


def crop_box_xyxy(
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    x, y, width, _ = scale_crop_from_center(
        image_width, image_height, target_ratio, center_xy, scale
    )
    return deployed_box_xyxy((x, y, width), target_ratio)


def scale_crop_xywh_exact(
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    x, y, width, _ = scale_crop_from_center(
        image_width, image_height, target_ratio, center_xy, scale
    )
    return float(x), float(y), float(width), width * target_ratio[1] / target_ratio[0]


def box_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4 or not all(_finite_number(v) for v in (*a, *b)):
        raise ValueError("boxes must contain finite xywh values")
    if a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        raise ValueError("box width and height must be positive")
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    if union <= 0:
        raise ValueError("degenerate box union")
    return intersection / union


def fit_scale_bins_from_rows(rows: Iterable[dict], count: int) -> list[float]:
    """Fit ordered quantile bins from training-only human crop scale ratios."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError("count must be an integer >= 2")
    values: list[float] = []
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("scale bins may only be fitted from train rows")
        value = row.get("scale_ratio")
        if not _finite_number(value) or value <= 0:
            raise ValueError("train scale_ratio must be finite and positive")
        values.append(min(float(value), 1.0))
    if not values:
        raise ValueError("no training scale ratios")
    values.sort()
    candidates = []
    quantile_count = count - 1
    if quantile_count == 1:
        candidates.append(values[0])
    else:
        for index in range(quantile_count):
            position = index * (len(values) - 1) / (count - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            fraction = position - lower
            candidates.append(values[lower] * (1 - fraction) + values[upper] * fraction)
    candidates.append(1.0)
    bins: list[float] = []
    for value in sorted(candidates):
        value = min(1.0, max(value, 1e-9))
        if not bins or abs(value - bins[-1]) > 1e-8:
            bins.append(value)
    return bins


def transpose_xywh(
    box_xywh: Sequence[float], image_width: int, image_height: int
) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
    """Transpose image axes and an xywh box; returned dimensions are swapped."""
    if len(box_xywh) != 4 or not all(_finite_number(v) for v in box_xywh):
        raise ValueError("box must be finite xywh")
    x, y, width, height = box_xywh
    if (width <= 0 or height <= 0 or x < 0 or y < 0
            or x + width > image_width + 1e-9 or y + height > image_height + 1e-9):
        raise ValueError("box must be positive and inside the image")
    return (y, x, height, width), (image_height, image_width)


def map_box_to_scale_id(
    box_xywh: Sequence[float],
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
    scale_bins: Sequence[float],
    tolerance_px: float = 1.0,
) -> int | None:
    """Match only a human crop that equals a fixed-center deployed scale action."""
    if not _finite_number(tolerance_px) or tolerance_px < 0:
        raise ValueError("tolerance_px must be finite and nonnegative")
    x, y, width, height = box_xywh
    if abs(height - width * target_ratio[1] / target_ratio[0]) > tolerance_px:
        return None
    best: tuple[float, int] | None = None
    for scale_id, scale in enumerate(scale_bins):
        projected = scale_crop_from_center(
            image_width, image_height, target_ratio, center_xy, float(scale)
        )
        delta = sum(abs(float(actual) - expected) for actual, expected in zip(
            (x, y, width, height), projected
        ))
        if max(abs(float(actual) - expected) for actual, expected in zip(
            (x, y, width, height), projected
        )) <= tolerance_px and (best is None or delta < best[0]):
            best = (delta, scale_id)
    return None if best is None else best[1]
