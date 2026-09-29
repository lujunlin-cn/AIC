"""Window-scorer targets: legal target-ratio windows vs human crop boxes.

Inputs: source size (W, H) in coded pixels, target ratio (w:h), candidate
windows, and the human boxes of one frame (one row per annotator).  Output:
per-annotator IoU, aggregates over annotators and explicit valid masks.

Everything that changes a number is a named, versioned option:

``COORD_VERSION = "xywh_halfopen_v1"``
    A box is ``[x, x+w) x [y, y+h)`` in coded source pixels (no rotation).
    RetargetVid / LIVE-YT-VC publish ``left, top, right, bottom``; they map to
    ``x=l, y=t, w=r-l, h=b-t`` with no clipping.  (Their widths equal
    ``r-l`` exactly, e.g. 120x360 for 1:3 on 640x360, so r/b act as exclusive.)

IoU conventions (``iou=``)
    ``"halfopen"``         continuous areas ``w*h``; the default.
    ``"inclusive_plus1"``  boxes rounded to integer ``l,t,r,b`` (``r = l + w``),
                           negatives clamped to 0, area ``(r-l+1)(b-t+1)``.
                           Same numbers as RetargetVid's upstream
                           ``bb_intersection_over_union`` and as the teacher
                           scripts (teacher_diag_eval.py, t5_crophead_probe.py).

GT boundary handling (``gt_boundary=``), applied to a copy, never to raw data
    ``"none"``         as published (LIVE has right=1921 on 92 boxes).
    ``"clip_v1"``      clip to the image extent ``[0,W] x [0,H]``.
    ``"clip_exp_v1"``  clip l,r to ``[0, W-1]`` and t,b to ``[0, H-1]``; this is
                       what t5_crophead_probe.py does for LIVE before its +1 IoU.

Human and derived labels are never mixed: ``gt_source`` is one value per call
and is echoed in the output.  ``gt_is_max_window`` records whether the human
box itself is a max-size target-ratio window (RetargetVid: yes; LIVE: no,
its boxes are variable-size 9:16 regions, so a max window can never reach IoU
1 with them; ``iou_ceiling`` reports the best a legal max window can reach).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

COORD_VERSION = "xywh_halfopen_v1"
SCORER_VERSION = "window_scorer_v1"
IOU_CONVENTIONS = ("halfopen", "inclusive_plus1")
GT_BOUNDARY = ("none", "clip_v1", "clip_exp_v1")
GT_SOURCES = ("human", "derived")


def parse_ratio(ratio) -> tuple[float, float]:
    """``[w, h]`` / ``(w, h)`` / ``"1-3"`` / ``"9:16"`` -> (rw, rh)."""
    if isinstance(ratio, str):
        a, b = ratio.replace(":", "-").split("-")
        ratio = (float(a), float(b))
    rw, rh = float(ratio[0]), float(ratio[1])
    if not (rw > 0 and rh > 0):
        raise ValueError(f"bad ratio {ratio!r}")
    return rw, rh


def max_window(W: float, H: float, ratio) -> tuple[float, float, Optional[int]]:
    """Largest window of the target ratio inside (W, H) and its free axis.

    Same geometry as ``aic.max_window_path.geometry``: axis 0 = window slides
    along x (full height), 1 = along y (full width), None = fills the frame.
    """
    rw, rh = parse_ratio(ratio)
    w = min(float(W), float(H) * rw / rh)
    h = w * rh / rw
    axis = 0 if w < W - 1e-9 else (1 if h < H - 1e-9 else None)
    return w, h, axis


def legal_candidates(W: float, H: float, ratio, n: int = 65,
                     extra_offsets: Iterable[float] = ()) -> dict:
    """Max-window candidates: ``n`` evenly spaced offsets on the free axis + extras.

    Offsets are top-left positions along the free axis in pixels
    (``np.linspace(0, L - s, n)``, the grid used by T5_CROPHEAD_V3).
    """
    w, h, axis = max_window(W, H, ratio)
    if axis is None:
        offs = np.zeros(1)
    else:
        L, s = (W, w) if axis == 0 else (H, h)
        offs = np.concatenate([np.linspace(0.0, L - s, n), np.asarray(list(extra_offsets), float)])
        offs = np.clip(offs, 0.0, L - s)
    boxes = np.zeros((len(offs), 4), np.float64)
    boxes[:, 2], boxes[:, 3] = w, h
    if axis == 0:
        boxes[:, 0] = offs
    elif axis == 1:
        boxes[:, 1] = offs
    return {"offsets": offs, "boxes_xywh": boxes, "axis": axis, "window_wh": (w, h)}


def check_candidates(W: float, H: float, ratio, boxes_xywh: np.ndarray,
                     px_tol: float = 1.0) -> dict:
    """Legality of arbitrary candidates: inside the image and of the target ratio.

    ``px_tol`` absorbs integer rounding (a legal 213-px-high 3:1 window on
    640 px is 213.33 before rounding).  ``is_max_window`` additionally
    requires the size of the largest legal window.
    """
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    rw, rh = parse_ratio(ratio)
    x, y, w, h = b.T
    inside = (x >= -px_tol) & (y >= -px_tol) & (x + w <= W + px_tol) & (y + h <= H + px_tol) & (w > 0) & (h > 0)
    ratio_ok = np.abs(w - h * rw / rh) <= px_tol + 1e-9
    mw, mh, _ = max_window(W, H, ratio)
    is_max = (np.abs(w - mw) <= px_tol) & (np.abs(h - mh) <= px_tol)
    return {"legal": inside & ratio_ok, "inside": inside, "ratio_ok": ratio_ok, "is_max_window": is_max & inside & ratio_ok}


def _ltrb(xywh: np.ndarray) -> np.ndarray:
    b = np.asarray(xywh, np.float64).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])


def apply_gt_boundary(gt_xywh: np.ndarray, W: float, H: float, mode: str) -> np.ndarray:
    """Return boundary-corrected GT (copy); raw input is never modified."""
    if mode not in GT_BOUNDARY:
        raise ValueError(f"gt_boundary must be one of {GT_BOUNDARY}")
    lt = _ltrb(gt_xywh)
    if mode == "clip_v1":
        lt[:, [0, 2]] = np.clip(lt[:, [0, 2]], 0, W)
        lt[:, [1, 3]] = np.clip(lt[:, [1, 3]], 0, H)
    elif mode == "clip_exp_v1":
        lt[:, [0, 2]] = np.clip(lt[:, [0, 2]], 0, W - 1)
        lt[:, [1, 3]] = np.clip(lt[:, [1, 3]], 0, H - 1)
    return np.column_stack([lt[:, 0], lt[:, 1], lt[:, 2] - lt[:, 0], lt[:, 3] - lt[:, 1]])


def iou_matrix(a_xywh: np.ndarray, b_xywh: np.ndarray, iou: str = "halfopen") -> tuple[np.ndarray, np.ndarray]:
    """IoU [len(a), len(b)] and intersection / area(b) (share of b kept by a)."""
    if iou not in IOU_CONVENTIONS:
        raise ValueError(f"iou must be one of {IOU_CONVENTIONS}")
    A, B = _ltrb(a_xywh), _ltrb(b_xywh)
    if iou == "inclusive_plus1":
        # teacher scripts: np.maximum(np.rint([x, y, x+w, y+h]).astype(int), 0)
        A = np.maximum(np.rint(A), 0)
        B = np.maximum(np.rint(B), 0)
        plus = 1.0
    else:
        plus = 0.0
    lo = np.maximum(A[:, None, :2], B[None, :, :2])
    hi = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(hi - lo + plus, 0, None).prod(-1)
    area_a = ((A[:, 2:] - A[:, :2]) + plus).prod(-1)
    area_b = ((B[:, 2:] - B[:, :2]) + plus).prod(-1)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter / union, np.nan)
        kept = np.where(area_b[None, :] > 0, inter / area_b[None, :], np.nan)
    return out, kept


def iou_ceiling(W: float, H: float, ratio, gt_xywh: np.ndarray, iou: str = "halfopen") -> np.ndarray:
    """Best IoU any legal max window reaches per GT box (integer-pixel offsets)."""
    w, h, axis = max_window(W, H, ratio)
    L, s = ((W, w) if axis == 0 else (H, h)) if axis is not None else (0, 0)
    n = int(np.floor(L - s)) + 1 if axis is not None else 1
    offs = np.arange(n, dtype=np.float64) if axis is not None else np.zeros(1)
    if axis is not None and offs[-1] < L - s - 1e-9:
        offs = np.append(offs, L - s)
    boxes = np.zeros((len(offs), 4))
    boxes[:, 2], boxes[:, 3] = w, h
    if axis == 0:
        boxes[:, 0] = offs
    elif axis == 1:
        boxes[:, 1] = offs
    m, _ = iou_matrix(boxes, gt_xywh, iou)
    return np.nanmax(m, axis=0) if len(m) else np.full(len(gt_xywh), np.nan)


def score_candidates(W: float, H: float, target_ratio, candidates_xywh: np.ndarray,
                     gt_xywh: np.ndarray, gt_valid: Optional[Sequence[bool]] = None, *,
                     gt_source: str, gt_is_max_window: bool, iou: str = "halfopen",
                     gt_boundary: str = "none", annotators: Optional[Sequence[str]] = None,
                     require_legal: bool = True, with_ceiling: bool = False) -> dict:
    """Score C candidate windows against A human (or derived) boxes of one frame.

    Returns
      iou[C, A]      IoU (NaN where ``valid`` is False)
      valid[C, A]    candidate legal (if ``require_legal``) and annotator box valid
      gt_kept[C, A]  |cand ∩ gt| / |gt|: share of the human box inside the window
      iou_mean/min/max[C], n_valid[C]   aggregates over valid annotators (NaN if none)
      cand_legal[C], cand_is_max_window[C], gt_valid[A]
      iou_ceiling[A] (optional) best IoU reachable by any legal max window
    """
    if gt_source not in GT_SOURCES:
        raise ValueError(f"gt_source must be one of {GT_SOURCES}; human and derived boxes are scored separately")
    cand = np.asarray(candidates_xywh, np.float64).reshape(-1, 4)
    gt_raw = np.asarray(gt_xywh, np.float64).reshape(-1, 4)
    gv = np.ones(len(gt_raw), bool) if gt_valid is None else np.asarray(gt_valid, bool).copy()
    gv &= np.isfinite(gt_raw).all(1) & (gt_raw[:, 2] > 0) & (gt_raw[:, 3] > 0)
    gt = apply_gt_boundary(np.nan_to_num(gt_raw), W, H, gt_boundary)
    chk = check_candidates(W, H, target_ratio, cand)
    m, kept = iou_matrix(cand, gt, iou)
    legal = chk["legal"] if require_legal else np.ones(len(cand), bool)
    valid = legal[:, None] & gv[None, :]
    m = np.where(valid, m, np.nan)
    kept = np.where(valid, kept, np.nan)
    n_valid = valid.sum(1)
    safe = np.where(valid, m, 0.0)
    mean = np.where(n_valid > 0, safe.sum(1) / np.maximum(n_valid, 1), np.nan)
    mn = np.where(n_valid > 0, np.where(valid, m, np.inf).min(1), np.nan)
    mx = np.where(n_valid > 0, np.where(valid, m, -np.inf).max(1), np.nan)
    out = {"iou": m.astype(np.float32), "valid": valid, "gt_kept": kept.astype(np.float32),
           "iou_mean": mean.astype(np.float32), "iou_min": mn.astype(np.float32),
           "iou_max": mx.astype(np.float32), "n_valid": n_valid,
           "cand_legal": chk["legal"], "cand_is_max_window": chk["is_max_window"], "gt_valid": gv,
           "gt_source": gt_source, "gt_is_max_window": bool(gt_is_max_window),
           "annotators": list(annotators) if annotators is not None else None,
           "conventions": {"coord": COORD_VERSION, "iou": iou, "gt_boundary": gt_boundary,
                           "scorer": SCORER_VERSION}}
    if with_ceiling:
        out["iou_ceiling"] = np.where(gv, iou_ceiling(W, H, target_ratio, gt, iou), np.nan).astype(np.float32)
    return out
