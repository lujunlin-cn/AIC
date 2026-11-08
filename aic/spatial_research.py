"""Auditable public-data crop research. All geometry is in original pixels.

The 129 windows reproduce R46/R70's max-window search, but use integer AIC
deployment boxes. Neither the missing B3 head nor an oracle is a model.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from aic.scale_policy import atomic_json, canonical_sha, center, legal, route, sha

SEED = 20261108
PARENT_ELEMENTS = 8_999_771_458
PARAMETER_CAP = 9_000_000_000
LABELS = {"crop_iou", "mos", "subject_ranking"}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    tmp.replace(path)


def candidate_boxes(width, height, ratio, count=129):
    """Return K official [x,y,w] boxes and a deduplicated-action mask.

    K remains 129, including on a small image; repeated integer windows are
    masked rather than counted as independent actions. No GT chooses windows.
    """
    if count != 129 or min(width, height, *ratio) <= 0:
        raise ValueError("129 candidates and positive geometry required")
    w = min(width, math.floor(height * ratio[0] / ratio[1] + 1e-9))
    if w < 1:
        raise ValueError("target ratio cannot form an integer legal window")
    h = w * ratio[1] / ratio[0]
    axis = 0 if width / height > ratio[0] / ratio[1] else 1
    span = width - w if axis == 0 else math.floor(height - h + 1e-9)
    offsets = np.rint(np.linspace(0, span, count)).astype(int)
    boxes = [[int(v) if axis == 0 else 0, int(v) if axis == 1 else 0, w] for v in offsets]
    seen, mask = set(), []
    for box in boxes:
        key = tuple(box); mask.append(key not in seen); seen.add(key)
        if not legal(box, width, height, ratio):
            raise ValueError("integer candidate is not deployment legal")
    return np.asarray(boxes, dtype=np.int64), np.asarray(mask, dtype=bool)


def xywh(boxes, ratio):
    a = np.asarray(boxes, dtype=float)
    if a.shape[-1] == 4:
        return a
    if a.shape[-1] != 3:
        raise ValueError("boxes must be xyw or xywh")
    return np.concatenate((a, a[..., 2:3] * ratio[1] / ratio[0]), axis=-1)


def crop_iou(boxes, truth, ratio):
    """Mean across human annotators, never max over annotators."""
    a = xywh(np.atleast_2d(boxes), ratio)[:, None, :]
    b = np.atleast_2d(np.asarray(truth, dtype=float))[None, :, :]
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b[..., 2:] <= 0).any():
        raise ValueError("invalid crop geometry")
    lo = np.maximum(a[..., :2], b[..., :2]); hi = np.minimum(a[..., :2] + a[..., 2:], b[..., :2] + b[..., 2:])
    inter = np.maximum(0, hi - lo).prod(-1)
    return (inter / np.maximum(a[..., 2:].prod(-1) + b[..., 2:].prod(-1) - inter, 1e-12)).mean(-1)


def source_stat(values, sources, bootstrap=2000, seed=SEED):
    if len(values) != len(sources):
        raise ValueError("source/value length mismatch")
    groups = defaultdict(list)
    for s, value in zip(sources, values):
        if not np.isfinite(value):
            raise ValueError("nonfinite metric")
        groups[tuple(s) if isinstance(s, (tuple, list)) else s].append(float(value))
    v = np.array([np.mean(groups[k]) for k in sorted(groups)], float)
    if not len(v):
        return {"sources": 0, "mean": None, "ci95": None, "positive_fraction": None, "negative_fraction": None}
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(bootstrap)])
    return {"sources": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "p5": float(np.quantile(v, .05)), "p95": float(np.quantile(v, .95)),
            "ci95": np.quantile(means, [.025, .975]).tolist() if len(v) > 1 else None,
            "positive_fraction": float((v > 0).mean()), "negative_fraction": float((v < 0).mean()),
            "unit": "independent_dataset_source", "per_source": {str(k): float(np.mean(groups[k])) for k in sorted(groups)}}


def validate_manifest(rows, require_baseline=False):
    """No source/image-byte leakage, hidden-test inputs, ambiguous labels or fake P2."""
    sources, hashes, ids = {}, {}, set()
    for row in rows:
        if row.get("split") not in {"train", "dev", "confirm"}:
            raise ValueError("only public train/dev/confirm allowed; no Test labels")
        if row.get("dataset_source") not in {"LIVE_YT_VC", "GNMC", "GAICD", "SACD", "RetargetVid"}:
            raise ValueError("unknown or official dataset forbidden")
        if row.get("label_type") not in LABELS:
            raise ValueError("explicit label semantics required")
        key = (row["dataset_source"], row["source_id"])
        old = sources.setdefault(key, row["split"])
        if old != row["split"]:
            raise ValueError("source leakage")
        digest = row.get("image_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("real image SHA256 required")
        old = hashes.setdefault(digest, row["split"])
        if old != row["split"]:
            raise ValueError("image-byte leakage")
        identity = (key, row["image_id"], tuple(row["target_ratio"] or []), row.get("subject_id"))
        if identity in ids:
            raise ValueError("duplicate input")
        ids.add(identity)
        if require_baseline and row["label_type"] == "crop_iou":
            if (row.get("position_kind") != "P2_final_deployed" or row.get("qp_floor") != .6
                    or not row.get("p2_path_cache_sha256") or not row.get("p2_base_identity_sha256")):
                raise ValueError("P2 actual deployed baseline provenance required; P1 is not P2")
            if not legal(row["p2_bbox_xyw"], row["image_width"], row["image_height"], row["target_ratio"]):
                raise ValueError("illegal P2 baseline")
    return {"rows": len(rows), "sources": len(sources), "unique_images": len(hashes), "no_leakage": True}


def preference_pairs(scores, mask, min_delta=.25):
    """GAICD MOS pairs only; caller retains label_type=mos, same image."""
    scores = np.asarray(scores, float); mask = np.asarray(mask, bool)
    if min_delta <= 0 or not np.isfinite(scores[mask]).all():
        raise ValueError("positive preregistered tie margin required")
    i, j = np.triu_indices(len(scores), 1)
    valid = mask[i] & mask[j] & (np.abs(scores[i] - scores[j]) > min_delta)
    i, j = i[valid], j[valid]
    hi = np.where(scores[i] > scores[j], i, j); lo = np.where(scores[i] > scores[j], j, i)
    return np.stack((hi, lo), -1)


def preference_field(boxes, scores, ratio, top_fraction=.2, overlap=.75):
    """E5 train-only descriptive peaks; never manufactures ranked pairs."""
    boxes = np.asarray(boxes); scores = np.asarray(scores, float)
    threshold = float(np.quantile(scores, 1 - top_fraction))
    remaining = set(np.flatnonzero(scores >= threshold).tolist()); clusters = []
    while remaining:
        peak = max(remaining, key=lambda i: (scores[i], -i))
        neighborhood = crop_iou(boxes, xywh(boxes[peak], ratio), ratio) >= overlap
        cluster = sorted(i for i in remaining if neighborhood[i]); remaining.difference_update(cluster)
        clusters.append({"peak_candidate": int(peak), "candidates": len(cluster), "peak_quality": float(scores[peak])})
    return {"threshold_from_this_train_image": threshold, "peak_count": len(clusters), "peaks": clusters,
            "supervision": "descriptive_discrete_human_MOS_field_not_continuous_ground_truth"}


def shot_dp(utilities, boxes, shot_ids, times, penalty=.02):
    """E8 learned-utility DP with shot resets, no GT/IoU in optimization.

    The calling evaluator compares resulting *absolute* deployed IoU to P2.
    This is a registered optional postprocessor, not the P2 QP itself.
    """
    u = np.asarray(utilities, float); b = np.asarray(boxes, float)
    if len(u) != len(shot_ids) or len(times) != len(u) or u.ndim != 2 or b.shape != (*u.shape, 3):
        raise ValueError("invalid trajectory tensors")
    if not np.isfinite(u).all() or penalty < 0 or len(u) == 0:
        raise ValueError("invalid utilities/penalty")
    chosen = np.zeros(len(u), dtype=int); first = 0
    while first < len(u):
        end = first + 1
        while end < len(u) and shot_ids[end] == shot_ids[first]:
            if times[end] <= times[end - 1]:
                raise ValueError("non-increasing shot timestamps")
            end += 1
        value = u[first].copy(); back = []
        for t in range(first + 1, end):
            delta = np.abs((b[t - 1, :, 0] + b[t - 1, :, 2]/2)[:, None] - (b[t, :, 0] + b[t, :, 2]/2)[None, :])
            delta += np.abs(b[t - 1, :, 1][:, None] - b[t, :, 1][None, :])
            delta /= max(1., float(b[t, :, 2].max()))
            score = value[:, None] - penalty * delta
            previous = score.argmax(0); value = score[previous, np.arange(u.shape[1])] + u[t]
            back.append(previous)
        chosen[end - 1] = int(value.argmax())
        for t in range(end - 2, first - 1, -1):
            chosen[t] = back[t - first][chosen[t + 1]]
        first = end
    return chosen


def promotion_gate(report, parameters, deployment):
    reasons = []
    if parameters > PARAMETER_CAP:
        reasons.append("complete inference parameter cap exceeded")
    video_pools = []
    for key, metric in report.items():
        if metric.get("media_kind") != "video":
            continue
        if metric.get("split") != "confirm" or metric.get("evaluation_geometry") != "original_P2_video_path":
            continue
        ci = metric.get("policy_minus_P2", {}).get("ci95")
        if ci and ci[0] > 0:
            video_pools.append(key)
    if len(video_pools) < 2:
        reasons.append("two positive independent heldout video pools not demonstrated")
    if not deployment.get("source_pools_disjoint_verified"):
        reasons.append("independent video-pool source separation unverified")
    if not deployment.get("native_A_and_B_noninferiority_passed"):
        reasons.append("native A/B deployment evidence incomplete")
    for name in ("exact_temporal_identity", "original_qp_replayed", "integer_geometry_valid", "inputs_and_code_locked"):
        if not deployment.get(name):
            reasons.append(name + " not passed")
    return {"passed": not reasons, "reasons": reasons, "positive_video_pools": video_pools,
            "official_F1": None, "promoted": False, "official_feedback_required": True}
