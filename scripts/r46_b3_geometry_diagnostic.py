#!/usr/bin/env python3
"""Audit the interaction between R26's temporal mask and B3 crop geometry.

This is a diagnostic only.  It never uses official labels and never treats
box motion as spatial quality.  It measures whether the center-85 temporal
parent samples a different B3 box trajectory regime than its omitted tails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def one(row: dict) -> dict:
    boxes = np.asarray([x["bboxes"] for x in row["predictions"]], dtype=np.float64)
    n = len(boxes)
    k = int(round(0.85 * n))
    lo = (n - k) // 2
    hi = lo + k
    jump = np.linalg.norm(np.diff(boxes, axis=0), axis=1) if n > 1 else np.zeros(0)
    jump = np.r_[0.0, jump]
    accel = np.linalg.norm(np.diff(boxes, n=2, axis=0), axis=1) if n > 2 else np.zeros(0)
    accel = np.r_[0.0, 0.0, accel]
    selected = np.arange(lo, hi, dtype=np.int64)
    tails = np.r_[np.arange(0, lo), np.arange(hi, n)]
    return {
        "video_id": str(row["video_id"]),
        "target_ratio": row.get("targetRatioWH"),
        "n": n,
        "k": k,
        "parent_lo": lo,
        "parent_hi": hi,
        "unique_boxes": int(len(np.unique(boxes, axis=0))),
        "selected_mean_jump": float(jump[selected].mean()),
        "tail_mean_jump": float(jump[tails].mean()) if len(tails) else 0.0,
        "selected_nonzero_jump": float(np.mean(jump[selected] > 0)),
        "tail_nonzero_jump": float(np.mean(jump[tails] > 0)) if len(tails) else 0.0,
        "selected_mean_accel": float(accel[selected].mean()),
        "tail_mean_accel": float(accel[tails].mean()) if len(tails) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.input.read_text().splitlines() if x.strip()]
    per_video = [one(row) for row in rows]
    def mean(key, xs=per_video):
        return float(np.mean([x[key] for x in xs])) if xs else 0.0
    summary = {
        "status": "DIAGNOSTIC_ONLY",
        "official_labels_used": False,
        "interpretation": "box trajectory is not an IoU or highlight-quality label",
        "videos": len(per_video),
        "frames": int(sum(x["n"] for x in per_video)),
        "mean": {k: mean(k) for k in (
            "selected_mean_jump", "tail_mean_jump", "selected_nonzero_jump",
            "tail_nonzero_jump", "selected_mean_accel", "tail_mean_accel")},
        "by_ratio": {},
        "per_video": per_video,
    }
    for ratio in sorted({str(x["target_ratio"]) for x in per_video}):
        xs = [x for x in per_video if str(x["target_ratio"]) == ratio]
        summary["by_ratio"][ratio] = {
            "videos": len(xs),
            "selected_mean_jump": mean("selected_mean_jump", xs),
            "tail_mean_jump": mean("tail_mean_jump", xs),
            "selected_nonzero_jump": mean("selected_nonzero_jump", xs),
            "tail_nonzero_jump": mean("tail_nonzero_jump", xs),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "videos", "frames", "mean", "by_ratio")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
