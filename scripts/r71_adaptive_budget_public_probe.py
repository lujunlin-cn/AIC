#!/usr/bin/env python3
"""Small human-label diagnostic for the R71 adaptive-budget rule.

This is deliberately separate from the package builder.  It uses the ten
manually annotated public windows and the frozen TimeLens spans already used
by the R47/R54 public probe.  It is a mechanism check, not an AIC score
estimate: the labels are a single-project annotation set and the windows are
not the official test videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, proxy_f, select  # noqa: E402


def union(intervals: list[list[float]]) -> float:
    parts = sorted((float(a), float(b)) for a, b in intervals if float(b) >= float(a))
    if not parts:
        return 0.0
    total = 0.0
    a, b = parts[0]
    for x, y in parts[1:]:
        if x <= b:
            b = max(b, y)
        else:
            total += b - a
            a, b = x, y
    return total + b - a


def rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort") / max(len(x) - 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--proxy", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    by_video: dict[str, list[dict[str, str]]] = {}
    with args.labels.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_video.setdefault(row["video_id"], []).append(row)
    source_rows = json.loads(args.proxy.read_text(encoding="utf-8"))["source"]
    spans = {}
    for row in source_rows:
        spans.setdefault(str(row["video_id"]), row["span_seconds"])
    missing = sorted(set(by_video) - set(spans))
    if missing:
        raise ValueError(f"missing public TimeLens spans: {missing}")

    occupancy = {vid: union(spans[vid]) / 30.0 for vid in by_video}
    q50 = float(np.median(np.asarray(list(occupancy.values()), dtype=float)))
    rows = []
    for vid, items in sorted(by_video.items()):
        items = sorted(items, key=lambda r: float(r["time_sec"]))
        times = np.asarray([float(r["time_sec"]) for r in items], dtype=float)
        y = np.asarray([GRADE[r["generic_highlight"]] for r in items], dtype=float)
        inside = np.asarray([any(a <= t <= b for a, b in spans[vid]) for t in times], dtype=float)
        center = 1.0 - np.abs(times / 29.0 - 0.5) * 2.0
        score = rank01(center) + 0.20 * (rank01(inside) - 0.5)
        base = select(score, round(0.85 * len(items)))
        fraction = float(np.clip(0.75 + 0.10 * occupancy[vid] / q50, 0.70, 0.85))
        adaptive = select(score, round(fraction * len(items)))
        rows.append({
            "video_id": vid,
            "occupancy": occupancy[vid],
            "adaptive_fraction": fraction,
            "adaptive_k": int(len(adaptive)),
            "fixed_k": int(len(base)),
            "soft_fixed": proxy_f(base, y, True),
            "soft_adaptive": proxy_f(adaptive, y, True),
            "binary_fixed": proxy_f(base, y, False),
            "binary_adaptive": proxy_f(adaptive, y, False),
        })
    soft_delta = float(np.mean([r["soft_adaptive"] - r["soft_fixed"] for r in rows]))
    binary_delta = float(np.mean([r["binary_adaptive"] - r["binary_fixed"] for r in rows]))
    output = {
        "experiment_id": "R71_ADAPTIVE_BUDGET_PUBLIC_PROBE",
        "status": "PUBLIC_MANUAL_LABEL_DIAGNOSTIC",
        "source_videos": len(rows),
        "label_rows": sum(len(v) for v in by_video.values()),
        "median_occupancy": q50,
        "rule": "clip(0.75 + 0.10 * occupancy / median_occupancy, 0.70, 0.85)",
        "rank": "center + 0.20 * interval membership rank",
        "mean_soft_fixed": float(np.mean([r["soft_fixed"] for r in rows])),
        "mean_soft_adaptive": float(np.mean([r["soft_adaptive"] for r in rows])),
        "soft_delta": soft_delta,
        "mean_binary_fixed": float(np.mean([r["binary_fixed"] for r in rows])),
        "mean_binary_adaptive": float(np.mean([r["binary_adaptive"] for r in rows])),
        "binary_delta": binary_delta,
        "positive_soft_views": int(sum(r["soft_adaptive"] > r["soft_fixed"] for r in rows)),
        "positive_binary_views": int(sum(r["binary_adaptive"] > r["binary_fixed"] for r in rows)),
        "limitations": [
            "ten public windows only; labels are project diagnostics, not official AIC GT",
            "binary result is slightly negative while the soft grade result is slightly positive",
            "no independent confirmation and no official score was used",
        ],
        "per_video": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
