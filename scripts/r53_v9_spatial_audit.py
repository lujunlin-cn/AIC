#!/usr/bin/env python3
"""Audit the size of the V9 vision-only spatial intervention.

This compares full-frame R26 B3 boxes with the already generated V9 boxes.
It deliberately computes geometry only; without native AIC crop GT it cannot
claim which box is better.  The selected-frame result uses the exact R26
centre-85 mask and is therefore the relevant isolation quantity for the
packaged ``R47_V9_B3_CENTER85`` candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, dict]:
    return {
        str((row := json.loads(line))["video_id"]): row
        for line in path.read_text().splitlines()
        if line.strip()
    }


def box_iou(a: np.ndarray, b: np.ndarray, rw: float, rh: float) -> float:
    ah = float(a[2]) * rh / rw
    bh = float(b[2]) * rh / rw
    ix = max(0.0, min(float(a[0] + a[2]), float(b[0] + b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[1] + ah), float(b[1] + bh)) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    union = float(a[2] * ah + b[2] * bh - inter)
    return inter / union if union > 0 else 1.0


def audit(r26_path: Path, v9_path: Path) -> dict:
    r26, v9 = load(r26_path), load(v9_path)
    if set(r26) != set(v9):
        raise ValueError("R26/V9 video sets differ")
    all_rows: list[dict] = []
    selected_rows: list[dict] = []
    for vid in sorted(r26, key=lambda x: int(x)):
        r0, r1 = r26[vid], v9[vid]
        if r0["targetRatioWH"] != r1["targetRatioWH"]:
            raise ValueError(f"{vid}: targetRatioWH differs")
        rw, rh = r0["targetRatioWH"]
        p0 = {int(x["frame"]): np.asarray(x["bboxes"], dtype=float) for x in r0["predictions"]}
        p1 = {int(x["frame"]): np.asarray(x["bboxes"], dtype=float) for x in r1["predictions"]}
        if set(p0) != set(p1):
            raise ValueError(f"{vid}: frame sets differ")
        n = len(p0)
        k = int(round(0.85 * n))
        lo = (n - k) // 2
        for j, frame in enumerate(sorted(p0)):
            rec = {
                "video_id": vid,
                "frame": frame,
                "ratio": [rw, rh],
                "box_iou": box_iou(p0[frame], p1[frame], rw, rh),
                # Ignore serializer-level float noise; this is a geometric
                # intervention only when overlap falls below 0.999999.
                "changed": bool(box_iou(p0[frame], p1[frame], rw, rh) < 0.999999),
                "selected_center85": bool(lo <= j < lo + k),
            }
            all_rows.append(rec)
            if rec["selected_center85"]:
                selected_rows.append(rec)

    def summarize(rows: list[dict], include_ratio: bool = True) -> dict:
        if not rows:
            return {"rows": 0}
        x = np.asarray([r["box_iou"] for r in rows], dtype=float)
        out = {
            "rows": len(rows),
            "mean_box_iou": float(x.mean()),
            "p01_box_iou": float(np.percentile(x, 1)),
            "changed_share": float(np.mean([r["changed"] for r in rows])),
        }
        if include_ratio:
            out["ratio"] = {
                f"{rw}:{rh}": summarize(
                    [r for r in rows if r["ratio"] == [rw, rh]], include_ratio=False
                )
                for rw, rh in ((16, 9), (9, 16))
            }
        return out

    return {
        "protocol": "R53_V9_SPATIAL_DISAGREEMENT_AUDIT",
        "claim_boundary": "Geometry-only; no AIC native crop GT and no official labels used.",
        "r26_sha256": None,
        "v9_sha256": None,
        "all_frames": summarize(all_rows),
        "selected_r26_center85": summarize(selected_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26", type=Path, required=True)
    ap.add_argument("--v9", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = audit(args.r26, args.v9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
