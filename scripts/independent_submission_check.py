#!/usr/bin/env python3
"""Dependency-free second checker for an AIC JSONL submission.

This intentionally does not import ``aic.contract`` so it can catch a shared
validator bug. It checks the frozen metadata index, complete video coverage,
finite JSON values, frame order/uniqueness, target ratio, legal [x,y,w]
geometry, and a measured model-size value when supplied.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line {number}")
        value = json.loads(line)
        if not isinstance(value, dict) or not finite_tree(value):
            raise ValueError(f"invalid/nonfinite JSON object at line {number}")
        rows.append(value)
    if not rows:
        raise ValueError("empty JSONL")
    return rows


def check(index_path, submission_path, expected_bytes=None, require_size=True):
    index_rows = read_jsonl(index_path)
    index = {r["video_id"]: r for r in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("duplicate index video_id")
    rows = read_jsonl(submission_path)
    seen_ids = set()
    sizes = []
    predictions = 0
    for row in rows:
        vid = row.get("video_id")
        if vid not in index or vid in seen_ids:
            raise ValueError(f"unknown/duplicate video_id: {vid!r}")
        seen_ids.add(vid)
        meta = index[vid]
        if row.get("targetRatioWH") != meta.get("targetRatioWH"):
            raise ValueError(f"targetRatioWH mismatch for {vid}")
        # preliminary-stage packages may omit the field entirely (rules 6.1)
        if "model_size_mb" in row or require_size:
            if "model_size_mb" not in row or not isinstance(row["model_size_mb"], (int, float)):
                raise ValueError(f"missing model_size_mb for {vid}")
            sizes.append(float(row["model_size_mb"]))
            if not 0.0 < sizes[-1] <= 9216.0 or not math.isfinite(sizes[-1]):
                raise ValueError(f"invalid model_size_mb for {vid}")
        rw, rh = map(float, meta["targetRatioWH"])
        last = None
        frames = set()
        for pred in row.get("predictions", []):
            if not isinstance(pred, dict):
                raise ValueError(f"prediction object expected for {vid}")
            frame = pred.get("frame")
            if type(frame) is not int or not 0 <= frame < int(meta["frame_count"]):
                raise ValueError(f"frame out of range for {vid}: {frame!r}")
            if frame in frames or (last is not None and frame < last):
                raise ValueError(f"duplicate/unsorted frame for {vid}")
            frames.add(frame); last = frame
            crop = pred.get("bboxes")
            if not isinstance(crop, list) or len(crop) != 3:
                raise ValueError(f"bboxes must be [x,y,w] for {vid}")
            x, y, w = map(float, crop)
            h = w * rh / rw
            if not all(math.isfinite(v) for v in (x, y, w, h)) or x < 0 or y < 0 or w <= 0:
                raise ValueError(f"invalid crop for {vid}")
            if x + w > int(meta["width"]) + 1e-7 or y + h > int(meta["height"]) + 1e-7:
                raise ValueError(f"crop out of bounds for {vid}")
            predictions += 1
    missing = sorted(set(index) - seen_ids)
    if missing:
        raise ValueError(f"missing video IDs: {missing[:5]} (total {len(missing)})")
    if not require_size and not sizes:
        return {"valid": True, "videos": len(rows), "predictions": predictions,
                "model_size_mb": None, "checker": "independent_submission_check_v1"}
    if len(sizes) != len(rows):
        raise ValueError("model_size_mb present on some rows only")
    if max(sizes) - min(sizes) > 1e-9:
        raise ValueError("model_size_mb differs across rows")
    if expected_bytes is not None and not math.isclose(sizes[0], expected_bytes / 1_000_000, abs_tol=1e-9):
        raise ValueError("model_size_mb does not equal supplied weight bytes")
    return {"valid": True, "videos": len(rows), "predictions": predictions,
            "model_size_mb": sizes[0], "checker": "independent_submission_check_v1"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--weight-bytes", type=int)
    args = ap.parse_args(argv)
    print(json.dumps(check(args.index, args.submission, args.weight_bytes), indent=2))


if __name__ == "__main__":
    main()
