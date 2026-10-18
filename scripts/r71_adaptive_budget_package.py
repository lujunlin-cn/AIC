#!/usr/bin/env python3
"""Build a fixed, per-video adaptive-budget TimeLens2 candidate.

The R54 temporal decoder is retained as the ranking function and its V9
spatial boxes are retained.  Only the number of already selected frames is
changed.  The predicted interval occupancy is used as an unlabeled estimate
of how much of a video contains useful event evidence:

    keep_fraction = clip(0.75 + 0.10 * occupancy / median_occupancy,
                         0.70, 0.85)

The median is computed once from the 426 inference responses.  This is a
predeclared rule with no official labels or score feedback.  The cap at 0.85
means the candidate never needs frames or spatial boxes absent from the R54
parent; it is therefore a clean budget-only intervention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np

TIME_LENS_PARAMS = 8_767_123_696
V9_PARAMS = 87_407_729
ETA = 0.20
FLOOR = 0.70
ANCHOR = 0.75
SLOPE = 0.10
CAP = 0.85
SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_spans(text: str) -> list[tuple[float, float]]:
    match = SPAN_RE.search(text or "")
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except Exception:
        return []
    out: list[tuple[float, float]] = []
    for item in value:
        if len(item) < 2:
            continue
        a, b = float(item[0]), float(item[1])
        if math.isfinite(a) and math.isfinite(b) and b >= a:
            out.append((a, b))
    return out


def union_occupancy(intervals: list[tuple[float, float]], duration: float) -> float:
    """Length of the clipped union of predicted intervals / video duration."""
    if duration <= 0 or not intervals:
        return 0.0
    clipped = sorted((max(0.0, a), min(duration, b)) for a, b in intervals if b >= 0 and a <= duration)
    clipped = [(a, b) for a, b in clipped if b >= a]
    if not clipped:
        return 0.0
    total = 0.0
    start, end = clipped[0]
    for a, b in clipped[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    total += end - start
    return float(np.clip(total / duration, 0.0, 1.0))


def rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(float) / (len(values) - 1)


def key(value: str):
    return int(value) if value.isdigit() else value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True, help="R54 V9 parent predictions JSONL")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    response_payload = json.loads(args.responses.read_text(encoding="utf-8"))
    response_rows = {str(x["video_id"]): str(x.get("response", "")) for x in response_payload.get("results", [])}
    intake = {str(x["video_id"]): x for x in rows(args.index)}
    parent = rows(args.parent)
    parent_map = {str(x["video_id"]): x for x in parent}
    if set(response_rows) != set(intake) or set(parent_map) != set(intake):
        raise ValueError("response/index/parent video coverage mismatch")

    occupancy = {}
    intervals_by_video = {}
    for vid, meta in intake.items():
        spans = parse_spans(response_rows[vid])
        intervals_by_video[vid] = spans
        occupancy[vid] = union_occupancy(spans, float(meta["duration"]))
    q50 = float(np.median(np.asarray(list(occupancy.values()), dtype=float)))
    if not math.isfinite(q50) or q50 <= 0:
        raise ValueError(f"invalid positive median occupancy: {q50}")

    output_rows: list[dict] = []
    details: list[dict] = []
    total = 0
    base_total = 0
    for vid in sorted(parent_map, key=key):
        parent_row = parent_map[vid]
        meta = intake[vid]
        selected = list(parent_row["predictions"])
        n_full = int(meta["frame_count"])
        base_k = int(round(CAP * n_full))
        if len(selected) != base_k:
            raise ValueError(f"{vid}: parent selected {len(selected)} != round(.85*{n_full})={base_k}")
        fps = float(meta["fps"])
        if fps <= 0:
            raise ValueError(f"{vid}: missing positive fps")
        q = occupancy[vid]
        fraction = float(np.clip(ANCHOR + SLOPE * q / q50, FLOOR, CAP))
        keep_k = min(base_k, max(1, int(round(fraction * n_full))))
        frame = np.asarray([int(item["frame"]) for item in selected], dtype=np.int64)
        times = frame / fps
        spans = intervals_by_video[vid]
        inside = np.asarray([any(a - 1e-6 <= t <= b + 1e-6 for a, b in spans) for t in times], dtype=float)
        center = 1.0 - np.abs(times / max(float(times[-1]), 1e-9) - 0.5) * 2.0
        score = rank01(center) + ETA * (rank01(inside) - 0.5)
        order = np.argsort(-score, kind="mergesort")
        chosen_idx = np.sort(order[:keep_k])
        out = dict(parent_row)
        out["predictions"] = [selected[int(i)] for i in chosen_idx]
        # R54's actual deployed components remain present in the candidate.
        out["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        out["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        output_rows.append(out)
        total += keep_k
        base_total += base_k
        details.append({
            "video_id": vid,
            "frame_count": n_full,
            "base_keep": base_k,
            "adaptive_keep": keep_k,
            "base_fraction": base_k / n_full,
            "adaptive_fraction": keep_k / n_full,
            "interval_occupancy": q,
            "intervals": spans,
            "selected_inside_count": int(inside[chosen_idx].sum()),
            "removed_count": base_k - keep_k,
        })

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in output_rows), encoding="utf-8")
    zip_path = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp)
        if sha(Path(temp) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip SHA mismatch")

    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "per-video adaptive frame budget from unlabeled TimeLens interval occupancy; R54 rank and V9 boxes retained",
        "budget_rule": {
            "formula": "clip(0.75 + 0.10 * occupancy / median_occupancy, 0.70, 0.85)",
            "occupancy_definition": "clipped union duration of TimeLens intervals / intake duration",
            "anchor": ANCHOR,
            "slope": SLOPE,
            "floor": FLOOR,
            "cap": CAP,
            "median_occupancy": q50,
            "official_labels_used": False,
            "official_score_used": False,
        },
        "temporal_rank": "R54 center rank + eta=.20 TimeLens interval rank, restricted to R54 selected frames",
        "spatial_mode": "R54 V9 vision-only boxes copied unchanged on retained frames",
        "parameter_components": [
            {"name": "TimeLens2-8B", "parameters": TIME_LENS_PARAMS},
            {"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS},
        ],
        "total_parameters": TIME_LENS_PARAMS + V9_PARAMS,
        "model_params_b": (TIME_LENS_PARAMS + V9_PARAMS) / 1e9,
        "video_count": len(output_rows),
        "prediction_count": total,
        "base_prediction_count": base_total,
        "removed_predictions": base_total - total,
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "responses_sha256": sha(args.responses),
        "parent_sha256": sha(args.parent),
        "index_sha256": sha(args.index),
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / (zip_path.name + ".sha256")).write_text(sha(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
