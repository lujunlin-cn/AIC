#!/usr/bin/env python3
"""Use the R70 native-crop utility as a fixed temporal residual.

The R70 head predicts the best legal crop utility for each 1-second keyframe.
This script ranks that utility within each video, then adds a fixed ``eta``
residual to the frozen R54 TimeLens2+centre score.  The temporal decision is
therefore changed while the V9 B3 spatial boxes remain byte-for-byte the
same.  Official labels and scores are never read.
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
SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def parse_spans(text: str) -> list[tuple[float, float]]:
    m = SPAN_RE.search(text or "")
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for item in arr:
        if len(item) < 2:
            continue
        a, b = float(item[0]), float(item[1])
        if math.isfinite(a) and math.isfinite(b) and b >= a:
            out.append((a, b))
    return out


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / max(len(values) - 1, 1)


def utility_for_video(path: Path, n: int, key: str) -> tuple[np.ndarray, dict]:
    d = json.loads(path.read_text())
    t = d["ratios"]["t"]
    if any(s != "ok_r70" for s in t["status"]):
        raise ValueError(f"{path.name}: R70 fallback status present")
    keyframes = np.asarray(d["keyframes"], dtype=np.float64)
    vals = np.asarray([float(x[key]) for x in t["utility"]], dtype=np.float64)
    if len(keyframes) != len(vals) or len(vals) < 1:
        raise ValueError(f"{path.name}: utility/keyframe length mismatch")
    frame = np.arange(n, dtype=np.float64)
    utility = np.interp(frame, keyframes, vals)
    return utility, {
        "key": key,
        "keyframes": int(len(keyframes)),
        "min": float(utility.min()),
        "max": float(utility.max()),
        "mean": float(utility.mean()),
        "std": float(utility.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True, help="V9 B3 full-frame JSONL")
    ap.add_argument("--r54", type=Path, required=True, help="R54 selected-frame JSONL, used as identity control")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--utility-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--utility-key", default="max_score", choices=("max_score", "center_score", "std_score"))
    ap.add_argument("--eta", type=float, default=ETA)
    args = ap.parse_args()
    if not (0.0 <= args.eta <= 2.0):
        raise ValueError("eta outside fixed residual range")

    intake = {str(x["video_id"]): x for x in rows(args.index)}
    parent = {str(x["video_id"]): x for x in rows(args.parent)}
    r54 = {str(x["video_id"]): x for x in rows(args.r54)}
    response = {str(x["video_id"]): x.get("response", "") for x in json.loads(args.responses.read_text())["results"]}
    if set(parent) != set(intake) or set(parent) != set(response):
        raise ValueError("parent/index/response coverage mismatch")

    out, details = [], []
    total = changed = 0
    for vid in sorted(parent, key=lambda x: int(x) if x.isdigit() else x):
        row = parent[vid]
        meta = intake[vid]
        n = int(meta["frame_count"])
        if len(row["predictions"]) != n:
            raise ValueError(f"{vid}: full parent coverage mismatch")
        fps = float(meta["fps"])
        frame = np.arange(n, dtype=np.int64)
        times = frame.astype(np.float64) / fps
        spans = parse_spans(response[vid])
        inside = np.asarray([any(a - 1e-6 <= t <= b + 1e-6 for a, b in spans) for t in times], dtype=np.float64)
        centre = 1.0 - np.abs(times / max(float(times[-1]), 1e-9) - 0.5) * 2.0
        base_score = rank01(centre) + 0.20 * (rank01(inside) - 0.5)
        k = int(round(0.85 * n))
        base_idx = np.argsort(-base_score, kind="mergesort")[:k]
        base_set = set(frame[base_idx].tolist())
        observed_r54 = {int(x["frame"]) for x in r54[vid]["predictions"]}
        if base_set != observed_r54:
            raise ValueError(f"{vid}: reconstructed R54 set differs ({len(base_set ^ observed_r54)} frames)")

        utility, us = utility_for_video(args.utility_dir / f"{vid}.json", n, args.utility_key)
        score = base_score + args.eta * (rank01(utility) - 0.5)
        new_idx = np.argsort(-score, kind="mergesort")[:k]
        new_set = set(frame[new_idx].tolist())
        source = {int(x["frame"]): x for x in row["predictions"]}
        # The V9 full parent contains legal B3 boxes on every original frame.
        q = dict(row)
        q["predictions"] = [row["predictions"][int(i)] for i in sorted(new_idx)]
        q["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        q["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        out.append(q)
        swaps = len(base_set ^ new_set) // 2
        changed += swaps
        total += k
        details.append({
            "video_id": vid,
            "n_full": n,
            "n_keep": k,
            "eta": float(args.eta),
            "spans": spans,
            "utility": us,
            "swaps_vs_r54": int(swaps),
            "overlap_vs_r54": float(len(base_set & new_set) / max(k, 1)),
        })

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    with pred.open("w") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n")
    zip_path = args.output / f"{args.output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip mismatch")

    total_params = TIME_LENS_PARAMS + V9_PARAMS
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "R70 native-crop utility max-score rank added as fixed eta residual to R54 TimeLens2+centre temporal ranking",
        "spatial_mode": "v9_b3_artifact",
        "spatial_note": "V9 B3 boxes copied byte-for-byte from full-frame parent; only temporal set changes",
        "utility_head": "R70 public native-crop readout, max legal-window score, 1-second keyframes linearly interpolated",
        "utility_key": args.utility_key,
        "eta": float(args.eta),
        "parameter_components": [
            {"name": "TimeLens2-8B", "parameters": TIME_LENS_PARAMS},
            {"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS},
        ],
        "total_parameters": total_params,
        "model_params_b": total_params / 1e9,
        "video_count": len(out),
        "prediction_count": total,
        "temporal_swaps_vs_r54": changed,
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "official_test_labels_used": False,
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / f"{zip_path.name}.sha256").write_text(sha(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
