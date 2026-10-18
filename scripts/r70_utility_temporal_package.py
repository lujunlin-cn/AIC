#!/usr/bin/env python3
"""Use R70 crop utility as a fixed joint time-space selection signal.

The rule is deliberately parameter free after per-video rank normalization:

  t = rank(center + .20 * (interval_rank - .5))
  u = rank(interpolated R70 top-crop utility)
  joint = t * (0.5 + 0.5*u)

The multiplicative form is an expected-utility proxy: a temporally valuable
frame is discounted when its predicted crop utility is low.  It keeps the
R54 interval residual and exact 85 percent budget, but changes the selection
using a spatial readout that was trained only on public native-crop data.
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

ETA = 0.20
TIME_LENS_PARAMS = 8_767_123_696
V9_PARAMS = 87_407_729
SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_intervals(text: str) -> list[tuple[float, float]]:
    m = SPAN_RE.search(text or "")
    if not m:
        return []
    try:
        value = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for item in value:
        if len(item) >= 2:
            a, b = float(item[0]), float(item[1])
            if math.isfinite(a) and math.isfinite(b) and b >= a:
                out.append((a, b))
    return out


def rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(float) / (len(values) - 1)


def utility_values(point_row: dict) -> tuple[np.ndarray, np.ndarray]:
    payload = point_row["ratios"]["t"]
    keyframes = np.asarray(point_row["keyframes"], dtype=np.float64)
    utility = payload.get("utility")
    if utility is None or len(utility) != len(keyframes):
        raise ValueError("R70 points do not contain aligned utility summaries")
    values = []
    for item in utility:
        if not isinstance(item, dict):
            values.append(np.nan)
        else:
            values.append(float(item.get("max_score", np.nan)))
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 2:
        raise ValueError("fewer than two finite R70 utility points")
    if not np.all(good):
        values[~good] = np.interp(keyframes[~good], keyframes[good], values[good])
    return keyframes, values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--utility-points", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True, help="full V9 vision-only frame predictions")
    ap.add_argument("--r54", type=Path, required=True, help="R54 selected predictions used for audit")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    response_rows = json.loads(args.responses.read_text(encoding="utf-8")).get("results", [])
    spans = {str(x["video_id"]): parse_intervals(str(x.get("response", ""))) for x in response_rows}
    point_files = list(args.utility_points.glob("*.json"))
    point_files += list(args.utility_points.glob("points/*.json"))
    point_files += list(args.utility_points.glob("shard_*/points/*.json"))
    points = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in point_files}
    intake = {str(x["video_id"]): x for x in rows(args.index)}
    full = {str(x["video_id"]): x for x in rows(args.parent)}
    r54 = {str(x["video_id"]): x for x in rows(args.r54)}
    if set(full) != set(intake) or set(full) != set(spans) or set(full) != set(points) or set(full) != set(r54):
        raise ValueError("full/R54/index/response/utility coverage mismatch")

    output_rows: list[dict] = []
    details: list[dict] = []
    for vid in sorted(full, key=lambda x: int(x) if x.isdigit() else x):
        meta = intake[vid]
        fp = full[vid]
        frame_items = fp["predictions"]
        frames = np.asarray([int(x["frame"]) for x in frame_items], dtype=np.int64)
        if len(frames) != int(meta["frame_count"]):
            raise ValueError(f"{vid}: full parent does not cover the original frame axis")
        fps = float(meta["fps"])
        times = frames / max(fps, 1e-9)
        intervals = spans[vid]
        inside = np.asarray([any(a - 1e-6 <= t <= b + 1e-6 for a, b in intervals) for t in times], dtype=float)
        center = 1.0 - np.abs(times / max(float(times[-1]), 1e-9) - 0.5) * 2.0
        temporal = rank01(center) + ETA * (rank01(inside) - 0.5)
        temporal_rank = rank01(temporal)
        kf, kval = utility_values(points[vid])
        utility = np.interp(frames.astype(float), kf, kval)
        utility_rank = rank01(utility)
        joint = temporal_rank * (0.5 + 0.5 * utility_rank)
        k = int(round(0.85 * len(frames)))
        order = np.argsort(-joint, kind="mergesort")
        chosen = np.sort(frames[order[:k]])
        by_frame = {int(x["frame"]): x for x in frame_items}
        out = dict(fp)
        out["predictions"] = [by_frame[int(f)] for f in chosen]
        out["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        out["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        output_rows.append(out)
        old = {int(x["frame"]) for x in r54[vid]["predictions"]}
        new = set(int(x) for x in chosen)
        details.append({
            "video_id": vid,
            "n_full": len(frames),
            "n_keep": k,
            "r54_intersection": len(old & new),
            "swaps_from_r54": len(old ^ new) // 2,
            "utility_rank_mean_selected": float(utility_rank[order[:k]].mean()),
            "temporal_rank_mean_selected": float(temporal_rank[order[:k]].mean()),
        })

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in output_rows), encoding="utf-8")
    zip_path = args.output / f"{args.output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip SHA mismatch")
    stats = {
        "videos": len(output_rows),
        "predictions": sum(len(x["predictions"]) for x in output_rows),
        "r54_mean_swaps": float(np.mean([x["swaps_from_r54"] for x in details])),
        "r54_videos_changed": int(sum(x["swaps_from_r54"] > 0 for x in details)),
    }
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "R70 topsoft native-crop utility drives parameter-free multiplicative joint temporal-spatial ranking",
        "utility_rule": "rank(R54_center+.20*rank(interval)) * (0.5 + 0.5*rank(interpolated_R70_top_score))",
        "utility_head": "R70 topsoft public native-crop objective",
        "official_test_labels_used": False,
        "total_parameters": TIME_LENS_PARAMS + V9_PARAMS,
        "model_params_b": (TIME_LENS_PARAMS + V9_PARAMS) / 1e9,
        "video_count": len(output_rows),
        "prediction_count": sum(len(x["predictions"]) for x in output_rows),
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "stats": stats,
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / f"{zip_path.name}.sha256").write_text(sha(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"per_video"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
