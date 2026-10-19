#!/usr/bin/env python3
"""Build a tie-safe geometry-disagreement residual on the exact R54 utility.

R54's centre and TimeLens interval ranks are retained byte-for-byte in their
stable ordinal form. Only the V9/R26 geometry disagreement uses average ranks,
so the large exact-zero tie class cannot create a hidden frame-index prior.
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
from scipy.stats import rankdata

TIME_LENS_PARAMS = 8_767_123_696
V9_PARAMS = 87_407_729
ETA = 0.20
KEEP = 0.85
TIME_RE = re.compile(r"\[\s*\[.*?\]\s*\]")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse(text: str) -> list[tuple[float, float]]:
    match = TIME_RE.search(text or "")
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
    except Exception:
        return []
    out: list[tuple[float, float]] = []
    for item in arr if isinstance(arr, list) else []:
        try:
            start, end = float(item[0]), float(item[1])
        except Exception:
            continue
        if math.isfinite(start) and math.isfinite(end) and end >= start:
            out.append((start, end))
    return out


def rank01(values: np.ndarray) -> np.ndarray:
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(n, dtype=np.float64)
    result[order] = np.arange(n, dtype=np.float64)
    return result / max(n - 1, 1)


def tie_rank01(values: np.ndarray) -> np.ndarray:
    n = len(values)
    return (rankdata(values, method="average") - 1.0) / max(n - 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("parent", "timelens", "index", "v9_full", "r26_full", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()

    parent = {str(x["video_id"]): x for x in read(args.parent)}
    index = {str(x["video_id"]): x for x in read(args.index)}
    v9_full = {str(x["video_id"]): x for x in read(args.v9_full)}
    r26_full = {str(x["video_id"]): x for x in read(args.r26_full)}
    timelens = json.loads(args.timelens.read_text())
    responses = {str(x["video_id"]): str(x.get("response", "")) for x in timelens.get("results", [])}

    rows: list[dict] = []
    details: list[dict] = []
    total_swaps = 0
    for vid in sorted(parent, key=lambda value: int(value) if value.isdigit() else value):
        meta = index[vid]
        n = int(meta["frame_count"])
        fps = float(meta["fps"])
        frames = v9_full[vid]["predictions"]
        old_frames = r26_full[vid]["predictions"]
        if len(frames) != n or len(old_frames) != n:
            raise ValueError(f"{vid}: full-frame coverage mismatch")

        times = np.arange(n, dtype=np.float64) / fps
        intervals = parse(responses.get(vid, ""))
        inside = np.asarray(
            [any(start - 1e-6 <= t <= end + 1e-6 for start, end in intervals) for t in times],
            dtype=np.float64,
        )
        centre = 1.0 - np.abs(times / max(times[-1], 1e-9) - 0.5) * 2.0

        old_by_frame = {int(item["frame"]): item["bboxes"] for item in old_frames}
        scale = max(float(meta.get("width", 1)), float(meta.get("height", 1)), 1.0)
        disagreement = np.asarray(
            [
                sum(abs(float(a) - float(b)) for a, b in zip(item["bboxes"], old_by_frame[int(item["frame"])]))
                / scale
                for item in frames
            ],
            dtype=np.float64,
        )

        # Exact R54 terms; only disagreement receives tie-aware ranks.
        score = rank01(centre) + ETA * (rank01(inside) - 0.5) + tie_rank01(disagreement) - 0.5
        keep = int(round(KEEP * n))
        chosen = np.sort(np.argsort(-score, kind="mergesort")[:keep])
        by_frame = {int(item["frame"]): item for item in frames}
        row = dict(v9_full[vid])
        row["predictions"] = [dict(by_frame[int(frame)]) for frame in chosen]
        row["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        row["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        rows.append(row)

        r54_set = {int(item["frame"]) for item in parent[vid]["predictions"]}
        chosen_set = set(map(int, chosen))
        swaps = len(r54_set ^ chosen_set) // 2
        total_swaps += swaps
        details.append(
            {
                "video_id": vid,
                "n_full": n,
                "n_keep": keep,
                "swaps_vs_r54": swaps,
                "position_mean": float(np.mean(chosen / max(n - 1, 1))),
                "zero_disagreement_fraction": float(np.mean(disagreement == 0.0)),
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
    zip_path = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temporary)
        if sha(Path(temporary) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip mismatch")

    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "exact R54 centre+TimeLens residual plus tie-aware V9/R26 geometry disagreement",
        "utility": "rank_stable(center)+0.20*rank_stable(TimeLens2 interval)+1.0*rank_tieavg(V9_R26 geometry disagreement)-0.5",
        "rank_tie_policy": "average ranks for geometry exact ties only; R54 terms unchanged",
        "video_count": len(rows),
        "prediction_count": sum(len(row["predictions"]) for row in rows),
        "total_parameters": TIME_LENS_PARAMS + V9_PARAMS,
        "model_params_b": (TIME_LENS_PARAMS + V9_PARAMS) / 1e9,
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "r54_swaps": total_swaps,
        "mean_position": float(np.mean([item["position_mean"] for item in details])),
        "mean_zero_disagreement_fraction": float(np.mean([item["zero_disagreement_fraction"] for item in details])),
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zip_path.name + ".sha256")).write_text(sha(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
