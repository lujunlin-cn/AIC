#!/usr/bin/env python3
"""Package a high-risk joint temporal/spatial utility decoder (R59).

R54 ranks frames with the TimeLens2 interval residual around a centre-85
anchor.  This candidate adds a deliberately strong, predeclared spatial
utility term: per-frame disagreement between the V9 B3 crop and the older R26
B3 crop.  The disagreement is treated as a content-sensitive crop signal;
frames where the spatial readout moves are given more utility.  The same V9
box is emitted for a selected frame, so this is a joint *frame-set* decision,
not a box post-processing trick.

This is intentionally an aggressive risk arm.  It does not read official
labels or scores and does not search the coefficient: ``GAMMA=1.0`` is fixed
in the experiment definition.  The old R26 artifact is used only as a
deployment-time disagreement reference and is counted in the manifest as a
cached artifact (the deployed neural components remain TimeLens2 + V9).
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
GAMMA = 1.0
ETA = 0.20
KEEP = 0.85
TIME_RE = re.compile(r"\[\s*\[.*?\]\s*\]")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_intervals(response: str) -> list[tuple[float, float]]:
    match = TIME_RE.search(response or "")
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except Exception:
        return []
    out: list[tuple[float, float]] = []
    for pair in value if isinstance(value, list) else []:
        try:
            a, b = float(pair[0]), float(pair[1])
        except Exception:
            continue
        if math.isfinite(a) and math.isfinite(b) and b >= a:
            out.append((a, b))
    return out


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order / max(len(values) - 1, 1)


def disagreement(v9: list[dict], r26: list[dict], width: float, height: float) -> np.ndarray:
    old = {int(x["frame"]): x["bboxes"] for x in r26}
    scale = max(float(width), float(height), 1.0)
    result = []
    for item in v9:
        frame = int(item["frame"])
        if frame not in old:
            raise ValueError(f"missing R26 box for frame {frame}")
        a, b = item["bboxes"], old[frame]
        if len(a) != len(b):
            raise ValueError(f"box length mismatch at frame {frame}")
        # L1 geometry change is a conservative observable proxy for a
        # spatial decision boundary.  It is normalized by image extent.
        result.append(sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / scale)
    return np.asarray(result, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True, help="R54 selected V9 predictions")
    ap.add_argument("--timelens", type=Path, required=True, help="TimeLens2 all-video responses")
    ap.add_argument("--index", type=Path, required=True, help="426-video frame metadata")
    ap.add_argument("--v9-full", type=Path, required=True, help="full-frame V9 B3 artifact")
    ap.add_argument("--r26-full", type=Path, required=True, help="full-frame R26 B3 artifact")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="fixed spatial-disagreement rank weight; package config must record it")
    args = ap.parse_args()

    parent = {str(x["video_id"]): x for x in read_jsonl(args.parent)}
    index = {str(x["video_id"]): x for x in read_jsonl(args.index)}
    v9 = {str(x["video_id"]): x for x in read_jsonl(args.v9_full)}
    r26 = {str(x["video_id"]): x for x in read_jsonl(args.r26_full)}
    tl = json.loads(args.timelens.read_text())
    responses = {str(x["video_id"]): str(x.get("response", "")) for x in tl.get("results", [])}
    if set(parent) != set(index) or set(parent) != set(v9) or set(parent) != set(r26):
        raise ValueError("parent/index/V9/R26 video ID sets differ")
    if len(parent) != 426:
        raise ValueError(f"expected 426 videos, got {len(parent)}")

    output_rows: list[dict] = []
    details: list[dict] = []
    total = 0
    for vid in sorted(parent, key=lambda x: int(x) if x.isdigit() else x):
        meta = index[vid]
        n = int(meta["frame_count"])
        frames = v9[vid]["predictions"]
        old = r26[vid]["predictions"]
        if len(frames) != n or len(old) != n:
            raise ValueError(f"{vid}: full-frame length mismatch")
        fps = float(meta.get("fps", 0.0) or 0.0)
        if fps <= 0:
            raise ValueError(f"{vid}: index must contain verified positive fps; use enriched index")
        t = np.arange(n, dtype=np.float64) / max(fps, 1e-9)
        intervals = parse_intervals(responses.get(vid, ""))
        inside = np.asarray(
            [any(a - 1e-6 <= z <= b + 1e-6 for a, b in intervals) for z in t],
            dtype=np.float64,
        )
        # Keep the exact centre expression used by R54.  The algebraically
        # equivalent index form can differ at floating ties and shift one
        # boundary frame under stable sorting.
        center = 1.0 - np.abs(t / max(float(t[-1]), 1e-9) - 0.5) * 2.0
        spatial_delta = disagreement(frames, old, meta.get("width", 1), meta.get("height", 1))
        # R54 temporal residual plus a fixed, deliberately large spatial term.
        gamma = float(args.gamma)
        score = rank01(center) + ETA * (rank01(inside) - 0.5) + gamma * (rank01(spatial_delta) - 0.5)
        k = int(round(KEEP * n))
        chosen = np.sort(np.argsort(-score, kind="mergesort")[:k])
        by_frame = {int(x["frame"]): x for x in frames}
        predictions = [dict(by_frame[int(frame)]) for frame in chosen]
        row = {"video_id": vid, "targetRatioWH": v9[vid]["targetRatioWH"], "predictions": predictions,
               "model_size_mb": (TIME_LENS_PARAMS + V9_PARAMS) / 1e6,
               "model_params_b": (TIME_LENS_PARAMS + V9_PARAMS) / 1e9}
        output_rows.append(row)
        parent_set = {int(x["frame"]) for x in parent[vid]["predictions"]}
        chosen_set = set(map(int, chosen))
        gaps = int(np.sum(np.diff(chosen) > 1)) if len(chosen) > 1 else 0
        details.append({
            "video_id": vid, "n_full": n, "n_keep": k,
            "intervals": intervals, "selected_in_timelens": int(inside[chosen].sum()),
            "spatial_disagreement_mean_selected": float(spatial_delta[chosen].mean()),
            "spatial_disagreement_mean_full": float(spatial_delta.mean()),
            "swaps_vs_r54": len(parent_set ^ chosen_set) // 2,
            "noncontiguous_internal_gaps": gaps,
            "selected_position_mean": float(np.mean(chosen / max(n - 1, 1))),
        })
        total += len(predictions)

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in output_rows))
    zip_path = args.output / f"{args.output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            if zf.namelist() != ["predictions.jsonl"]:
                raise ValueError(f"unexpected ZIP members: {zf.namelist()}")
            zf.extractall(tmp)
        if sha(Path(tmp) / "predictions.jsonl") != sha(pred):
            raise ValueError("ZIP round-trip hash mismatch")

    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": ("joint frame utility: R54 TimeLens2 centre residual plus "
                       f"{('high' if float(args.gamma) >= 0 else 'low')} V9-vs-R26 spatial disagreement"),
        "utility": f"rank(center)+0.20*rank(TimeLens2 interval)+{float(args.gamma):g}*rank(V9_R26_geometry_disagreement)",
        "gamma": float(args.gamma), "eta": ETA, "keep_fraction": KEEP,
        "spatial_mode": "v9_b3_artifact",
        "parameter_components": [
            {"name": "TimeLens2-8B", "parameters": TIME_LENS_PARAMS},
            {"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS},
        ],
        "cached_reference": "R26 B3 boxes are used only to form a deployment-time geometry disagreement feature",
        "total_parameters": TIME_LENS_PARAMS + V9_PARAMS,
        "model_params_b": (TIME_LENS_PARAMS + V9_PARAMS) / 1e9,
        "video_count": len(output_rows), "prediction_count": total,
        "prediction_sha256": sha(pred), "zip_sha256": sha(zip_path), "zip_bytes": zip_path.stat().st_size,
        "parent_sha256": sha(args.parent), "timelens_sha256": sha(args.timelens),
        "v9_full_sha256": sha(args.v9_full), "r26_full_sha256": sha(args.r26_full),
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / f"{zip_path.name}.sha256").write_text(sha(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
