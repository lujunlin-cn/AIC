#!/usr/bin/env python3
"""Package a TimeLens2 event-completion residual candidate.

R54 uses the within-video rank of a binary interval mask.  That rank gives
every frame in an interval a small frame-order ramp, including when several
intervals overlap.  This candidate uses the model's interval contract more
explicitly: each frame receives its normalized phase within the nearest
predicted event (late, outcome-like frames get higher evidence), while the
centre-85 policy remains the anchor.  No official labels or score feedback
are read.
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
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def parse(text: str) -> list[tuple[float, float]]:
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


def rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) / max(len(x) - 1, 1)


def phase_evidence(times: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    """Return max normalized completion phase over all predicted intervals.

    Outside intervals the evidence is zero.  Zero length intervals use a
    one-frame denominator and therefore only reward the exact endpoint.
    """
    out = np.zeros(len(times), dtype=float)
    for a, b in intervals:
        inside = (times >= a - 1e-6) & (times <= b + 1e-6)
        phase = np.clip((times - a) / max(b - a, 1e-6), 0.0, 1.0)
        out = np.maximum(out, np.where(inside, phase, 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    response_rows = json.loads(args.responses.read_text()).get("results", [])
    spans = {str(x["video_id"]): parse(x.get("response", "")) for x in response_rows}
    intake = {str(x["video_id"]): x for x in rows(args.index)}
    parent = rows(args.parent)
    if set(spans) != set(intake):
        raise ValueError(f"response/index coverage mismatch: {len(spans)} vs {len(intake)}")

    out, details, total = [], [], 0
    key = lambda x: int(str(x["video_id"])) if str(x["video_id"]).isdigit() else str(x["video_id"])
    for row in sorted(parent, key=key):
        vid = str(row["video_id"])
        meta = intake[vid]
        allp = row["predictions"]
        n = int(meta["frame_count"])
        if len(allp) != n:
            raise ValueError(f"{vid}: parent coverage {len(allp)} != {n}")
        fps = float(meta.get("fps", 0.0))
        if fps <= 0:
            raise ValueError(f"{vid}: missing positive fps")
        frame = np.asarray([int(p["frame"]) for p in allp], dtype=np.int64)
        times = frame / fps
        intervals = spans.get(vid, [])
        phase = phase_evidence(times, intervals)
        centre = 1.0 - np.abs(times / max(float(times[-1]), 1e-9) - 0.5) * 2.0
        score = rank01(centre) + ETA * (rank01(phase) - 0.5)
        k = int(round(0.85 * n))
        chosen = np.sort(frame[np.argsort(-score, kind="mergesort")[:k]])
        source = {int(p["frame"]): p for p in allp}
        q = dict(row)
        q["predictions"] = [source[int(f)] for f in chosen]
        q["model_size_mb"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e6
        q["model_params_b"] = (TIME_LENS_PARAMS + V9_PARAMS) / 1e9
        total += len(q["predictions"])
        out.append(q)
        details.append({
            "video_id": vid,
            "n_full": n,
            "n_keep": k,
            "intervals": intervals,
            "phase_nonzero": int(np.count_nonzero(phase)),
            "phase_mean": float(phase.mean()) if len(phase) else 0.0,
            "selected_in_intervals": int(sum(phase[np.isin(frame, chosen)] > 0)),
        })

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    with pred.open("w") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n")
    zip_path = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip SHA mismatch")

    total_params = TIME_LENS_PARAMS + V9_PARAMS
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "official_platform_score": None,
        "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "mechanism": "TimeLens2 predicted intervals converted to normalized completion phase, used as fixed eta=.20 residual around exact centre-85 rank",
        "spatial_mode": "v9_b3_artifact",
        "spatial_note": "V9 vision-only B3 boxes copied from audited full-frame artifact",
        "eligibility": "PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED",
        "parameter_components": [
            {"name": "TimeLens2-8B", "parameters": TIME_LENS_PARAMS},
            {"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS},
        ],
        "total_parameters": total_params,
        "model_params_b": total_params / 1e9,
        "eta": ETA,
        "video_count": len(out),
        "prediction_count": total,
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "responses_sha256": sha(args.responses),
        "parent_sha256": sha(args.parent),
        "index_sha256": sha(args.index),
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zip_path.name + ".sha256")).write_text(sha(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
