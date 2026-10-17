#!/usr/bin/env python3
"""Convert fixed TimeLens2 interval responses into a legal B3 submission.

TimeLens2 supplies only temporal intervals.  The spatial predictor is the
full-frame V9 vision-only B3 runtime, whose boxes are passed through without
modification.  The converter is deterministic: interval membership is the
primary rank and the normalized centre rank breaks ties, followed by the
exact ``round(.85 * N)`` budget.  Missing or unparsable responses fall back to
the R26 centre set for that video.

This script does not read official labels or scores.  TimeLens2 is counted in
the manifest together with the V9 vision tower, B3 head, and YuNet auxiliary.
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

import cv2
import numpy as np

KEEP = 0.85
PARAMS = 8_854_531_425
PARAMS_B = PARAMS / 1e9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def parse_intervals(text: str) -> list[tuple[float, float]]:
    """Parse only a JSON array of numeric two-element intervals."""
    m = re.search(r"\[\s*\[.*?\]\s*\]", text or "", flags=re.S)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for pair in raw if isinstance(raw, list) else []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            a, b = float(pair[0]), float(pair[1])
        except Exception:
            continue
        if math.isfinite(a) and math.isfinite(b) and b >= a >= 0:
            out.append((a, b))
    return out


def frame_meta(video_root: Path | None, vid: str, n: int,
               metadata: dict[str, dict] | None = None) -> tuple[float, int]:
    # The official intake index is authoritative for the original frame time
    # axis.  This avoids silently treating VFR videos as 30 fps when the media
    # files are not mounted on the packaging host.
    if metadata is not None and vid in metadata:
        item = metadata[vid]
        fps = float(item.get("fps", 0.0) or 0.0)
        count = int(item.get("frame_count", n) or n)
        if count != n:
            raise ValueError(f"{vid}: B3 frame count {n} != metadata count {count}")
        if fps > 0:
            return fps, count
    if video_root is None:
        return 30.0, n
    cap = cv2.VideoCapture(str(video_root / f"{vid}.mp4"))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or count <= 0:
        # The B3 row is authoritative for the frame axis.  This fallback is
        # only for a missing metadata file and keeps seconds approximately
        # aligned rather than fabricating a frame count.
        return 30.0, n
    if count != n:
        raise ValueError(f"{vid}: B3 frame count {n} != decoded count {count}")
    return fps, count


def rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / max(len(x) - 1, 1)


def choose(row: dict, intervals: list[tuple[float, float]], fps: float) -> tuple[list[int], bool]:
    preds = row["predictions"]
    ids = np.asarray([int(p["frame"]) for p in preds], dtype=np.int64)
    if len(ids) != len(set(ids.tolist())) or not np.all(ids == np.sort(ids)):
        raise ValueError(f"{row['video_id']}: full-frame B3 row is not sorted/unique")
    n = len(ids)
    k = int(round(KEEP * n))
    center = 1.0 - np.abs(ids - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
    valid = bool(intervals)
    inside = np.zeros(n, dtype=np.float64)
    if valid:
        t = ids.astype(np.float64) / max(fps, 1e-9)
        for a, b in intervals:
            inside[(t >= a) & (t <= b)] = 1.0
        # A malformed or wildly out-of-range response should not discard the
        # safe temporal parent.  Empty intersection means fallback.
        valid = bool(inside.any())
    score = inside * 2.0 + rank01(center) * 1e-3
    chosen = np.sort(np.argsort(-score, kind="mergesort")[:k]) if valid else np.arange((n - k) // 2, (n - k) // 2 + k)
    return [int(ids[i]) for i in chosen], not valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, nargs="+", required=True,
                    help="TimeLens2 JSON outputs, each containing results")
    ap.add_argument("--b3-allframes", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, default=None,
                    help="optional mounted videos; metadata is preferred")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="official intake index JSONL with fps/frame_count")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    responses = {}
    raw_hashes = {}
    for p in args.raw:
        raw_hashes[str(p)] = sha256(p)
        doc = json.loads(p.read_text(encoding="utf-8"))
        for item in doc.get("results", []):
            responses[str(item["video_id"])] = parse_intervals(item.get("response", ""))
    rows = read_jsonl(args.b3_allframes)
    metadata = None
    if args.metadata is not None:
        metadata = {str(x["video_id"]): x for x in read_jsonl(args.metadata)}
    if len(rows) != 426 or len({str(x["video_id"]) for x in rows}) != 426:
        raise ValueError("expected 426 unique full-frame B3 rows")
    out, details = [], []
    total = fallback = 0
    for row in sorted(rows, key=lambda x: int(str(x["video_id"]))):
        vid = str(row["video_id"])
        fps, _ = frame_meta(args.video_root, vid, len(row["predictions"]), metadata)
        spans = responses.get(vid, [])
        chosen, used_fallback = choose(row, spans, fps)
        by_frame = {int(p["frame"]): p for p in row["predictions"]}
        q = dict(row)
        q["predictions"] = [by_frame[f] for f in chosen]
        q["model_size_mb"] = PARAMS_B * 1000.0
        q["model_params_b"] = PARAMS_B
        out.append(q)
        total += len(chosen)
        fallback += int(used_fallback)
        details.append({"video_id": vid, "n_full": len(row["predictions"]),
                        "n_keep": len(chosen), "fps": fps,
                        "intervals": spans, "fallback": used_fallback})

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    with pred.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    name = args.output.name
    zp = args.output / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zp) as z:
            if z.namelist() != ["predictions.jsonl"]:
                raise ValueError("unexpected ZIP members")
            z.extractall(td)
        if sha256(Path(td) / "predictions.jsonl") != sha256(pred):
            raise ValueError("ZIP roundtrip changed predictions")
    manifest = {
        "submission_id": name, "status": "PACKAGED_NOT_UPLOADED", "uploaded": False,
        "official_platform_score": None, "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "mechanism": "TimeLens2-8B fixed generic-highlight intervals converted to temporal membership; V9 vision-only B3 spatial boxes",
        "changed_factor": "temporal frame set only; spatial boxes copied from full-frame B3",
        "keep_rule": "round(0.85 * full original frame count)",
        "video_count": len(out), "prediction_count": total,
        "fallback_videos": fallback, "total_parameters": PARAMS,
        "model_params_b": PARAMS_B, "model_size_mb": PARAMS_B * 1000,
        "eligibility": "PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED",
        "eligibility_note": "The V9 vision-only spatial predictor is a complete audited 87,407,729-parameter artifact. This package uses its full-frame predictions directly; it does not combine the 450M full LFM runtime with TimeLens2.",
        "parameter_components": [
            {"name": "TimeLens2-8B", "parameters": 8767123696},
            {"name": "LFM2.5-VL SigLIP2 NaFlex vision tower", "parameters": 85843200},
            {"name": "V8 B3 candidate utility head", "parameters": 1511425},
            {"name": "YuNet", "parameters": 53104},
        ],
        "predictions_sha256": sha256(pred), "zip_sha256": sha256(zp),
        "zip_bytes": zp.stat().st_size, "raw_response_sha256": raw_hashes,
        "spatial_boxes_source": "LFM_V9_B3_VISIONONLY_SEMIFINAL / full-frame B3",
        "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zp.name + ".sha256")).write_text(sha256(zp) + "  " + zp.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
