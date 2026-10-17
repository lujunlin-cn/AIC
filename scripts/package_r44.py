#!/usr/bin/env python3
"""Package and audit the R44 temporal residual candidate.

R44 is intentionally a strict R26 child: only the selected frame set may
change.  Every selected frame must use the byte-identical B3 box from the
full-frame parent, and the per-video budget is round(.85 * N).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from scripts.independent_submission_check import check


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--full-parent", type=Path, required=True)
    ap.add_argument("--r26", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    candidate = read_jsonl(args.candidate)
    full = {str(r["video_id"]): r for r in read_jsonl(args.full_parent)}
    r26 = {str(r["video_id"]): r for r in read_jsonl(args.r26)}
    if set(full) != set(r26) or set(full) != {str(r["video_id"]) for r in candidate}:
        raise SystemExit("video coverage mismatch")
    out_rows = []
    changed_videos = 0
    total = 0
    changed_frames = 0
    per_video = []
    for row in candidate:
        vid = str(row["video_id"])
        parent = full[vid]
        old = r26[vid]
        by_frame = {int(p["frame"]): p for p in parent["predictions"]}
        n = len(parent["predictions"])
        expected = int(round(0.85 * n))
        preds = row.get("predictions", [])
        frames = [int(p["frame"]) for p in preds]
        if len(preds) != expected:
            raise SystemExit(f"{vid}: budget {len(preds)} != round(.85*{n})={expected}")
        if frames != sorted(set(frames)):
            raise SystemExit(f"{vid}: frames are not sorted and unique")
        for p in preds:
            f = int(p["frame"])
            if f not in by_frame:
                raise SystemExit(f"{vid}: frame {f} missing from full parent")
            if p.get("bboxes") != by_frame[f].get("bboxes"):
                raise SystemExit(f"{vid}: B3 box changed at frame {f}")
        old_frames = [int(p["frame"]) for p in old["predictions"]]
        delta = sum(a != b for a, b in zip(frames, old_frames))
        if frames != old_frames:
            changed_videos += 1
            changed_frames += delta
        clean = {k: v for k, v in row.items() if k not in {"model_size_mb", "model_params_b"}}
        clean["model_size_mb"] = 450.283377
        clean["model_params_b"] = 0.450283377
        out_rows.append(clean)
        total += len(preds)
        per_video.append({"video_id": vid, "n_full": n, "n_keep": len(preds),
                          "changed_vs_r26": frames != old_frames, "frame_position_differences": delta})

    args.output.mkdir(parents=True, exist_ok=False)
    pred_path = args.output / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    validation = check(args.index, pred_path, require_size=True)
    zip_path = args.output / "R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            if z.namelist() != ["predictions.jsonl"]:
                raise SystemExit("unexpected ZIP members")
            z.extractall(td)
        roundtrip = check(args.index, Path(td) / "predictions.jsonl", require_size=True)
        if sha256(Path(td) / "predictions.jsonl") != sha256(pred_path):
            raise SystemExit("ZIP roundtrip changed predictions")
    manifest = {
        "submission_id": "R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3",
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "parent_official_score": 36.04,
        "mechanism": "source-held-out human-label calibrated local-state + audited IV2 residual",
        "changed_factor": "temporal frame ranking only; B3 spatial boxes copied byte-for-byte",
        "keep_rule": "round(0.85 * full original frame count)",
        "video_count": len(out_rows),
        "prediction_count": total,
        "changed_videos": changed_videos,
        "changed_frame_positions": changed_frames,
        "total_parameters": 450283377,
        "model_params_b": 0.450283377,
        "model_size_mb": 450.283377,
        "human_label_rows": 540,
        "human_label_sources": 12,
        "eta": 0.20,
        "ridge_alpha": 10.0,
        "predictions_sha256": sha256(pred_path),
        "zip_sha256": sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "validator": validation,
        "unzip_validator": roundtrip,
        "per_video": per_video,
    }
    (args.output / "check.json").write_text(json.dumps(validation, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zip_path.name + ".sha256")).write_text(sha256(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
