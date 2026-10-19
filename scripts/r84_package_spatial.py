#!/usr/bin/env python3
"""Finalize an R84 point directory into a checked spatial submission package.

The point inference is performed by r84_official_points.py.  This wrapper
reuses the audited R70 exporter for frame-axis reconstruction, then rewrites
the package metadata with the actual R84 parameter inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


PARAMS = 8_853_590_257


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--obs-cache", type=Path, required=True)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    tmp = args.output.parent / (args.output.name + ".r70tmp")
    subprocess.run([
        "python", "scripts/r70_package_spatial.py", "--parent", str(args.parent),
        "--index", str(args.index), "--obs-cache", str(args.obs_cache),
        "--points", str(args.points), "--output", str(tmp)
    ], check=True)
    args.output.mkdir(parents=True)
    src_pred = tmp / "predictions.jsonl"
    pred = args.output / "predictions.jsonl"
    with src_pred.open() as fi, pred.open("w") as fo:
        for line in fi:
            row = json.loads(line)
            row["model_params_b"] = PARAMS / 1e9
            row["model_size_mb"] = PARAMS / 1e6
            fo.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    zip_path = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred, "predictions.jsonl")
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED_R84_COORD_CONSISTENCY",
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "small native-crop utility head with coordinate-consistency loss; R54 time mask fixed; only spatial boxes changed",
        "parameter_components": [
            {"name": "TimeLens2-8B", "parameters": 8_767_123_696},
            {"name": "V9 vision tower", "parameters": 85_843_200},
            {"name": "R84 coordinate-consistency head", "parameters": 623_361},
        ],
        "total_parameters": PARAMS,
        "model_params_b": PARAMS / 1e9,
        "video_count": 426,
        "prediction_count": sum(len(json.loads(x)["predictions"]) for x in pred.open()),
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "official_test_labels_used": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / (zip_path.name + ".sha256")).write_text(f"{sha(zip_path)}  {zip_path.name}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
