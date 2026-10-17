#!/usr/bin/env python3
"""Package the vision-only V9 B3 spatial predictor with R26 centre-85 time.

This is a spatial-control candidate: it uses no official labels and changes
the framewise crop generator while keeping the exact centre ``round(.85*N)``
mask.  The V9 prediction file must contain all original frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

PARAMS = 87_407_729


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = rows(args.input)
    if len(source) != 426 or len({str(x["video_id"]) for x in source}) != 426:
        raise ValueError("expected 426 unique V9 rows")
    out = []
    count = 0
    per_video = []
    for row in sorted(source, key=lambda x: int(str(x["video_id"]))):
        pred = row["predictions"]
        ids = [int(x["frame"]) for x in pred]
        if ids != sorted(set(ids)):
            raise ValueError(f"{row['video_id']}: frames must be sorted and unique")
        n = len(pred)
        k = int(round(.85 * n))
        lo = (n - k) // 2
        chosen = pred[lo:lo + k]
        q = dict(row)
        q["predictions"] = chosen
        q["model_size_mb"] = PARAMS / 1e6
        q["model_params_b"] = PARAMS / 1e9
        out.append(q)
        count += k
        per_video.append({"video_id": str(row["video_id"]), "n_full": n,
                          "n_keep": k, "lo": lo, "hi": lo + k})

    args.output.mkdir(parents=True, exist_ok=False)
    pred_path = args.output / "predictions.jsonl"
    with pred_path.open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    zip_path = args.output / f"{args.output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            if z.namelist() != ["predictions.jsonl"]:
                raise ValueError("unexpected ZIP members")
            z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred_path):
            raise ValueError("ZIP roundtrip mismatch")
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "mechanism": "R26 exact centre-85 temporal mask with V9 vision-only B3 spatial boxes",
        "changed_factor": "spatial crop runtime/predictions only; exact temporal mask retained",
        "keep_rule": "round(0.85 * full original frame count), start=(N-K)//2",
        "video_count": len(out), "prediction_count": count,
        "total_parameters": PARAMS, "model_params_b": PARAMS / 1e9,
        "model_size_mb": PARAMS / 1e6,
        "parameter_components": [
            {"name": "LFM2.5-VL SigLIP2 NaFlex vision tower", "parameters": 85843200},
            {"name": "V8 B3 candidate utility head", "parameters": 1511425},
            {"name": "YuNet", "parameters": 53104},
        ],
        "source_predictions_sha256": sha(args.input),
        "predictions_sha256": sha(pred_path),
        "zip_sha256": sha(zip_path), "zip_bytes": zip_path.stat().st_size,
        "per_video": per_video,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / f"{zip_path.name}.sha256").write_text(sha(zip_path) + "  " + zip_path.name + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
