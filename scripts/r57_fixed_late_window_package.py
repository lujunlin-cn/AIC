#!/usr/bin/env python3
"""Package a fixed +2 second late-window control using audited V9 boxes.

The public source probe showed that R54's movement is close to a late-window
prior.  This is one predeclared offset control, not an offset sweep.  It keeps
the exact 85 percent budget and changes no spatial artifact except using V9.
"""
from __future__ import annotations

import argparse, hashlib, json, tempfile, zipfile
from pathlib import Path
import numpy as np

V9_PARAMS = 87_407_729
SHIFT_SECONDS = 2.0

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def rows(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def key(x):
    return int(str(x["video_id"])) if str(x["video_id"]).isdigit() else str(x["video_id"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    idx = {str(x["video_id"]): x for x in rows(args.index)}
    parent = rows(args.parent)
    out, details, total = [], [], 0
    for row in sorted(parent, key=key):
        vid = str(row["video_id"]); meta = idx[vid]; allp = row["predictions"]
        n = int(meta["frame_count"]); fps = float(meta["fps"])
        if len(allp) != n: raise ValueError(f"{vid}: {len(allp)} != {n}")
        k = int(round(.85 * n)); base = (n - k) // 2
        start = min(max(base + int(round(SHIFT_SECONDS * fps)), 0), n - k)
        chosen = allp[start:start+k]
        q = dict(row); q["predictions"] = chosen
        q["model_size_mb"] = V9_PARAMS / 1e6; q["model_params_b"] = V9_PARAMS / 1e9
        out.append(q); total += len(chosen)
        details.append({"video_id": vid, "n_full": n, "n_keep": k, "center_start": base,
                        "late_start": start, "fps": fps, "shift_seconds": SHIFT_SECONDS})
    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(q, ensure_ascii=False, separators=(",", ":")) + "\n" for q in out))
    zpath = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z: z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as z: z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred): raise RuntimeError("ZIP round trip mismatch")
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "official_platform_score": None,
        "parent_submission": "R26_POS_CENTER_EXACT085_B3",
        "mechanism": "fixed +2.0 second late shift of exact centre-85 contiguous window",
        "spatial_mode": "v9_b3_artifact",
        "eligibility": "PARAMETER_COMPLIANT_IF_V9_ARTIFACT_ACCEPTED",
        "parameter_components": [{"name": "V9 vision-only SigLIP2 + B3 head + YuNet", "parameters": V9_PARAMS}],
        "total_parameters": V9_PARAMS, "model_params_b": V9_PARAMS / 1e9,
        "shift_seconds": SHIFT_SECONDS, "video_count": len(out), "prediction_count": total,
        "prediction_sha256": sha(pred), "zip_sha256": sha(zpath), "zip_bytes": zpath.stat().st_size,
        "parent_sha256": sha(args.parent), "index_sha256": sha(args.index), "per_video": details,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (args.output / (zpath.name + ".sha256")).write_text(sha(zpath) + "  " + zpath.name + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
