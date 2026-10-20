#!/usr/bin/env python3
"""Prepare a frozen Valen confirmation table for RetargetVid 601--700.

The V9 feature shards are produced by ``r70_spatial_utility_pilot.py`` from
the public native-crop tables.  This script joins those rows back to their
source frame, computes the same frozen ``topsoft`` head and preserves the
129-window / four-candidate contract used by the Valen SFT run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--rv-frames", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(args.feature_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from r70_spatial_utility_pilot import Head, boxes
    from aic.highlight_decisions import diverse_candidates

    # Reconstruct exactly the iteration order used by the feature extractor.
    units = pickle.loads((Path(args.tables) / "rv_confirm2.pkl").read_bytes())
    metadata = []
    for unit in units:
        for jj, j in enumerate(unit["J"]):
            kf = int(unit["keys"][int(j)])
            gf = {int(f): i for i, f in enumerate(unit["gf"])}
            if kf not in gf:
                continue
            metadata.append((unit, jj, kf, gf[kf]))

    import torch
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    head = Head.make(torch).to(args.device).float().eval()
    head.load_state_dict(ck["state_dict"])
    mu = torch.as_tensor(ck["mu"], device=args.device).float()
    sd = torch.as_tensor(ck["sd"], device=args.device).float()

    records = []
    n = 0
    for shard_path in sorted((root / "shards").glob("shard_*.npz")):
        z = np.load(shard_path, allow_pickle=True)
        x = z["x"].astype(np.float32)
        y = z["y"].astype(np.float32)
        source = z["source"].astype(str)
        ratio = z["ratio"].astype(str)
        W = z["W"].astype(float)
        H = z["H"].astype(float)
        offsets = z["offsets"].astype(np.float32)
        with torch.inference_mode():
            scores = []
            for st in range(0, len(x), 64):
                q = torch.from_numpy(x[st:st + 64]).to(args.device).float()
                scores.append(head((q - mu) / sd).cpu().numpy())
        scores = np.concatenate(scores, axis=0) if scores else np.empty((0, 129), np.float32)
        for j in range(len(x)):
            unit, jj, kf, gf = metadata[n]
            if source[j] != f"rv:{unit['vid']}" or ratio[j] != "-".join(str(int(v)) for v in unit["ratio"]):
                raise RuntimeError(f"feature/table order mismatch at {n}: {source[j]} {ratio[j]} {unit['vid']} {unit['ratio']}")
            cand = [int(v) for v in diverse_candidates(scores[j], 4)]
            bb = boxes(offsets[j], float(W[j]), float(H[j]), unit["ratio"])
            frame_gt = unit["gt"][:, gf, :].astype(float).tolist()
            image = Path(args.rv_frames) / "d1" / str(unit["vid"]) / f"{kf}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            records.append({
                "row_index": n,
                "key": f"rv:{unit['vid']}:{kf}",
                "source": f"rv:{unit['vid']}",
                "split": "confirm2_exposed",
                "name": "rv_confirm2",
                "ratio": [float(v) for v in unit["ratio"]],
                "W": float(W[j]), "H": float(H[j]), "image": str(image),
                "head_index": int(np.argmax(scores[j])),
                "candidates4": cand,
                "candidates8": [int(v) for v in diverse_candidates(scores[j], 8)],
                "boxes": bb.tolist(),
                "iou": y[j].astype(float).tolist(),
                "gt": frame_gt,
            })
            n += 1
    if n != len(metadata):
        raise RuntimeError(f"feature rows {n} != table rows {len(metadata)}")
    (out / "records.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    # Confirmation uses all legal keyframes; source is the independent unit.
    (out / "pilot_indices.json").write_text(json.dumps(list(range(len(records)))) + "\n")
    manifest = {
        "records": len(records),
        "sources": len({r["source"] for r in records}),
        "ratios": {str(r): sum(tuple(x["ratio"]) == tuple(r) for x in records) for r in ([1.0, 3.0], [3.0, 1.0])},
        "feature_root": str(root),
        "feature_shards": [str(p) for p in sorted((root / "shards").glob("shard_*.npz"))],
        "feature_root_sha256": hashlib.sha256((root / "meta.json").read_bytes()).hexdigest() if (root / "meta.json").exists() else None,
        "head": str(args.head),
        "head_sha256": hashlib.sha256(Path(args.head).read_bytes()).hexdigest(),
        "tables": str(args.tables),
        "contract": "R70 V9 vision features + frozen topsoft head + 129 legal windows + four candidates",
        "official_test_used": False,
        "historical_status": "exposed_retargetvid_601_700; previously used by legacy diagnostics/training branches",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
