#!/usr/bin/env python3
"""Extract DINOv3 CLS features for the existing PHD2 fragment frame cache.

The remote PHD2 pool already contains decoded JPEG anchors.  This script keeps
their filenames as the time contract and derives weak union/soft-vote labels
from the original train.json without treating PHD2 testing.csv as training.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_runtime import encode_rgb, load_model


def load_annotations(path: Path):
    raw = json.loads(path.read_text())
    out = {}
    for src, users in raw.items():
        intervals = []
        for user, segs in users.items():
            for seg in segs:
                t0, t1 = float(seg["t0"]), float(seg["t1"])
                if t1 > t0:
                    intervals.append((user, t0, t1))
        out[src] = intervals
    return out


def labels(times: np.ndarray, intervals):
    users = sorted({u for u, _, _ in intervals})
    union = np.zeros(len(times), np.float32)
    votes = np.zeros(len(times), np.float32)
    for i, t in enumerate(times):
        n = sum(t0 <= float(t) < t1 for _, t0, t1 in intervals)
        votes[i] = n / max(1, len(users))
        union[i] = float(n > 0)
    return union, votes, len(users)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dinov3-repo", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batch-fragments", type=int, default=32,
                    help="number of 8-frame fragments per encoder forward")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (args.pool / "index_clean.jsonl").read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    ann = load_annotations(args.annotations)
    args.out.mkdir(parents=True, exist_ok=True)
    model, model_meta = load_model(args.dinov3_repo, args.weights, args.device, args.dtype)
    manifest = []
    started = time.time()
    pending = []
    decode_pool = ThreadPoolExecutor(max_workers=16)
    completed = 0

    def flush_pending():
        nonlocal completed
        if not pending:
            return
        all_frames = list(decode_pool.map(lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8),
                                          [p for item in pending for p in item["paths"]]))
        encoded = encode_rgb(model, all_frames, args.device, args.batch_size, 224, "letterbox", False)
        cursor = 0
        for item in pending:
            n = len(item["paths"])
            pooled = encoded["pooled"][cursor:cursor + n]
            cursor += n
            union, soft, n_users = labels(item["times"], ann.get(item["src"], []))
            target = args.out / f"{item['vid']}.npz"
            np.savez_compressed(target, pooled=pooled.astype(np.float16),
                                timestamps=item["times"], union=union, soft=soft,
                                mask=np.ones(len(item["times"]), dtype=np.bool_),
                                user_count=np.asarray(n_users, dtype=np.int32),
                                metadata_json=np.asarray(json.dumps({"video_id": item["vid"], "src": item["src"],
                                    "model": model_meta, "label_semantics": {
                                        "union": "any train user selected the time",
                                        "soft": "fraction of train users selecting the time",
                                        "unobserved_warning": "absence is not a confirmed negative"}}, sort_keys=True)))
            manifest.append({"video_id": item["vid"], "src": item["src"], "path": str(target),
                             "frames": n, "users": n_users})
            completed += 1
        pending.clear()
        print(json.dumps({"rows": completed, "total": len(rows),
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)

    for no, row in enumerate(rows, 1):
        vid, src = str(row["video_id"]), str(row["src"])
        target = args.out / f"{vid}.npz"
        if target.exists() and not args.overwrite:
            manifest.append({"video_id": vid, "src": src, "path": str(target), "frames": 8})
            completed += 1
            continue
        frame_dir = args.pool / "frames" / vid
        paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: float(p.stem))
        if not paths:
            raise FileNotFoundError(frame_dir)
        # Decode in the shared pool during the next flush; keeping paths here
        # avoids serial PIL work in the producer loop.
        frames = None
        times = np.asarray([float(p.stem) for p in paths], dtype=np.float64)
        pending.append({"vid": vid, "src": src, "paths": paths, "frames": frames, "times": times})
        if len(pending) >= args.batch_fragments:
            flush_pending()
    flush_pending()
    decode_pool.shutdown(wait=True)
    (args.out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest) + "\n")
    (args.out / "run_meta.json").write_text(json.dumps({"model": model_meta, "rows": len(rows),
        "pool": str(args.pool), "annotations": str(args.annotations)}, indent=2) + "\n")
    print(json.dumps({"status": "ok", "rows": len(rows), "elapsed_s": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
