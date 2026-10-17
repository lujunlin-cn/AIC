#!/usr/bin/env python3
"""Extract frozen InternVideo2 visual embeddings for R44.

This is an inference-only feature pass.  It uses the audited R43 loader and
the same four-frame, one-second observation contract, but saves the pooled
visual embedding instead of the prompt score.  Public labels are only used in
the subsequent small readout; official videos have no labels here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("r43_iv2_embeddings", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def anchors_for(cap: cv2.VideoCapture) -> tuple[np.ndarray, float, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = max((n - 1) / max(fps, 1e-6), 0.0)
    anchors = np.arange(0.0, duration + 1e-6, 1.0, dtype=np.float64)
    if len(anchors) == 0 or anchors[-1] < duration:
        anchors = np.r_[anchors, duration]
    return anchors, fps, n


def one_video(mod, model, path: Path, device: torch.device, dtype: torch.dtype,
              batch_size: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    anchors, fps, n = anchors_for(cap)
    if n <= 0 or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"bad metadata: {path}")
    offsets = (-1.5, -0.5, 0.5, 1.5)
    all_embeddings = []
    pending = []
    with torch.inference_mode():
        def flush():
            if not pending:
                return
            batch = torch.stack(pending, dim=0)
            emb = model.get_vid_feat(batch).reshape(len(pending), -1)
            all_embeddings.extend(emb.float().cpu().numpy())
            pending.clear()
        for sec in anchors:
            ids = [int(round((float(sec) + off) * fps)) for off in offsets]
            pending.append(torch.stack([
                mod._frame_tensor(cap, j, n, device, dtype) for j in ids
            ], dim=0))
            if len(pending) >= batch_size:
                flush()
        flush()
    cap.release()
    emb = np.asarray(all_embeddings, dtype=np.float32)
    if emb.shape[0] != len(anchors) or not np.isfinite(emb).all():
        raise RuntimeError(f"embedding contract failed for {path}: {emb.shape}")
    return {"embeddings": emb, "anchors": anchors, "fps": fps, "frame_count": n}


def ids_from_index(index: Path) -> list[str]:
    return [str(json.loads(line)["video_id"])
            for line in index.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r43-script", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--public-root", type=Path, required=True)
    ap.add_argument("--public-ids", type=Path, required=True)
    ap.add_argument("--official-root", type=Path, required=True)
    ap.add_argument("--official-index", type=Path, required=True)
    ap.add_argument("--public-out", type=Path, required=True)
    ap.add_argument("--official-out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--only", choices=["all", "public", "official"], default="all")
    args = ap.parse_args()
    mod = load_module(args.r43_script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, _module, params = mod._load_model(args.model_root, device, dtype)
    public_ids = [x.strip() for x in args.public_ids.read_text().splitlines() if x.strip()]
    official_ids = ids_from_index(args.official_index)
    jobs = []
    if args.only in {"all", "public"}:
        jobs += [("public", vid, args.public_root / f"{vid}.mp4", args.public_out)
                 for vid in public_ids]
    if args.only in {"all", "official"}:
        jobs += [("official", vid, args.official_root / f"{vid}.mp4", args.official_out)
                 for vid in official_ids]
    for pos, (kind, vid, path, out_dir) in enumerate(jobs):
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{vid}.npz"
        if dst.exists():
            print(json.dumps({"resumed": pos + 1, "total": len(jobs), "kind": kind, "video_id": vid}), flush=True)
            continue
        rec = one_video(mod, model, path, device, dtype, args.batch)
        np.savez_compressed(dst, **rec)
        print(json.dumps({"completed": pos + 1, "total": len(jobs), "kind": kind,
                          "video_id": vid, "anchors": int(len(rec["anchors"])),
                          "dim": int(rec["embeddings"].shape[1]), "params": params}), flush=True)
    print(json.dumps({"public": len(public_ids), "official": len(official_ids),
                      "model_params": params, "device": str(device)}))


if __name__ == "__main__":
    main()
