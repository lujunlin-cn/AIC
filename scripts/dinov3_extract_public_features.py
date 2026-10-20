#!/usr/bin/env python3
"""Extract DINOv3 global/patch features from an explicit public-video manifest.

Manifest JSONL records must contain ``video_id`` and ``video_path``.  Optional
``frame_indices`` selects exact decoded frame numbers; otherwise ``every_n``
controls sampling.  The extractor always stores decoded frame indices and PTS,
so a later temporal head cannot silently confuse sampled positions with the
original frame axis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import av
import numpy as np

from dinov3_runtime import encode_rgb, jsonable_meta, load_model


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    for row in rows:
        for key in ("video_id", "video_path"):
            if key not in row:
                raise ValueError(f"manifest row missing {key}: {row}")
    return rows


def decode_selected(path: Path, frame_indices: set[int] | None, every_n: int,
                    max_frames: int) -> tuple[list[np.ndarray], list[int], list[float], dict[str, Any]]:
    frames, indices, pts = [], [], []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise RuntimeError(f"no video stream: {path}")
        nominal_fps = float(stream.average_rate) if stream.average_rate else None
        for idx, frame in enumerate(container.decode(stream)):
            choose = (idx in frame_indices) if frame_indices is not None else (idx % every_n == 0)
            if choose:
                frames.append(frame.to_ndarray(format="rgb24"))
                indices.append(idx)
                if frame.pts is not None and frame.time_base is not None:
                    pts.append(float(frame.pts * frame.time_base))
                elif nominal_fps:
                    pts.append(float(idx / nominal_fps))
                else:
                    pts.append(float("nan"))
                if max_frames and len(frames) >= max_frames:
                    break
            if frame_indices is not None and indices and idx > max(frame_indices):
                break
    if not frames:
        raise RuntimeError(f"sampling selected no frames: {path}")
    probe = frames[0]
    meta = {"width": int(probe.shape[1]), "height": int(probe.shape[0]),
            "decoded_count_at_least": int(indices[-1] + 1),
            "nominal_fps": nominal_fps}
    return frames, indices, pts, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dinov3-repo", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--resize-mode", choices=["square", "letterbox"], default="letterbox")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-patch", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)
    model, model_meta = load_model(args.dinov3_repo, args.weights, args.device, args.dtype)
    rows = read_manifest(args.manifest)
    run_meta = {"manifest": str(args.manifest.resolve()), "model": model_meta,
                "image_size": args.image_size, "resize_mode": args.resize_mode,
                "batch_size": args.batch_size, "every_n": args.every_n,
                "save_patch": args.save_patch}
    (args.out / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    started = time.time()
    for no, row in enumerate(rows, 1):
        vid = str(row["video_id"])
        target = args.out / f"{vid}.npz"
        if target.exists() and not args.overwrite:
            print(json.dumps({"video_id": vid, "status": "skip_exists"}), flush=True)
            continue
        video_path = Path(row["video_path"]).expanduser()
        selected = row.get("frame_indices")
        frame_set = None if selected is None else {int(v) for v in selected}
        frames, indices, pts, video_meta = decode_selected(video_path, frame_set, args.every_n, args.max_frames)
        enc = encode_rgb(model, frames, device=args.device, batch_size=args.batch_size,
                         size=args.image_size, resize_mode=args.resize_mode,
                         save_patch=args.save_patch)
        metadata = {"video_id": vid, "video_path": str(video_path.resolve()),
                    "video_metadata": video_meta, "frame_selection": row.get("frame_indices", f"every_{args.every_n}"),
                    "model": model_meta, "image_size": args.image_size,
                    "resize_mode": args.resize_mode, "feature_contract": "DINOv3 official CLS + patch tokens",
                    "source_group": row.get("source_group", vid), "label_source": row.get("label_source")}
        payload = {"pooled": enc["pooled"].astype(np.float16),
                   "storage": enc["storage"].astype(np.float16),
                   "frame_indices": np.asarray(indices, dtype=np.int64),
                   "timestamps": np.asarray(pts, dtype=np.float64),
                   "metadata_json": np.asarray(jsonable_meta(metadata))}
        if args.save_patch:
            payload["patch_tokens"] = enc["patch_tokens"].astype(np.float16)
            payload["grid_shape"] = np.asarray(enc["grid_shape"], dtype=np.int64)
        np.savez_compressed(target, **payload)
        print(json.dumps({"video_id": vid, "status": "ok", "frames": len(indices),
                          "patch": bool(args.save_patch), "elapsed_s": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
