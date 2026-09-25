#!/usr/bin/env python3
"""Extract frozen timm DeiT-S/16 embeddings with the pinned video protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aic.features import extract_video_cache


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="headless DeiT-S state dict")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True, help="cache directory")
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    import timm

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    result = []
    for record in records:
        video_id = str(record["video_id"])
        cache = output / f"{video_id}.npz"
        metadata = {"video_id": video_id, "split": record.get("split"),
                    "source_group": record.get("source_group"),
                    "backbone": "deit_small_patch16_224_imagenet1k",
                    "feature_dim": 384,
                    "feature_protocol": "letterbox224_imagenet_norm"}
        info = extract_video_cache(model, record.get("video_path", record["path"]), cache,
                                   record.get("labels_path"), device=args.device,
                                   sample_fps=args.sample_fps, batch_size=args.batch_size,
                                   metadata=metadata)
        row = dict(record); row.update({"path": str(cache), "feature_dim": 384,
                                        "backbone": "deit_small_patch16_224",
                                        "cache_frames": info["frames"]})
        result.append(row)
        print(json.dumps(info, ensure_ascii=False), flush=True)
    Path(args.output_manifest).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in result) + "\n",
        encoding="utf-8")
    print(json.dumps({"count": len(result), "output": str(args.output),
                      "output_manifest": args.output_manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
