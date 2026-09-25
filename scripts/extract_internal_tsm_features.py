#!/usr/bin/env python3
"""Extract ResNet features with a real intermediate feature-map TSM.

The historical A1 route shifts the final ``[T,D]`` embedding cache.  This
extractor instead runs ResNet stem+layer1 per temporal chunk, applies
``temporal_shift_feature_map`` to the resulting ``[T,64,H,W]`` map (with one
frame of context on either side), then runs layer2..layer4 and global average
pooling.  The output is an ordinary feature cache so the temporal head can be
trained with the same protocol as A0, but the cache metadata records the
encoder version and source frame alignment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.models import resnet18

from aic.features import _letterbox, align_labels, save_feature_cache
from aic.models import temporal_shift_feature_map
from aic.video import iter_sampled_frames


def load_backbone(path: str | None, device: torch.device) -> torch.nn.Module:
    net = resnet18(weights=None)
    net.fc = torch.nn.Identity()
    if path:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("backbone."): v for k, v in state.items()}
        net.load_state_dict(state, strict=False)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def encode_internal_tsm(net: torch.nn.Module, images: np.ndarray,
                        device: torch.device, chunk_size: int = 128) -> np.ndarray:
    """Encode sampled RGB images with canonical intermediate TSM.

    A one-frame context is included around each chunk because TSM moves one
    channel fold from the preceding/following timestep.  We retain only the
    chunk centre, yielding the same result as shifting the complete sequence
    while bounding activation memory.
    """
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError(f"Expected RGB [T,H,W,3], got {images.shape}")
    total = len(images)
    outputs: list[np.ndarray] = []
    for start in range(0, total, int(chunk_size)):
        stop = min(total, start + int(chunk_size))
        context_start, context_stop = max(0, start - 1), min(total, stop + 1)
        x = _letterbox(images[context_start:context_stop]).to(device)
        # torchvision ResNet up through layer1; preserve temporal ordering.
        z = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        z = net.layer1(z)
        nctx = context_stop - context_start
        fmap = z.reshape(1, nctx, z.shape[1], z.shape[2], z.shape[3])
        shifted = temporal_shift_feature_map(fmap)
        z = shifted.reshape(nctx, *shifted.shape[2:])
        z = net.layer2(z)
        z = net.layer3(z)
        z = net.layer4(z)
        z = net.avgpool(z).flatten(1)
        keep_start, keep_stop = start - context_start, stop - context_start
        # Keep the slice explicit to avoid relying on chunk boundaries when
        # the first/last context is missing.
        outputs.append(z[keep_start:keep_stop].float().cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    if result.shape != (total, 512) or not np.isfinite(result).all():
        raise FloatingPointError(f"Invalid internal TSM features: {result.shape}")
    return result


def extract_record(record: dict[str, Any], output: Path, net: torch.nn.Module,
                   device: torch.device, sample_fps: float,
                   chunk_size: int) -> dict[str, Any]:
    sampled = list(iter_sampled_frames(record.get("video_path", record["path"]),
                                       sample_fps=sample_fps, size=224))
    if not sampled:
        raise ValueError(f"No sampled frames: {record['video_id']}")
    frame_indices = np.asarray([x[0] for x in sampled], dtype=np.int64)
    timestamps = np.asarray([x[1] for x in sampled], dtype=np.float64)
    images = np.stack([x[2] for x in sampled], axis=0)
    features = encode_internal_tsm(net, images, device, chunk_size)
    label_data = None
    label_path = record.get("labels_path", record.get("annotation_path"))
    if label_path and Path(label_path).exists():
        with np.load(label_path, allow_pickle=False) as d:
            label_data = {k: np.asarray(d[k]) for k in d.files}
    labels, mask = align_labels(label_data, frame_indices)
    metadata = {
        "video_id": record.get("video_id"),
        "split": record.get("split"),
        "source_group": record.get("source_group"),
        "video_path": record.get("video_path", record.get("path")),
        "sample_fps": float(sample_fps),
        "backbone": "ResNet18_without_classifier",
        "encoder_variant": "canonical_internal_tsm_layer1_fold8_v1",
        "tsm_location": "after_resnet_layer1_before_layer2",
        "tsm_fold_div": 8,
        "chunk_size": int(chunk_size),
        "label_source": label_path,
    }
    save_feature_cache(output, features, frame_indices, timestamps, labels, mask, metadata)
    return {"video_id": record["video_id"], "path": str(output),
            "frames": int(len(features)), "feature_dim": int(features.shape[1]),
            "labeled_frames": int(mask.sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--backbone-state", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device if args.device != "auto" or torch.cuda.is_available() else "cpu")
    records = [json.loads(x) for x in Path(args.manifest).read_text().splitlines()
               if x.strip() and not x.lstrip().startswith("#")]
    if args.limit:
        records = records[:args.limit]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    net = load_backbone(args.backbone_state, device)
    result_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        output = out_dir / f"{record['video_id']}.npz"
        if output.exists() and not args.overwrite:
            result_records.append({**record, "path": str(output),
                                   "cache_frames": int(np.load(output, allow_pickle=False)["features"].shape[0]),
                                   "feature_dim": 512})
            continue
        print(f"[{index}/{len(records)}] {record['video_id']}", flush=True)
        result = extract_record(record, output, net, device, args.sample_fps, args.chunk_size)
        result_records.append({**record, "path": str(output),
                               "video_path": record.get("video_path", record.get("path")),
                               "cache_frames": result["frames"],
                               "feature_dim": result["feature_dim"]})
    Path(args.output_manifest).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in result_records) + "\n",
        encoding="utf-8")
    print(json.dumps({"count": len(result_records), "output_dir": str(out_dir),
                      "encoder_variant": "canonical_internal_tsm_layer1_fold8_v1",
                      "device": str(device)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
