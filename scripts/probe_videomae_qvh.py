#!/usr/bin/env python3
"""VideoMAEv2 and matched image-encoder clip controls on QVH weak labels.

Not AIC F_video. Uses only train/val manifests. Version 2 fixes channel
normalization, required cache timestamps, and average ranks for tied targets.
All four encoders share clip boundaries, clip-mean labels and temporal training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch import nn

from aic.features import (FeatureCacheDataset, align_labels, collate_feature_batch,
                          load_feature_cache, save_feature_cache)
from aic.models import TemporalUNet
from aic.train import _spearman
from aic.video import letterbox_rgb

PROTOCOL = "qvh_clip16_stride4_2fps_mean_seed_v2"
TARGET_THRESHOLD = 0.5
PREDICTION_THRESHOLD = 0.5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_spans(length, clip_len, stride):
    if length < 1 or clip_len < 1 or stride < 1:
        raise ValueError("non-positive clip dimension")
    starts = list(range(0, max(1, length - clip_len + 1), stride))
    if length > clip_len and starts[-1] != length - clip_len:
        starts.append(length - clip_len)
    return [(s, min(s + clip_len, length)) for s in starts]


def normalize_video(x, mean, std):
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError("expected [B,C,T,H,W] with C=3")
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1, 1)
    return (x / 255.0 - mean) / std


def decode_sampled(path, sample_fps):
    frames, indices, timestamps = [], [], []
    next_time = 0.0
    with av.open(path) as container:
        stream = next(s for s in container.streams.video
                      if not (s.disposition & av.stream.Disposition.attached_pic))
        first = previous = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"missing PTS: {path}")
            absolute = float(frame.pts * frame.time_base)
            if first is None:
                first = absolute
            relative = absolute - first
            if previous is not None and relative <= previous:
                raise ValueError(f"non-increasing PTS: {path}")
            previous = relative
            if relative + 1e-9 < next_time:
                continue
            frames.append(letterbox_rgb(frame.to_ndarray(format="rgb24"), 224))
            indices.append(index)
            timestamps.append(relative)
            next_time = (np.floor((relative + 1e-9) * sample_fps) + 1) / sample_fps
    if not frames:
        raise ValueError(f"empty video: {path}")
    return np.stack(frames), np.asarray(indices), np.asarray(timestamps)


def encode_video(model, frames, device, spans, clip_len, batch_clips, mean, std):
    out = []
    with torch.inference_mode():
        for offset in range(0, len(spans), batch_clips):
            clips = []
            for start, stop in spans[offset:offset + batch_clips]:
                clip = frames[start:stop]
                if len(clip) < clip_len:
                    clip = np.concatenate([clip, np.repeat(clip[-1:], clip_len - len(clip), axis=0)])
                clips.append(clip)
            x = torch.from_numpy(np.stack(clips)).to(device=device, dtype=torch.float32)
            x = normalize_video(x.permute(0, 4, 1, 2, 3), mean, std)
            features = model(pixel_values=x).float()
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise ValueError("invalid VideoMAE features")
            out.append(features.cpu().numpy())
    return np.concatenate(out)


def extract_all(model, manifests, output, device, args, mean, std):
    variants = {"videomaev2": None, "deits_clip_mean": "features",
                "a0_clip_mean": "features_a0", "vitb_clip_mean": "features_b0"}
    all_manifests = {v: {} for v in variants}
    for split, manifest in manifests.items():
        records = {v: [] for v in variants}
        for index, row in enumerate(rows(manifest)):
            frames, indices, times = decode_sampled(row["video_path"], args.sample_fps)
            with np.load(row["labels_path"], allow_pickle=False) as data:
                label_data = {k: np.array(data[k]) for k in data.files}
            labels, mask = align_labels(label_data, indices)
            if not mask.all():
                raise ValueError(f"unmatched sampled labels in {row['video_id']}")
            spans = clip_spans(len(frames), args.clip_len, args.stride)
            centers = np.asarray([s + (e - s) // 2 for s, e in spans])
            targets = np.asarray([labels[s:e].mean() for s, e in spans], dtype=np.float32)
            encoded = encode_video(model, frames, device, spans, args.clip_len,
                                   args.batch_clips, mean, std)
            for variant, directory in variants.items():
                if directory is None:
                    features = encoded
                else:
                    cache = load_feature_cache(args.control_root / directory / f"{row['video_id']}.npz")
                    np.testing.assert_array_equal(cache["frame_indices"], indices)
                    np.testing.assert_allclose(cache["timestamps"], times, atol=1e-7, rtol=0)
                    np.testing.assert_allclose(cache["labels"], labels, atol=1e-7, rtol=0)
                    if not cache["mask"].all():
                        raise ValueError("invalid control label mask")
                    features = np.stack([cache["features"][s:e].mean(axis=0) for s, e in spans])
                path = output / variant / split / f"{row['video_id']}.npz"
                metadata = {"feature_protocol": PROTOCOL, "backbone": variant,
                            "label_protocol": "qvh_seed_timeline_linear_v1",
                            "clip_mean_target": True, "source_label_sha256": sha256(row["labels_path"]),
                            "spatial_preprocessing": "letterbox224_imagenet_norm",
                            "official_aic_gt": False}
                save_feature_cache(path, features, indices[centers], times[centers],
                                   labels=targets, label_mask=np.ones(len(targets), dtype=bool),
                                   metadata=metadata)
                records[variant].append({**row, "path": str(path), "feature_protocol": PROTOCOL})
            print(json.dumps({"stage": "extract", "split": split, "index": index + 1,
                              "video_id": row["video_id"], "clips": len(spans)}), flush=True)
        for variant in variants:
            mpath = output / variant / f"{split}.jsonl"
            mpath.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records[variant]))
            all_manifests[variant][split] = mpath
    return all_manifests


def metrics(p, target, video_id):
    prediction, truth = p >= PREDICTION_THRESHOLD, target >= TARGET_THRESHOLD
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    denom = 2 * tp + fp + fn
    return {"video_id": video_id, "f1": 2 * tp / denom if denom else 1.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "spearman": _spearman(p, target), "empty_prediction": bool(not prediction.any()),
            "prediction_rate": float(prediction.mean()), "target_rate": float(truth.mean()),
            "score_quantiles": np.quantile(p, [0, .1, .5, .9, 1]).tolist()}


def fit(manifests, output, device, epochs, seed):
    seed_everything(seed)
    train_ds, val_ds = FeatureCacheDataset(manifests["train"]), FeatureCacheDataset(manifests["val"])
    train_items = [train_ds[i] for i in range(len(train_ds))]
    val_items = [val_ds[i] for i in range(len(val_ds))]
    dim = train_items[0]["features"].shape[1]
    head = TemporalUNet(dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, history = None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        head.train()
        losses = []
        for index in rng.permutation(len(train_items)):
            b = collate_feature_batch([train_items[index]])
            x, y = b["features"].to(device), b["labels"].to(device)
            z = head(x, lengths=b["lengths"].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(z, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        per_video, val_losses = [], []
        with torch.inference_mode():
            for item in val_items:
                b = collate_feature_batch([item])
                x, y = b["features"].to(device), b["labels"].to(device)
                z = head(x, lengths=b["lengths"].to(device))
                val_losses.append(float(nn.functional.binary_cross_entropy_with_logits(z, y).cpu()))
                per_video.append(metrics(z[0].sigmoid().cpu().numpy(), y[0].cpu().numpy(), item["video_id"]))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "val_loss": float(np.mean(val_losses)),
                  "val_f1": float(np.mean([r["f1"] for r in per_video])),
                  "val_spearman": float(np.mean([r["spearman"] for r in per_video])),
                  "empty_prediction_rate": float(np.mean([r["empty_prediction"] for r in per_video])),
                  "per_video": per_video}
        history.append(record)
        if best is None or record["val_f1"] > best["val_f1"]:
            best = record
            torch.save({"model": head.state_dict(), "input_dim": dim, "epoch": epoch,
                        "protocol": PROTOCOL, "seed": seed}, output / "temporal_best.pt")
        print(json.dumps({"stage": "fit", "variant": output.name,
                          **{k: v for k, v in record.items() if k != "per_video"}}), flush=True)
    torch.save({"model": head.state_dict(), "input_dim": dim, "epoch": epochs - 1,
                "protocol": PROTOCOL, "seed": seed}, output / "temporal_last.pt")
    return {"feature_dim": int(dim), "training_seconds": time.perf_counter() - t0,
            "head_parameter_count": sum(p.numel() for p in head.parameters()),
            "checkpoint_bytes": (output / "temporal_best.pt").stat().st_size,
            "checkpoint_sha256": sha256(output / "temporal_best.pt"), "best": best,
            "history": history}


def main():
    ap = argparse.ArgumentParser()
    for name in ["train", "val", "model", "output", "control-root"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch-clips", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--code-commit", required=True)
    args = ap.parse_args()
    if (args.sample_fps, args.clip_len, args.stride) != (2.0, 16, 4):
        raise ValueError("Change protocol version before changing clip policy")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        raise FileExistsError("Use a new run directory; output config is immutable")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"protocol": PROTOCOL, "official_aic_gt": False,
                   "target_threshold": TARGET_THRESHOLD, "prediction_threshold": PREDICTION_THRESHOLD,
                   "manifest_sha256": {s: sha256(getattr(args, s)) for s in ["train", "val"]},
                   "script_sha256": sha256(__file__)})
    train_sources = {r["source_id"][:11] for r in rows(args.train)}
    val_sources = {r["source_id"][:11] for r in rows(args.val)}
    if train_sources & val_sources:
        raise ValueError("Train/val share YouTube source IDs")
    config["youtube_source_overlap"] = 0
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    seed_everything(args.seed)
    device = torch.device(args.device)
    from transformers import AutoModel
    t0 = time.perf_counter()
    preproc = json.loads((args.model / "preprocessor_config.json").read_text())
    model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    model.eval().to(device)
    params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    manifests = extract_all(model, {"train": args.train, "val": args.val}, args.output,
                            device, args, preproc["image_mean"], preproc["image_std"])
    extraction_s = time.perf_counter() - t0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    fitted = {variant: fit(paths, args.output / variant, device, args.epochs, args.seed)
              for variant, paths in manifests.items()}
    result = {"run_id": args.output.name, "protocol": PROTOCOL, "official_f_video": None,
              "competition_score": None, "official_aic_gt": False, "model": "VideoMAEv2-Base",
              "parameter_count": params, "model_weight_bytes": (args.model / "model.safetensors").stat().st_size,
              "model_sha256": sha256(args.model / "model.safetensors"), "extraction_seconds": extraction_s,
              "peak_allocated_vram_bytes": peak_bytes, "elapsed_seconds": time.perf_counter() - t0,
              "train_videos": len(rows(args.train)), "val_videos": len(rows(args.val)),
              "variants": fitted, "config": config,
              "preprocess_note": "ImageNet stats from model card; letterbox geometry matched to image controls, not model-card center crop"}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": "complete", "metrics": str(args.output / "metrics.json")}), flush=True)


if __name__ == "__main__":
    main()
