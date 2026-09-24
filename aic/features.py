"""Feature-cache IO and deterministic ResNet18 extraction for A0.

Cache records are intentionally self-describing.  A cache is an experiment
artifact, not a substitute for the raw-video manifest: labels remain aligned
to the original decoded frame indices.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn


CACHE_VERSION = 1
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def save_backbone_state(model: nn.Module, path: str | Path,
                        provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist the exact headless encoder used to produce caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    bundle = {"format_version": 1, "architecture": "ResNet18_without_classifier",
              "state_dict": state, "provenance": dict(provenance or {})}
    torch.save(bundle, path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size,
            "MB_decimal": path.stat().st_size / 1_000_000, "sha256": digest,
            "parameter_count": sum(value.numel() for value in state.values()),
            "provenance": bundle["provenance"]}


def _letterbox(frames: np.ndarray, size: int = 224) -> Tensor:
    """RGB uint8 [N,H,W,3] -> normalized [N,3,size,size], preserving geometry."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB uint8 [N,H,W,3], got {frames.shape}")
    # iter_sampled_frames already returns precisely this letterboxed size.  For
    # direct callers use the same PIL implementation as aic.video, avoiding a
    # second interpolation or a subtly different padding convention.
    if frames.shape[1] != size or frames.shape[2] != size:
        from .video import letterbox_rgb
        frames = np.stack([letterbox_rgb(frame, size) for frame in frames])
    x = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).contiguous().float() / 255.0
    out = x
    mean = IMAGENET_MEAN.to(out)
    std = IMAGENET_STD.to(out)
    return (out - mean) / std


def _read_labels(path: str | Path | None) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    if "labels" not in result and "scores" in result:
        scores = result["scores"].astype(np.float32)
        # TVSum-style files may retain one score column per annotator.
        if scores.ndim > 1:
            scores = np.nanmean(scores, axis=-1)
        result["scores"] = scores
        result["labels"] = np.clip((scores - 1.0) / 4.0, 0.0, 1.0)
    if "labels" not in result:
        raise ValueError(f"labels_path={path} has neither labels nor scores")
    return result


def align_labels(label_data: dict[str, np.ndarray] | None,
                  frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map source labels to selected original frame indices; mask missing labels."""
    n = len(frame_indices)
    labels = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.bool_)
    if label_data is None:
        return labels, mask
    if "labels" in label_data:
        values = np.asarray(label_data["labels"], dtype=np.float32)
        if values.ndim > 1:
            values = np.nanmean(values, axis=-1)
        values = values.reshape(-1)
    elif "scores" in label_data:
        # TVSum and related sources expose 1..5 importance scores. Keep this
        # conversion explicit and local; it is not a claim of competition GT.
        scores = np.asarray(label_data["scores"], dtype=np.float32)
        if scores.ndim > 1:
            scores = np.nanmean(scores, axis=-1)
        scores = scores.reshape(-1)
        values = np.clip((scores - 1.0) / 4.0, 0.0, 1.0)
    else:
        raise ValueError("label_data requires labels or scores")
    if "frame_indices" in label_data:
        source = np.asarray(label_data["frame_indices"], dtype=np.int64).reshape(-1)
        lookup = {int(index): float(value) for index, value in zip(source, values)
                  if math.isfinite(float(value))}
        for i, frame_index in enumerate(frame_indices):
            value = lookup.get(int(frame_index))
            if value is not None:
                labels[i], mask[i] = np.clip(value, 0.0, 1.0), True
    else:
        valid = frame_indices >= 0
        valid &= frame_indices < len(values)
        labels[valid] = np.nan_to_num(values[frame_indices[valid]], nan=0.0,
                                      posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        mask[valid] = np.isfinite(values[frame_indices[valid]])
    return labels, mask


def save_feature_cache(path: str | Path, features: np.ndarray | Tensor,
                       frame_indices: np.ndarray, timestamps: np.ndarray,
                       labels: np.ndarray | None = None,
                       label_mask: np.ndarray | None = None,
                       metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features.detach().cpu() if isinstance(features, Tensor) else features,
                          dtype=np.float32)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(frame_indices) or len(features) != len(timestamps):
        raise ValueError("features/frame_indices/timestamps must have matching [T] dimension")
    if not np.isfinite(features).all():
        raise FloatingPointError("Attempting to save non-finite feature cache")
    if labels is None:
        labels = np.zeros(len(features), dtype=np.float32)
    if label_mask is None:
        label_mask = np.zeros(len(features), dtype=np.bool_)
    labels = np.asarray(labels, dtype=np.float32)
    label_mask = np.asarray(label_mask, dtype=np.bool_)
    if labels.shape != (len(features),) or label_mask.shape != labels.shape:
        raise ValueError("labels and label_mask must be [T]")
    metadata = dict(metadata or {})
    metadata.update({"cache_version": CACHE_VERSION, "feature_dim": int(features.shape[1]),
                     "length": int(len(features)), "dtype": "float32"})
    np.savez_compressed(path, features=features, labels=labels, mask=label_mask,
                        frame_indices=frame_indices, timestamps=timestamps,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"features", "labels", "mask", "frame_indices", "timestamps"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Feature cache missing keys: {sorted(missing)}")
        metadata_raw = data["metadata_json"] if "metadata_json" in data else np.asarray("{}")
        metadata = json.loads(str(metadata_raw.item()))
        result = {key: np.array(data[key]) for key in required}
    if result["features"].ndim != 2 or not np.isfinite(result["features"]).all():
        raise ValueError(f"Invalid/non-finite cache features in {path}")
    if len({len(result["features"]), len(result["labels"]), len(result["mask"]),
            len(result["frame_indices"]), len(result["timestamps"])}) != 1:
        raise ValueError(f"Inconsistent cache lengths in {path}")
    result["metadata"] = metadata
    return result


class FeatureCacheDataset(torch.utils.data.Dataset):
    """Manifest-backed dataset; each item is one whole variable-length video."""
    def __init__(self, manifest: str | Path | Iterable[dict[str, Any]]):
        if isinstance(manifest, (str, Path)):
            self.records = [json.loads(line) for line in Path(manifest).read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith("#")]
        else:
            self.records = list(manifest)
        if not self.records:
            raise ValueError("Feature manifest is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cache = load_feature_cache(record["path"])
        return {**cache, "video_id": str(record.get("video_id", Path(record["path"]).stem)),
                "record": record}


def collate_feature_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Empty batch")
    max_t = max(len(item["features"]) for item in items)
    dim = items[0]["features"].shape[1]
    batch = len(items)
    features = torch.zeros(batch, max_t, dim, dtype=torch.float32)
    labels = torch.zeros(batch, max_t, dtype=torch.float32)
    mask = torch.zeros(batch, max_t, dtype=torch.bool)
    lengths = torch.zeros(batch, dtype=torch.long)
    for i, item in enumerate(items):
        t = len(item["features"])
        features[i, :t] = torch.from_numpy(item["features"])
        labels[i, :t] = torch.from_numpy(item["labels"])
        mask[i, :t] = torch.from_numpy(item["mask"].astype(np.bool_))
        lengths[i] = t
    return {"features": features, "labels": labels, "mask": mask,
            "lengths": lengths, "video_ids": [item["video_id"] for item in items]}


@torch.inference_mode()
def extract_video_cache(model: nn.Module, video_path: str | Path, output_path: str | Path,
                        labels_path: str | Path | None = None, device: str = "cpu",
                        sample_fps: float = 2.0, batch_size: int = 32,
                        metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract features via ``aic.video.iter_sampled_frames`` and save one NPZ."""
    from .video import iter_sampled_frames
    sampled = list(iter_sampled_frames(video_path, sample_fps=sample_fps, size=224))
    if not sampled:
        raise ValueError(f"Video produced no frames: {video_path}")
    frame_indices = np.asarray([item[0] for item in sampled], dtype=np.int64)
    timestamps = np.asarray([item[1] for item in sampled], dtype=np.float64)
    model = model.to(device).eval()
    outputs: list[np.ndarray] = []
    for offset in range(0, len(sampled), batch_size):
        frames = np.stack([item[2] for item in sampled[offset:offset + batch_size]])
        tensor = _letterbox(frames).to(device)
        value = model(tensor).float().detach().cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite backbone output for {video_path}")
        outputs.append(value)
    features = np.concatenate(outputs, axis=0)
    labels, mask = align_labels(_read_labels(labels_path), frame_indices)
    meta = dict(metadata or {})
    meta.update({"video_path": str(video_path), "sample_fps": sample_fps,
                 "backbone": "ResNet18_without_classifier", "label_source": str(labels_path) if labels_path else None})
    save_feature_cache(output_path, features, frame_indices, timestamps, labels, mask, meta)
    return {"path": str(output_path), "frames": len(features), "feature_dim": features.shape[1],
            "labeled_frames": int(mask.sum())}


def extract_manifest(manifest: str | Path, output_dir: str | Path, model: nn.Module,
                     device: str = "cpu", sample_fps: float = 2.0, batch_size: int = 32,
                     skip_existing: bool = True) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(manifest).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for record in records:
        video_id = str(record["video_id"])
        output = output_dir / f"{video_id}.npz"
        if skip_existing and output.exists():
            results.append({"path": str(output), "video_id": video_id, "skipped": True})
            continue
        result = extract_video_cache(model, record["path"], output, record.get("labels_path"),
                                     device, sample_fps, batch_size,
                                     {key: record.get(key) for key in ("video_id", "split", "source_id", "source_group")})
        result["video_id"] = video_id
        results.append(result)
    return results
