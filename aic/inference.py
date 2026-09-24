"""Reproducible raw-video -> submission JSONL inference for A0.

The pipeline deliberately expands sampled probabilities back to original decoded
frame indices before emitting records. A `--dummy` run is the public center-crop
format baseline (all decoded frames); it is never labelled as a trained model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (ContractError, VideoMetadata, center_crop, load_index,
                       load_jsonl, write_submission)
from .video import expand_scores, frame_timeline, iter_sampled_frames, probe_video


def _normalise(images: Sequence[np.ndarray]):
    import torch

    if not images:
        raise ValueError("at least one image is required")
    values = np.stack(images).astype(np.float32) / 255.0
    values = values.transpose(0, 3, 1, 2)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return torch.from_numpy((values - mean) / std)


def _row(video_id: str, ratio: Sequence[float], selected_frames: Sequence[int],
         width: int, height: int, model_size_mb: float | None) -> dict[str, Any]:
    crop = center_crop(width, height, ratio)
    predictions = [{"frame": int(frame), "bboxes": crop.copy()} for frame in selected_frames]
    result: dict[str, Any] = {"video_id": video_id, "targetRatioWH": list(ratio)}
    if model_size_mb is not None:
        result["model_size_mb"] = float(model_size_mb)
    result["predictions"] = predictions
    return result


def _record_path(record: Mapping[str, Any], video_root: str | Path | None) -> Path:
    raw = record.get("video_path", record.get("path", record.get("video")))
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"index row {record.get('video_id')!r} lacks video_path")
    path = Path(raw)
    if not path.is_absolute() and video_root is not None:
        path = Path(video_root) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_inference(index_path: str | Path, output_path: str | Path, *,
                  model_path: str | Path | None = None,
                  video_root: str | Path | None = None,
                  sample_fps: float = 2.0, threshold: float = .5,
                  device: str = "cpu", stage: str = "preliminary",
                  model_size_mb: float | None = None, dummy: bool = False,
                  batch_size: int = 16) -> dict[str, Any]:
    """Run A0 and validate the complete output before replacing output_path."""
    if model_path is None and not dummy:
        raise ContractError("model_path is required unless --dummy is explicit")
    if not np.isfinite(threshold):
        raise ContractError("threshold must be finite")
    records = load_jsonl(index_path)
    index = load_index(index_path)
    model = None
    if model_path is not None:
        from .models import load_inference_model
        model, metadata = load_inference_model(model_path, device=device)
        if model_size_mb is None:
            model_size_mb = float(metadata["loaded_bytes"]) / 1_000_000
    rows: list[dict[str, Any]] = []
    with __import__("torch").inference_mode():
        for record in records:
            video_id = record["video_id"]
            metadata = index[video_id]
            path = _record_path(record, video_root)
            info = probe_video(path)
            observed = (info.width, info.height, info.frame_count)
            expected = (metadata.width, metadata.height, metadata.frame_count)
            if observed != expected:
                raise ContractError(f"{video_id}: index dimensions/frame count {expected} != decoded {observed}")
            if dummy:
                # Public baseline behavior: every original frame with the
                # largest legal centered crop. This is a format/reference run,
                # not a learned competition result.
                selected = [stamp.index for stamp in frame_timeline(path)]
            else:
                sampled = list(iter_sampled_frames(path, sample_fps=sample_fps, size=224))
                if not sampled:
                    raise ContractError(f"{video_id}: no sampled frames")
                images = [entry[2] for entry in sampled]
                features = []
                for start in range(0, len(images), batch_size):
                    batch = _normalise(images[start:start + batch_size]).to(device)
                    encoded = model.encode_frames(batch).detach().float().cpu()
                    if not __import__("torch").isfinite(encoded).all():
                        raise FloatingPointError(f"{video_id}: nonfinite visual features")
                    features.append(encoded)
                features = __import__("torch").cat(features, dim=0)
                scores = model(features.unsqueeze(0).to(device))[0].detach().float().cpu().sigmoid().numpy()
                selected = expand_scores(frame_timeline(path), [x[1] for x in sampled], scores, threshold)
            rows.append(_row(video_id, record["targetRatioWH"], selected,
                             info.width, info.height, model_size_mb))
    report = write_submission(output_path, rows, index, stage=stage,
                              actual_model_size_mb=model_size_mb)
    return {"output": str(output_path), "dummy": dummy, "rows": len(rows),
            "selected_predictions": sum(len(r["predictions"]) for r in rows),
            "model_size_mb": model_size_mb, "validation": report.to_dict()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--video-root")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=["preliminary", "final"], default="preliminary")
    parser.add_argument("--model-size-mb", type=float)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        result = run_inference(index_path=args.index, output_path=args.output,
                               model_path=args.model, video_root=args.video_root,
                               sample_fps=args.sample_fps, threshold=args.threshold,
                               device=args.device, stage=args.stage,
                               model_size_mb=args.model_size_mb, dummy=args.dummy,
                               batch_size=args.batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ContractError, OSError, ValueError, FloatingPointError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
