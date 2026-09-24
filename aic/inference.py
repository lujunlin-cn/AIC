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


SPATIAL_MODES = ("center", "saliency", "subject")


def _normalise(images: Sequence[np.ndarray]):
    import torch

    if not images:
        raise ValueError("at least one image is required")
    values = np.stack(images).astype(np.float32) / 255.0
    values = values.transpose(0, 3, 1, 2)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return torch.from_numpy((values - mean) / std)


def _spatial_crop(image: np.ndarray, mode: str, ratio: Sequence[float],
                  width: int, height: int) -> list[float]:
    """Select one explicit spatial policy for a selected frame.

    ``image`` is the same sampled, letterboxed RGB tensor used to produce the
    temporal features.  The geometry output is always in original coded pixel
    coordinates.  Saliency and subject are lightweight candidates; they do not
    claim detector boxes or crop IoU without spatial ground truth.
    """
    if mode not in SPATIAL_MODES:
        raise ContractError(f"spatial_mode must be one of {SPATIAL_MODES}, got {mode!r}")
    if mode == "center":
        return center_crop(width, height, ratio)
    from .spatial import saliency_crop, subject_crop
    if mode == "saliency":
        # Candidate functions operate in image dimensions; map their normalized
        # centre to the original coded-pixel geometry while retaining the legal
        # crop width chosen by the policy.
        crop = saliency_crop(image, ratio)
    else:
        crop = subject_crop(image, ratio)
    image_h, image_w = image.shape[:2]
    candidate_x, candidate_y, candidate_w = crop
    candidate_h = candidate_w * float(ratio[1]) / float(ratio[0])
    cx = (candidate_x + candidate_w / 2) / max(1, image_w) * width
    cy = (candidate_y + candidate_h / 2) / max(1, image_h) * height
    from .spatial import place_crop
    return place_crop(width, height, ratio, cx, cy, center_crop(width, height, ratio)[2])


def _row(video_id: str, ratio: Sequence[float], selected_frames: Sequence[int],
         width: int, height: int, model_size_mb: float | None,
         spatial_mode: str = "center",
         frame_crops: Mapping[int, Sequence[float]] | None = None) -> dict[str, Any]:
    center = center_crop(width, height, ratio)
    predictions = []
    for frame in selected_frames:
        crop = list(frame_crops.get(int(frame), center) if frame_crops else center)
        predictions.append({"frame": int(frame), "bboxes": crop})
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
                  batch_size: int = 16, spatial_mode: str = "center") -> dict[str, Any]:
    """Run A0 and validate the complete output before replacing output_path."""
    if model_path is None and not dummy:
        raise ContractError("model_path is required unless --dummy is explicit")
    if not np.isfinite(threshold):
        raise ContractError("threshold must be finite")
    if spatial_mode not in SPATIAL_MODES:
        raise ContractError(f"spatial_mode must be one of {SPATIAL_MODES}, got {spatial_mode!r}")
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
                model_input = features.unsqueeze(0).to(device)
                model_aux = None
                if getattr(model, "feature_bank_enabled", False):
                    # A2's aux is deterministic and must be built from exactly
                    # the sampled RGB frames used by the backbone.  This keeps
                    # raw-video inference aligned with feature-cache training.
                    from .features import compute_feature_bank
                    aux = compute_feature_bank(np.stack(images, axis=0))
                    model_aux = __import__("torch").from_numpy(aux).unsqueeze(0).to(device)
                scores = model(model_input, model_aux)[0].detach().float().cpu().sigmoid().numpy()
                selected = expand_scores(frame_timeline(path), [x[1] for x in sampled], scores, threshold)
            frame_crops: dict[int, Sequence[float]] | None = None
            if not dummy and spatial_mode != "center":
                # Dense frame selection is expanded from sampled timestamps.
                # Associate each selected original frame with the nearest
                # sampled image; this is deterministic and avoids a second
                # image decode pass while keeping temporal/spatial inputs tied.
                frame_crops = {}
                sampled_indices = np.asarray([x[0] for x in sampled], dtype=np.int64)
                sampled_images = [x[2] for x in sampled]
                for frame in selected:
                    nearest = int(np.argmin(np.abs(sampled_indices - int(frame))))
                    frame_crops[int(frame)] = _spatial_crop(
                        sampled_images[nearest], spatial_mode,
                        record["targetRatioWH"], info.width, info.height)
            rows.append(_row(video_id, record["targetRatioWH"], selected,
                             info.width, info.height, model_size_mb,
                             spatial_mode=spatial_mode, frame_crops=frame_crops))
    report = write_submission(output_path, rows, index, stage=stage,
                              actual_model_size_mb=model_size_mb)
    return {"output": str(output_path), "dummy": dummy, "rows": len(rows),
            "selected_predictions": sum(len(r["predictions"]) for r in rows),
            "model_size_mb": model_size_mb, "spatial_mode": spatial_mode,
            "validation": report.to_dict()}


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
    parser.add_argument("--spatial-mode", choices=SPATIAL_MODES, default="center",
                        help="center, saliency, or lightweight subject proxy")
    args = parser.parse_args(argv)
    try:
        result = run_inference(index_path=args.index, output_path=args.output,
                               model_path=args.model, video_root=args.video_root,
                               sample_fps=args.sample_fps, threshold=args.threshold,
                               device=args.device, stage=args.stage,
                               model_size_mb=args.model_size_mb, dummy=args.dummy,
                               batch_size=args.batch_size, spatial_mode=args.spatial_mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ContractError, OSError, ValueError, FloatingPointError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
