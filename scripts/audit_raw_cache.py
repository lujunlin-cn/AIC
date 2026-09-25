#!/usr/bin/env python3
"""Compare raw-video and cached-feature inference on one video.

The command is deliberately a same-environment audit: decoder versions can
change RGB values before the backbone, so the output records Python/PyAV and
PyTorch versions alongside frame, feature, auxiliary and probability errors.
It does not train or alter either artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

# Allow ``python scripts/audit_raw_cache.py`` from a source checkout without
# requiring an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.features import compute_feature_bank, load_feature_cache
from aic.inference import _normalise
from aic.models import load_inference_model
from aic.video import iter_sampled_frames


def _errors(raw: np.ndarray, cached: np.ndarray) -> dict[str, object]:
    if raw.shape != cached.shape:
        raise ValueError(f"shape mismatch: raw={raw.shape}, cache={cached.shape}")
    delta = np.abs(raw.astype(np.float64) - cached.astype(np.float64))
    result: dict[str, object] = {
        "shape": list(raw.shape),
        "max_abs_error": float(delta.max()) if delta.size else 0.0,
        "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
        "equal": bool(np.array_equal(raw, cached)),
    }
    return result


def audit(video: str | Path, cache: str | Path, model_path: str | Path,
          *, sample_fps: float = 2.0, device: str = "cpu",
          batch_size: int = 32, threshold: float = .5) -> dict[str, object]:
    sampled = list(iter_sampled_frames(video, sample_fps=sample_fps, size=224))
    if not sampled:
        raise ValueError(f"video produced no sampled frames: {video}")
    images = [item[2] for item in sampled]
    model, metadata = load_inference_model(model_path, device=device)
    model.eval()
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = _normalise(images[start:start + batch_size]).to(device)
            values.append(model.encode_frames(batch).float().cpu())
        raw_features = torch.cat(values, dim=0)
        raw_aux = None
        if getattr(model, "feature_bank_enabled", False):
            raw_aux = torch.from_numpy(compute_feature_bank(np.stack(images)))
            raw_aux = raw_aux.unsqueeze(0).to(device)
        raw_logits = model(raw_features.unsqueeze(0).to(device), raw_aux)
        raw_probabilities = raw_logits[0].sigmoid().float().cpu().numpy()

    cached = load_feature_cache(cache)
    cached_features = cached["features"].astype(np.float32)
    cached_aux = cached["aux"].astype(np.float32)
    with torch.inference_mode():
        cache_aux_tensor = None
        if getattr(model, "feature_bank_enabled", False):
            if cached_aux.shape[1] != 32:
                raise ValueError(f"A2 model requires 32D cache aux, got {cached_aux.shape}")
            cache_aux_tensor = torch.from_numpy(cached_aux).unsqueeze(0).to(device)
        cache_logits = model(torch.from_numpy(cached_features).unsqueeze(0).to(device),
                             cache_aux_tensor)
        cache_probabilities = cache_logits[0].sigmoid().float().cpu().numpy()

    raw_indices = np.asarray([item[0] for item in sampled], dtype=np.int64)
    raw_times = np.asarray([item[1] for item in sampled], dtype=np.float64)
    cached_indices = np.asarray(cached["frame_indices"])
    cached_times = np.asarray(cached["timestamps"])
    result: dict[str, object] = {
        "video": str(video),
        "cache": str(cache),
        "model": str(model_path),
        "sample_fps": float(sample_fps),
        "sampled_frames": len(sampled),
        "decoder": {},
        "frame_indices": _errors(raw_indices, cached_indices),
        "timestamps": _errors(raw_times, cached_times),
        "features": _errors(raw_features.numpy(), cached_features),
        "probabilities": _errors(raw_probabilities, cache_probabilities),
        "selected_at_threshold": {
            "threshold": float(threshold),
            "raw": int(np.count_nonzero(raw_probabilities >= threshold)),
            "cache": int(np.count_nonzero(cache_probabilities >= threshold)),
        },
        "loaded_model_bytes": int(metadata["loaded_bytes"]),
    }
    if getattr(model, "feature_bank_enabled", False):
        result["aux_32d"] = _errors(raw_aux[0].cpu().numpy(), cached_aux)
    try:
        import av
        result["decoder"] = {"pyav": av.__version__, "torch": torch.__version__}
    except Exception:
        pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(args.video, args.cache, args.model,
                   sample_fps=args.sample_fps, device=args.device,
                   batch_size=args.batch_size, threshold=args.threshold)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
