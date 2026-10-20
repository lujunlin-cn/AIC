#!/usr/bin/env python3
"""Small, explicit DINOv3 ViT-B/16 runtime used by the AIC probes.

The official DINOv3 repository is intentionally supplied as a path instead of
vendored into this repository.  This keeps the AIC checkout small while the
runtime manifest records the exact upstream commit and weight hash.  The
wrapper only exposes the normalized CLS and patch tokens; storage/register
tokens are kept separate and never enter the spatial grid.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


ARCH = "dinov3_vitb16"
EMBED_DIM = 768
PATCH_SIZE = 16
N_STORAGE_TOKENS = 4


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_model(repo: str | Path, weights: str | Path, device: str = "cpu",
               dtype: str = "float32") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the official model from a local source checkout and local weights."""
    repo = Path(repo).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if not (repo / "hubconf.py").exists():
        raise FileNotFoundError(f"DINOv3 source checkout missing hubconf.py: {repo}")
    if not weights.exists():
        raise FileNotFoundError(weights)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    # Importing backbones directly avoids hubconf importing optional detection
    # and segmentation dependencies that are irrelevant to this probe.
    backbones = importlib.import_module("dinov3.hub.backbones")
    model = backbones.dinov3_vitb16(pretrained=True, weights=str(weights), check_hash=True)
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None or not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    # CPU inference with fp16 is both slow and numerically fragile.  Keep the
    # requested dtype in the manifest, but use fp32 for CPU smoke runs.
    actual_dtype = torch.float32 if str(device).startswith("cpu") else torch_dtype
    model = model.to(device=device, dtype=actual_dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = sum(int(p.numel()) for p in model.parameters())
    if params != 85_669_632:
        raise RuntimeError(f"Unexpected DINOv3 ViT-B/16 parameter count: {params}")
    meta = {
        "architecture": ARCH,
        "embed_dim": EMBED_DIM,
        "patch_size": PATCH_SIZE,
        "storage_tokens": N_STORAGE_TOKENS,
        "parameter_count": params,
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": sha256_file(weights),
        "official_repo": str(repo),
        "requested_dtype": dtype,
        "runtime_dtype": str(actual_dtype).replace("torch.", ""),
        "device": str(device),
    }
    return model, meta


def _resize_square(image: Image.Image, size: int) -> tuple[Image.Image, dict[str, float]]:
    out = image.resize((size, size), Image.Resampling.BICUBIC)
    return out, {"scale_x": size / image.width, "scale_y": size / image.height,
                 "pad_x": 0.0, "pad_y": 0.0, "source_w": float(image.width),
                 "source_h": float(image.height), "output_size": float(size)}


def _resize_letterbox(image: Image.Image, size: int) -> tuple[Image.Image, dict[str, float]]:
    scale = min(size / image.width, size / image.height)
    nw = max(1, round(image.width * scale))
    nh = max(1, round(image.height * scale))
    resized = image.resize((nw, nh), Image.Resampling.BICUBIC)
    # ImageNet mean in byte space is the least surprising neutral padding for
    # the pretrained normalization and does not create a black-border cue.
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    px = (size - nw) // 2
    py = (size - nh) // 2
    canvas.paste(resized, (px, py))
    return canvas, {"scale_x": nw / image.width, "scale_y": nh / image.height,
                    "pad_x": float(px), "pad_y": float(py),
                    "source_w": float(image.width), "source_h": float(image.height),
                    "output_size": float(size)}


def preprocess_rgb(frame: np.ndarray, size: int = 224,
                   mode: str = "letterbox") -> tuple[torch.Tensor, dict[str, float]]:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB [H,W,3], got {frame.shape} {frame.dtype}")
    image = Image.fromarray(frame, mode="RGB")
    if mode == "square":
        image, transform = _resize_square(image, size)
    elif mode == "letterbox":
        image, transform = _resize_letterbox(image, size)
    else:
        raise ValueError(f"unknown resize mode: {mode}")
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (x - mean) / std, transform


@torch.inference_mode()
def encode_rgb(model: torch.nn.Module, frames: Iterable[np.ndarray], device: str = "cpu",
               batch_size: int = 16, size: int = 224, resize_mode: str = "letterbox",
               save_patch: bool = False) -> dict[str, Any]:
    frames = list(frames)
    if not frames:
        raise ValueError("encode_rgb received no frames")
    tensors, transforms = zip(*(preprocess_rgb(f, size=size, mode=resize_mode) for f in frames))
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    cls_parts, patch_parts, storage_parts = [], [], []
    for start in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[start:start + batch_size]).to(device=device, dtype=model_dtype)
        out = model.forward_features(x)
        cls_parts.append(out["x_norm_clstoken"].float().cpu())
        storage_parts.append(out["x_storage_tokens"].float().cpu())
        if save_patch:
            patch = out["x_norm_patchtokens"].float().cpu()
            h = int(round(patch.shape[1] ** 0.5))
            if h * h != patch.shape[1]:
                raise RuntimeError(f"patch token count is not square: {patch.shape}")
            patch_parts.append(patch.reshape(len(x), h, h, EMBED_DIM))
    result: dict[str, Any] = {
        "pooled": torch.cat(cls_parts).numpy(),
        "storage": torch.cat(storage_parts).numpy(),
        "transforms": list(transforms),
        "grid_shape": [int(patch_parts[0].shape[1]), int(patch_parts[0].shape[2])] if patch_parts else None,
    }
    if save_patch:
        result["patch_tokens"] = torch.cat(patch_parts).numpy()
    return result


def jsonable_meta(meta: dict[str, Any]) -> str:
    return json.dumps(meta, sort_keys=True, separators=(",", ":"))
