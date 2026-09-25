"""Frozen DeiT-S/16 temporal bundle used for the S-tier challenger."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn

from .models import TemporalUNet, sha256_file


class DeiTSmallTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("deit_small_patch16_224", pretrained=False,
                                          num_classes=0)
        self.temporal = TemporalUNet(384)
        self.feature_bank_enabled = False
        self.temporal_shift_enabled = False

    def encode_frames(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward(self, features: Tensor, aux: Tensor | None = None,
                lengths: Tensor | None = None) -> Tensor:
        if features.ndim == 5:
            batch, steps = features.shape[:2]
            features = self.encode_frames(features.reshape(batch * steps, *features.shape[2:]))
            features = features.reshape(batch, steps, -1)
        return self.temporal(features, lengths=lengths)


def load_deit_bundle(path: str | Path, device: str | torch.device = "cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("architecture") != "Bs0_deit_s_temporal":
        raise ValueError("Unsupported DeiT-S bundle")
    model = DeiTSmallTemporal()
    model.backbone.load_state_dict(bundle["backbone_state"], strict=True)
    model.temporal.load_state_dict(bundle["temporal_state"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    metadata = {k: v for k, v in bundle.items()
                if k not in {"backbone_state", "temporal_state"}}
    metadata.update({"loaded_path": str(path), "loaded_bytes": Path(path).stat().st_size,
                     "loaded_sha256": sha256_file(path), "runtime_dtype": "float32"})
    return model, metadata
