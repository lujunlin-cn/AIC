"""Frozen torchvision ViT-B/16 + shared temporal head for the B0 pilot."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
from torchvision.models import vit_b_16
from .models import TemporalUNet, sha256_file

class B0ViTTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_b_16(weights=None)
        self.backbone.heads = nn.Identity()
        self.temporal = TemporalUNet(768)
        self.feature_bank_enabled = False
        self.temporal_shift_enabled = False

    def encode_frames(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward(self, features: Tensor, aux: Tensor | None = None,
                lengths: Tensor | None = None) -> Tensor:
        if features.ndim == 5:
            b,t=features.shape[:2]
            features=self.encode_frames(features.reshape(b*t,*features.shape[2:])).reshape(b,t,-1)
        return self.temporal(features, lengths=lengths)

def load_b0_bundle(path: str | Path, device: str | torch.device = "cpu"):
    bundle=torch.load(path,map_location="cpu",weights_only=True)
    if bundle.get("architecture") != "B0_vit_b16_temporal_probe": raise ValueError("Unsupported B0 bundle")
    model=B0ViTTemporal(); model.backbone.load_state_dict(bundle["backbone_state"],strict=True); model.temporal.load_state_dict(bundle["temporal_state"],strict=True)
    model.to(device=device,dtype=torch.float32).eval()
    meta={k:v for k,v in bundle.items() if k not in {"backbone_state","temporal_state"}}; meta.update({"loaded_path":str(path),"loaded_bytes":Path(path).stat().st_size,"loaded_sha256":sha256_file(path),"runtime_dtype":"float32"})
    return model,meta
