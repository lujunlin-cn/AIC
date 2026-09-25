"""A0: a single ResNet18 visual encoder and the shared temporal U-Net.

The default constructor never downloads weights. Explicit pretrained loading is
limited to feature preparation; exported inference files contain all weights.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int, dilations=(1, 1)):
        layers = []
        for dilation in dilations:
            layers.extend([
                nn.Conv1d(incoming, outgoing, 3, padding=dilation,
                          dilation=dilation, bias=False),
                nn.GroupNorm(32, outgoing),
                nn.ReLU(inplace=True),
            ])
            incoming = outgoing
        super().__init__(*layers)


class TemporalUNet(nn.Module):
    """Architecture in research document 02 section 4.2; ``[B,T,D]`` -> ``[B,T]``.

    ``lengths`` must be supplied whenever ``features`` is right-padded (as in
    :func:`collate_feature_batch`).  GroupNorm and temporal pooling otherwise
    see the padding and can change valid-timestep logits.
    """

    def __init__(self, input_dim: int = 512, aux_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.aux_dim = int(aux_dim)
        self.adapter = nn.Linear(input_dim, 128)
        self.aux_adapter = nn.Sequential(nn.Linear(self.aux_dim, 64), nn.ReLU(), nn.Linear(64, 128)) if self.aux_dim else None
        self.encoder0 = ConvBlock(128, 128)
        self.encoder1 = ConvBlock(128, 192)
        self.encoder2 = ConvBlock(192, 256)
        self.bottleneck = ConvBlock(256, 256, dilations=(2, 4))
        self.decoder1 = ConvBlock(256 + 192, 192)
        self.decoder0 = ConvBlock(192 + 128, 128)
        self.output = nn.Conv1d(128, 1, 1)

    def forward(self, features: Tensor, aux: Tensor | None = None,
                lengths: Tensor | None = None) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected [B,T,{self.input_dim}], got {features.shape}")
        if features.shape[1] == 0:
            raise ValueError("Cannot process an empty sequence")
        # A DataLoader batch contains right-padded variable-length videos.  A
        # GroupNorm layer normalizes over both channels and temporal positions,
        # so those padding values would otherwise alter every valid timestep.
        # Process each sequence at its true length when lengths are supplied;
        # this also keeps pooling/interpolation geometry identical to a
        # standalone sequence.  The returned tensor retains the padded batch
        # shape, with zero logits outside each valid sequence.
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
                raise ValueError("lengths must be [B] matching features")
            lengths = lengths.to(device=features.device, dtype=torch.long)
            if bool((lengths <= 0).any()) or bool((lengths > features.shape[1]).any()):
                raise ValueError("lengths must be in [1, T]")
            # Fast path when all samples have identical lengths: there is no
            # padding to contaminate normalization and batching is cheaper.
            if bool(torch.all(lengths == features.shape[1])):
                lengths = None
            else:
                outputs = features.new_zeros((features.shape[0], features.shape[1]))
                for index, length in enumerate(lengths.tolist()):
                    sample_aux = aux[index:index + 1, :length] if aux is not None else None
                    sample = self.forward(features[index:index + 1, :length], sample_aux)
                    outputs[index, :length] = sample[0]
                return outputs
        original_length = features.shape[1]
        # Two stride-2 reductions require at least four temporal positions.
        if original_length < 4:
            features = F.pad(features, (0, 0, 0, 4 - original_length))
            if aux is not None:
                aux = F.pad(aux, (0, 0, 0, 4 - original_length))
        projected = self.adapter(features)
        if self.aux_dim:
            if aux is None or aux.ndim != 3 or aux.shape[:2] != features.shape[:2] or aux.shape[-1] != self.aux_dim:
                raise ValueError(f"Expected aux [B,T,{self.aux_dim}] for feature-bank model")
            projected = projected + self.aux_adapter(aux)
        x = projected.transpose(1, 2)
        e0 = self.encoder0(x)
        e1 = self.encoder1(F.max_pool1d(e0, 2, ceil_mode=True))
        e2 = self.encoder2(F.max_pool1d(e1, 2, ceil_mode=True))
        middle = self.bottleneck(e2)
        d1 = self.decoder1(torch.cat([
            F.interpolate(middle, size=e1.shape[-1], mode="linear", align_corners=False), e1
        ], dim=1))
        d0 = self.decoder0(torch.cat([
            F.interpolate(d1, size=e0.shape[-1], mode="linear", align_corners=False), e0
        ], dim=1))
        return self.output(d0).squeeze(1)[:, :original_length]


def temporal_shift(features: Tensor, fold_div: int = 8) -> Tensor:
    """Parameter-free TSM channel shift on a cached [B,T,D] sequence."""
    if features.ndim != 3 or features.shape[1] == 0:
        raise ValueError(f"Expected nonempty [B,T,D], got {tuple(features.shape)}")
    fold = features.shape[-1] // fold_div
    if fold == 0:
        return features
    shifted = torch.zeros_like(features)
    shifted[:, 1:, :fold] = features[:, :-1, :fold]
    shifted[:, :-1, fold:2 * fold] = features[:, 1:, fold:2 * fold]
    shifted[:, :, 2 * fold:] = features[:, :, 2 * fold:]
    return shifted


def temporal_shift_feature_map(features: Tensor, fold_div: int = 8) -> Tensor:
    """Canonical TSM on an intermediate ``[B,T,C,H,W]`` feature map.

    This is deliberately separate from :func:`temporal_shift`, which is the
    historical feature-level embedding shift used by A1.  The operation moves
    channels along the video-time axis before the remaining CNN blocks and
    therefore cannot reuse a final embedding cache as an equivalent encoder.
    """
    if features.ndim != 5 or features.shape[1] == 0:
        raise ValueError(f"Expected nonempty [B,T,C,H,W], got {tuple(features.shape)}")
    fold = features.shape[2] // fold_div
    if fold == 0:
        return features
    shifted = torch.zeros_like(features)
    shifted[:, 1:, :fold] = features[:, :-1, :fold]
    shifted[:, :-1, fold:2 * fold] = features[:, 1:, fold:2 * fold]
    shifted[:, :, 2 * fold:] = features[:, :, 2 * fold:]
    return shifted


class A0Model(nn.Module):
    def __init__(self, feature_dim: int = 512, backbone: nn.Module | None = None,
                 temporal_shift_enabled: bool = False, feature_bank_enabled: bool = False):
        super().__init__()
        if backbone is None:
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)
            backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.temporal_shift_enabled = bool(temporal_shift_enabled)
        self.feature_bank_enabled = bool(feature_bank_enabled)
        self.temporal = TemporalUNet(self.feature_dim, 32 if self.feature_bank_enabled else 0)

    def encode_frames(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    @property
    def temporal_head(self) -> TemporalUNet:
        """Compatibility name used by inference/ablation scripts."""
        return self.temporal

    def forward(self, features: Tensor, aux: Tensor | None = None,
                lengths: Tensor | None = None) -> Tensor:
        if features.ndim == 5:
            batch, steps = features.shape[:2]
            features = self.encode_frames(features.reshape(batch * steps, *features.shape[2:]))
            features = features.reshape(batch, steps, -1)
        if self.temporal_shift_enabled:
            features = temporal_shift(features)
        return self.temporal(features, aux if self.feature_bank_enabled else None, lengths=lengths)


# Descriptive aliases retained for scripts/configs that refer to the route by
# its research name rather than the experiment identifier.
ResNet18TemporalUNet = A0Model
TemporalUNet1D = TemporalUNet


def load_imagenet_backbone() -> tuple[nn.Module, dict[str, Any]]:
    """Explicit, pinned torchvision ImageNet-1K V1 initialization."""
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    provenance = {
        "architecture": "torchvision.resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "source_url": weights.url,
        "code_license_url": "https://github.com/pytorch/vision/blob/main/LICENSE",
        "training_data": "ImageNet-1K; upstream data terms apply",
        "preprocess": "RGB letterbox 224; ImageNet mean/std; no center crop",
    }
    return backbone, provenance


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= 0 or window <= 0 or not 0 <= overlap < window:
        raise ValueError("Need positive length/window and 0 <= overlap < window")
    return list(range(0, max(1, length - overlap), window - overlap))


@torch.inference_mode()
def predict_features(model: nn.Module, features: Tensor, window: int = 256,
                     overlap: int = 64, aux: Tensor | None = None) -> Tensor:
    """Fixed-length windows, uniform overlap probability averaging, no threshold.

    ``aux`` is optional for A0/A1 and required for a feature-bank model.  It
    follows the same ``[T,F]`` frame alignment as ``features`` and is padded
    with each window so A2 cached inference uses exactly the training-side
    auxiliary definition.
    """
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Expected nonempty [T,D] feature sequence")
    if aux is not None and (aux.ndim != 2 or aux.shape[0] != features.shape[0]):
        raise ValueError("Expected aux [T,F] aligned with feature sequence")
    model.eval()
    probabilities = torch.zeros(len(features), device=features.device, dtype=torch.float32)
    counts = torch.zeros_like(probabilities)
    for start in window_starts(len(features), window, overlap):
        stop = min(start + window, len(features))
        x = features[start:stop]
        valid_length = len(x)
        x = F.pad(x, (0, 0, 0, window - len(x)))
        aux_window = None
        if aux is not None:
            aux_window = aux[start:stop]
            aux_window = F.pad(aux_window, (0, 0, 0, window - valid_length))
        # Keep the model's temporal geometry tied to the valid portion.  A
        # right-padded tail must not alter GroupNorm statistics or pooling
        # behavior for the logits that will be emitted.
        lengths = torch.tensor([valid_length], dtype=torch.long, device=x.device)
        result = model(x.unsqueeze(0),
                       aux_window.unsqueeze(0) if aux_window is not None else None,
                       lengths=lengths).float().sigmoid()[0, :valid_length]
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite temporal inference output")
        probabilities[start:stop] += result
        counts[start:stop] += 1
    if not torch.all(counts > 0):
        raise RuntimeError("Temporal window coverage failure")
    return probabilities / counts


def export_inference(model: A0Model, directory: str | Path,
                     config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Export *alternative* complete FP32/FP16 inference bundles and audit bytes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    original = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        state = {key: value.to(dtype) if value.is_floating_point() else value
                 for key, value in original.items()}
        architecture = ("A2_resnet18_featurebank_tunet" if model.feature_bank_enabled else
                        "A1_resnet18_tsm_tunet" if model.temporal_shift_enabled else "A0_resnet18_tunet")
        bundle = {"format_version": 1, "architecture": architecture,
                  "state_dict": state, "config": copy.deepcopy(config),
                  "backbone_provenance": provenance, "persistent_dtype": name,
                  "temporal_shift_enabled": model.temporal_shift_enabled,
                  "feature_bank_enabled": model.feature_bank_enabled}
        path = directory / f"model_{name}.pt"
        torch.save(bundle, path)
        size = path.stat().st_size
        mb = size / 1_000_000
        outputs[name] = {
            "path": str(path), "bytes": size, "MB_decimal": mb,
            "MiB_binary": size / (1024**2), "sha256": sha256_file(path),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "components": ["ResNet18 without classifier", "TemporalUNet including adapter"] +
                          (["parameter-free temporal shift"] if model.temporal_shift_enabled else []) +
                          (["32D feature-bank MLP"] if model.feature_bank_enabled else []),
            "size_tier_decimal_assumption": "S" if mb <= 100 else "M" if mb <= 500 else "L" if mb <= 9216 else "invalid",
            "size_coefficient_decimal_assumption": 1.0 if mb <= 100 else .95 if mb <= 500 else .9 if mb <= 9216 else None,
        }
    return {"alternatives_not_simultaneously_loaded": True, "exports": outputs,
            "default_inference": "fp32", "int8": None,
            "mb_unit_note": "Official MB byte basis remains unconfirmed; report bytes and both units."}


def load_inference_model(path: str | Path, device: str | torch.device = "cpu"
                         ) -> tuple[A0Model, dict[str, Any]]:
    """Offline loader: the single file includes all actually used model weights."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") == 1 and bundle.get("architecture") == "B0_vit_b16_temporal_probe":
        from .b0 import load_b0_bundle
        return load_b0_bundle(path, device)
    if bundle.get("format_version") == 1 and bundle.get("architecture") == "Bs0_deit_s_temporal":
        from .smallvit import load_deit_bundle
        return load_deit_bundle(path, device)
    if bundle.get("format_version") != 1 or bundle.get("architecture") not in {
        "A0_resnet18_tunet", "A1_resnet18_tsm_tunet", "A2_resnet18_featurebank_tunet"
    }:
        raise ValueError("Unsupported inference bundle")
    model = A0Model(temporal_shift_enabled=bool(
        bundle.get("temporal_shift_enabled", bundle.get("architecture") == "A1_resnet18_tsm_tunet")
    ), feature_bank_enabled=bool(bundle.get("feature_bank_enabled", bundle.get("architecture") == "A2_resnet18_featurebank_tunet")))
    model.load_state_dict(bundle["state_dict"], strict=True)
    # CPU FP16 convolution support is uneven; explicit FP32 runtime conversion
    # does not change the persisted file used for this candidate.
    model.to(device=device, dtype=torch.float32).eval()
    metadata = {key: value for key, value in bundle.items() if key != "state_dict"}
    metadata.update({"loaded_path": str(path), "loaded_bytes": Path(path).stat().st_size,
                     "loaded_sha256": sha256_file(path), "runtime_dtype": "float32"})
    return model, metadata
