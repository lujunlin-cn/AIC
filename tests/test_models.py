import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aic.features import (FeatureCacheDataset, align_labels, collate_feature_batch,
                          load_feature_cache, save_feature_cache)
from aic.models import A0Model, TemporalUNet, export_inference, load_inference_model, predict_features, temporal_shift


def test_temporal_unet_variable_short_lengths_and_finite():
    torch.manual_seed(0)
    model = TemporalUNet(512).eval()
    for length in (1, 2, 3, 4, 7, 16):
        result = model(torch.randn(2, length, 512))
        assert result.shape == (2, length)
        assert torch.isfinite(result).all()


def test_feature_cache_roundtrip_and_label_alignment(tmp_path: Path):
    indices = np.array([0, 5, 9], dtype=np.int64)
    labels, mask = align_labels({"frame_indices": np.array([5, 9]),
                                 "scores": np.array([3., 5.])}, indices)
    assert np.allclose(labels, [0, .5, 1])
    assert mask.tolist() == [False, True, True]
    path = tmp_path / "one.npz"
    save_feature_cache(path, np.zeros((3, 512), np.float32), indices,
                       np.array([0., .5, 1.]), labels, mask, {"video_id": "one"})
    item = load_feature_cache(path)
    assert item["features"].shape == (3, 512)
    ds = FeatureCacheDataset([{"path": str(path), "video_id": "one"}])
    batch = collate_feature_batch([ds[0]])
    assert batch["features"].shape == (1, 3, 512)
    assert batch["mask"].sum() == 2


def test_inference_export_is_complete_alternatives(tmp_path: Path):
    torch.manual_seed(1)
    model = A0Model().eval()
    audit = export_inference(model, tmp_path, {"run_id": "unit"}, {"source": "unit"})
    assert audit["alternatives_not_simultaneously_loaded"]
    assert audit["exports"]["fp32"]["bytes"] > 0
    loaded, metadata = load_inference_model(tmp_path / "model_fp32.pt")
    assert metadata["runtime_dtype"] == "float32"
    result = loaded(torch.randn(1, 5, 512))
    assert result.shape == (1, 5)


def test_windowed_prediction_covers_all_frames():
    model = TemporalUNet(512).eval()
    result = predict_features(model, torch.randn(17, 512), window=8, overlap=2)
    assert result.shape == (17,)
    assert torch.isfinite(result).all()


def test_temporal_shift_is_parameter_free_and_loader_roundtrips(tmp_path: Path):
    x = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    shifted = temporal_shift(x, fold_div=4)
    assert shifted.shape == x.shape and torch.isfinite(shifted).all()
    model = A0Model(temporal_shift_enabled=True).eval()
    audit = export_inference(model, tmp_path, {"run_id": "A1_001"}, {"source": "unit"})
    assert audit["exports"]["fp32"]["parameter_count"] == sum(p.numel() for p in model.parameters())
    loaded, _ = load_inference_model(tmp_path / "model_fp32.pt")
    assert loaded.temporal_shift_enabled


def test_feature_bank_fusion_has_finite_output_and_extra_head():
    model = A0Model(temporal_shift_enabled=True, feature_bank_enabled=True).eval()
    result = model(torch.randn(2, 7, 512), torch.randn(2, 7, 32))
    assert result.shape == (2, 7) and torch.isfinite(result).all()
    assert sum(p.numel() for p in model.parameters()) > 12807489
