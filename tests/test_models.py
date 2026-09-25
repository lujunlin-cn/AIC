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


def test_temporal_unet_padding_regression_requires_lengths():
    """Right padding must not change valid logits when true lengths are given.

    The pre-fix path passed a max-length padded batch directly through
    GroupNorm and the two temporal pooling stages.  Consequently a 5-step
    sequence produced different valid logits when padded to 8 or 12 steps,
    and also differed when batched beside a longer video.  ``lengths`` makes
    the production train/inference path evaluate each sequence at its true
    geometry and keeps this behavior invariant.
    """
    torch.manual_seed(7)
    model = TemporalUNet(8).eval()
    valid = torch.randn(1, 5, 8)
    standalone = model(valid)

    for padded_length in (8, 12, 17):
        padded = torch.cat([valid, torch.zeros(1, padded_length - 5, 8)], dim=1)
        result = model(padded, lengths=torch.tensor([5]))
        assert torch.allclose(result[:, :5], standalone, atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(result[:, 5:]) == 0

    longer = torch.randn(1, 12, 8)
    batch = torch.cat([torch.cat([valid, torch.zeros(1, 7, 8)], dim=1), longer], dim=0)
    result = model(batch, lengths=torch.tensor([5, 12]))
    assert torch.allclose(result[0, :5], standalone[0], atol=1e-6, rtol=1e-6)


def test_temporal_unet_padding_difference_is_detectable_without_lengths():
    """Guard the regression fixture: omitted lengths remains observably unsafe."""
    torch.manual_seed(7)
    model = TemporalUNet(8).eval()
    valid = torch.randn(1, 5, 8)
    padded = torch.cat([valid, torch.zeros(1, 7, 8)], dim=1)
    standalone = model(valid)
    unsafe = model(padded)[:, :5]
    assert not torch.allclose(unsafe, standalone, atol=1e-5, rtol=1e-5)


def test_feature_bank_short_sequence_aux_padding_is_consistent():
    torch.manual_seed(8)
    model = TemporalUNet(8, aux_dim=32).eval()
    features = torch.randn(1, 2, 8)
    aux = torch.randn(1, 2, 32)
    standalone = model(features, aux)
    padded = torch.cat([features, torch.zeros(1, 5, 8)], dim=1)
    padded_aux = torch.cat([aux, torch.zeros(1, 5, 32)], dim=1)
    result = model(padded, padded_aux, lengths=torch.tensor([2]))
    assert result.shape == (1, 7)
    assert torch.allclose(result[:, :2], standalone, atol=1e-6, rtol=1e-6)


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


def test_windowed_feature_bank_prediction_accepts_aligned_aux():
    model = A0Model(feature_bank_enabled=True).eval()
    result = predict_features(model, torch.randn(9, 512), window=5, overlap=2,
                              aux=torch.randn(9, 32))
    assert result.shape == (9,)
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
