import json
import subprocess

import numpy as np
import torch

import aic.models
from aic.inference import run_inference
from aic.contract import load_jsonl
from aic.features import _letterbox, compute_feature_bank
from aic.inference import _normalise, _spatial_crop
from aic.video import iter_sampled_frames


def _video(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=80x40:rate=5:duration=1", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_dummy_raw_video_to_valid_jsonl(tmp_path):
    video = tmp_path / "one.mp4"
    _video(video)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"video_id": "one", "video_path": str(video),
                                 "width": 80, "height": 40, "frame_count": 5,
                                 "targetRatioWH": [9, 16]}) + "\n")
    output = tmp_path / "submission.jsonl"
    result = run_inference(index, output, dummy=True, stage="preliminary")
    assert result["validation"]["valid"]
    row = load_jsonl(output)[0]
    assert len(row["predictions"]) == 5
    assert row["predictions"][0]["frame"] == 0
    assert row["targetRatioWH"] == [9, 16]


def test_raw_preprocess_matches_feature_cache_preprocess(tmp_path):
    video = tmp_path / "one.mp4"
    _video(video)
    sampled = list(iter_sampled_frames(video, sample_fps=2.0, size=224))
    images = np.stack([item[2] for item in sampled])
    # These are the two independent call sites used by raw inference and
    # extract_video_cache.  Exact equality is expected before the backbone.
    assert torch.equal(_normalise(list(images)), _letterbox(images))
    first = compute_feature_bank(images)
    second = compute_feature_bank(images.copy())
    assert np.array_equal(first, second)


def test_feature_bank_aux_and_spatial_modes_reach_raw_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "one.mp4"
    _video(video)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"video_id": "one", "video_path": str(video),
                                 "width": 80, "height": 40, "frame_count": 5,
                                 "targetRatioWH": [9, 16]}) + "\n")

    class FakeA2(torch.nn.Module):
        feature_bank_enabled = True

        def encode_frames(self, images):
            return images.mean(dim=(2, 3), keepdim=False).mean(dim=1, keepdim=True).repeat(1, 512)

        def forward(self, features, aux=None):
            assert aux is not None and tuple(aux.shape) == (1, features.shape[1], 32)
            assert torch.isfinite(aux).all()
            return torch.ones(features.shape[:2], dtype=features.dtype, device=features.device)

    monkeypatch.setattr(aic.models, "load_inference_model",
                        lambda path, device="cpu": (FakeA2(), {"loaded_bytes": 1}))
    for mode in ("center", "saliency", "subject"):
        output = tmp_path / f"{mode}.jsonl"
        result = run_inference(index, output, model_path=tmp_path / "model.pt",
                               threshold=.5, spatial_mode=mode)
        assert result["validation"]["valid"]
        row = load_jsonl(output)[0]
        assert len(row["predictions"]) == 5
        for prediction in row["predictions"]:
            x, y, w = prediction["bboxes"]
            assert 0 <= x <= 80 and 0 <= y <= 40 and w > 0


def test_spatial_candidates_are_legal_and_saliency_moves_center():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:, 175:] = 255
    center = _spatial_crop(image, "center", [9, 16], 80, 40)
    saliency = _spatial_crop(image, "saliency", [9, 16], 80, 40)
    subject = _spatial_crop(image, "subject", [9, 16], 80, 40)
    assert all(np.isfinite(c) for c in center + saliency + subject)
    assert saliency[0] >= center[0]
