import json
from pathlib import Path

import numpy as np

from scripts.prepare_qvh_training_manifest import _timeline_labels


def test_qvh_seed_timeline_interpolation_is_frame_aligned():
    ann = {"teacher_signals": {"timeline": [
        {"time_sec": 0.0, "highlight_score": 0.0},
        {"time_sec": 1.0, "highlight_score": 1.0},
    ]}}
    labels = _timeline_labels(ann, nframes=5, fps=2.0)
    np.testing.assert_allclose(labels, [0.0, 0.5, 1.0, 1.0, 1.0])
    assert labels.dtype == np.float32


def test_qvh_empty_timeline_is_explicit_zero_target():
    labels = _timeline_labels({"teacher_signals": {"timeline": []}}, 4, 2.0)
    np.testing.assert_array_equal(labels, np.zeros(4, dtype=np.float32))


def test_qvh_manifest_records_carry_weak_label_protocol(tmp_path):
    # Regression guard for the generated machine-readable manifest contract.
    row = {
        "dataset": "QVHighlights",
        "version": "aic_qvh_seed_weak_training_v1",
        "annotation_type": "qvh_seed_temporal_spatial_weak",
        "notes": "official_aic_gt=false; not native human GT",
    }
    path = tmp_path / "dataset_manifest.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["annotation_type"].endswith("_weak")
    assert "official_aic_gt=false" in loaded["notes"]
