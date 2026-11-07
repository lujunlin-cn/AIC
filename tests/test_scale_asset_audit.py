import json
from pathlib import Path

from scripts.audit_scale_assets import audit_live


def test_live_audit_requires_complete_train_val_manifest_and_rejects_test_rows(tmp_path: Path):
    dataset = tmp_path / ".cache_videoora_data/LIVE_YT_VC"
    dataset.mkdir(parents=True)
    labels = dataset / "video_bbox_labels.csv"
    labels.write_text("video,frame0\nsource-a,\nsource-b,\n", encoding="utf-8")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"source-a": "train", "source-b": "dev"}), encoding="utf-8")

    accepted = audit_live(tmp_path, manifest)
    assert accepted["source_level_gate"] is True
    assert accepted["source_split_manifest"]["source_counts"] == {"train": 1, "val": 1}
    assert accepted["source_split_manifest"]["invalid_split_sources"] == 0

    manifest.write_text(json.dumps({"source-a": "train", "source-b": "test"}), encoding="utf-8")
    rejected = audit_live(tmp_path, manifest)
    assert rejected["source_level_gate"] is False
    assert rejected["source_split_manifest"]["invalid_split_sources"] == 1
