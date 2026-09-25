import json
from pathlib import Path

from scripts.validate_local_protocol import validate


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_local_protocol_matches_manifest():
    protocol_path = ROOT / "splits" / "local_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = validate(protocol_path, ROOT / protocol["manifest_path"])
    assert report["valid"] is True
    assert report["counts"] == {"train": 27, "dev": 16, "lockbox": 7}
    assert report["fold_counts"] == {
        "fold0": 8, "fold1": 9, "fold2": 10, "fold3": 7, "fold4": 9
    }
    assert report["lockbox_frozen"] is True


def test_lockbox_is_disjoint_from_development_pool():
    protocol = json.loads(
        (ROOT / "splits" / "local_protocol_v1.json").read_text(encoding="utf-8")
    )
    train = set(protocol["train_video_ids"])
    dev = set(protocol["dev_video_ids"])
    lockbox = set(protocol["lockbox_video_ids"])
    assert not train & dev
    assert not train & lockbox
    assert not dev & lockbox
    assert len(train | dev | lockbox) == 50
