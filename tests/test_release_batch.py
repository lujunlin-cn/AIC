import json
from pathlib import Path

from scripts.release_batch import build_index


def test_release_batch_builds_raw_video_index_without_labels(tmp_path, monkeypatch):
    first = tmp_path / "eval_a.mp4"
    second = tmp_path / "eval_b.mp4"
    first.write_bytes(b"video-a")
    second.write_bytes(b"video-b")

    def fake_probe(path):
        return type("Info", (), {"width": 320, "height": 180,
                                  "frame_count": 17, "fps": 5.0})()

    monkeypatch.setattr("aic.video.probe_video", fake_probe)
    index = build_index([first, second], (9, 16), tmp_path / "index.jsonl")
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    assert [row["video_id"] for row in rows] == ["eval_a", "eval_b"]
    assert rows[0]["targetRatioWH"] == [9.0, 16.0]
    assert rows[0]["frame_count"] == 17
    assert "labels_path" not in rows[0]
