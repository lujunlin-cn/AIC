import json
import subprocess

from aic.index import load_compact_index


def test_compact_index_accepts_common_video_path_keys(tmp_path):
    path = tmp_path / "compact.jsonl"
    path.write_text(json.dumps({"video_id": "v", "file_name": "nested/v.mp4",
                                "targetRatioWH": [16, 9]}) + "\n")
    rows = load_compact_index(path)
    assert rows[0]["file_name"] == "nested/v.mp4"


def test_intake_version_is_explicit():
    from aic.eval_intake import INTAKE_VERSION
    assert INTAKE_VERSION == "AIC_EVAL_INTAKE_V1"


def test_eval_intake_writes_decode_verified_index_and_manifest(tmp_path):
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video = video_root / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=64x32:rate=4:duration=1", "-an",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps([{"video_id": "v", "targetRatioWH": [9, 16]}]))
    from aic.eval_intake import prepare_eval_set
    manifest = prepare_eval_set(compact, video_root, tmp_path / "index.jsonl",
                                tmp_path / "manifest.json")
    assert manifest["labels_present"] is False
    row = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
    assert row["frame_count"] == 4
    assert row["targetRatioWH"] == [9, 16]
    assert row["index_schema_version"] == "aic.input_index.v1"
    assert row["pts_verified"] is True


def test_eval_intake_refuses_raw_index_or_same_output_overwrite(tmp_path):
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps([{"video_id": "v", "targetRatioWH": [1, 1]}]))
    from aic.eval_intake import prepare_eval_set
    import pytest
    with pytest.raises(ValueError, match="raw compact index"):
        prepare_eval_set(compact, tmp_path, compact, tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="different files"):
        prepare_eval_set(compact, tmp_path, tmp_path / "same", tmp_path / "same")
