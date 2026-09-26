import json
import subprocess
import pytest

from aic.index import enrich_index, load_compact_index
from aic.contract import ContractError, load_index, validate_submission_file


def test_compact_index_is_enriched_from_decoded_video(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    video = video_dir / "0.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x32:rate=4:duration=1", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(video)], check=True)
    compact = tmp_path / "test_index.json"
    compact.write_text(json.dumps([{"video_id": "0", "targetRatioWH": [16, 9]}]))
    output = tmp_path / "index.jsonl"
    result = enrich_index(compact, video_dir, output)
    assert result["videos"] == 1
    row = json.loads(output.read_text().splitlines()[0])
    assert row["frame_count"] == 4
    assert row["width"] == 64 and row["height"] == 32
    assert row["index_schema_version"] == "aic.input_index.v1"
    assert load_compact_index(compact)[0]["video_id"] == "0"


def test_compact_index_rejects_nonfinite_ratio_duplicate_keys_and_blank_lines(tmp_path):
    bad_ratio = tmp_path / "bad.json"
    bad_ratio.write_text('{"video_id":"v","targetRatioWH":[NaN,9]}')
    with pytest.raises(ValueError, match="nonstandard JSON"):
        load_compact_index(bad_ratio)
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"video_id":"v","targetRatioWH":[1,1]}\n'
                         '{"video_id":"v","targetRatioWH":[1,1]}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_compact_index(duplicate)
    blank = tmp_path / "blank.jsonl"
    blank.write_text('{"video_id":"v","targetRatioWH":[1,1]}\n\n')
    with pytest.raises(ValueError, match="blank"):
        load_compact_index(blank)


def test_contract_load_index_accepts_json_array_and_batch_validator(tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps([{"video_id": "v", "width": 10, "height": 10,
                                  "frame_count": 1, "targetRatioWH": [1, 1]}]))
    assert load_index(index)["v"].frame_count == 1
    submission = tmp_path / "submission.jsonl"
    submission.write_text(json.dumps({"video_id": "v", "targetRatioWH": [1, 1],
                                      "predictions": []}) + "\n")
    assert validate_submission_file(submission, index).valid
