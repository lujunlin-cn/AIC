import json
import math
from pathlib import Path

import pytest

from aic.contract import (ContractError, LetterboxTransform, VideoMetadata,
                          center_crop, load_jsonl, validate_submission,
                          write_submission)


def metadata(**kwargs):
    values = dict(video_id="v", width=100, height=50, frame_count=10,
                  target_ratio=(16, 9))
    values.update(kwargs)
    return VideoMetadata(**values)


def row(video_id="v", predictions=None, ratio=(16, 9), size=None):
    result = {"video_id": video_id, "targetRatioWH": list(ratio),
              "predictions": [] if predictions is None else predictions}
    if size is not None:
        result["model_size_mb"] = size
    return result


def pred(frame=0, crop=(0, 0, 10)):
    return {"frame": frame, "bboxes": list(crop)}


def test_valid_empty_predictions_and_zero_based_bounds():
    report = validate_submission([row(size=50)], {"v": metadata()}, stage="final")
    assert report.valid
    assert report.prediction_count == 0

    assert validate_submission([row(predictions=[pred(9)])], {"v": metadata()}).valid
    assert not validate_submission([row(predictions=[pred(10)])], {"v": metadata()}).valid


def test_all_videos_covered_and_no_duplicate_rows():
    index = {"v": metadata(), "w": metadata(video_id="w")}
    missing = validate_submission([row()], index)
    assert any(i.code == "missing_video" for i in missing.issues)
    duplicate = validate_submission([row(), row()], {"v": metadata()})
    assert any(i.code == "duplicate_video" for i in duplicate.issues)


def test_frame_duplicates_sorting_and_geometry():
    predictions = [pred(2), pred(1), pred(2), pred(3, (0, 0, 100))]
    report = validate_submission([row(predictions=predictions)], {"v": metadata()})
    codes = {issue.code for issue in report.issues}
    assert {"unsorted_frames", "duplicate_frame", "invalid_prediction"} <= codes
    assert not validate_submission([row(predictions=[pred(0, (0, 0, 100))])],
                                   {"v": metadata()}).valid


def test_nonfinite_and_strict_json_duplicate_keys(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"video_id":"v","video_id":"x"}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON object key"):
        load_jsonl(path)
    path.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="nonstandard JSON"):
        load_jsonl(path)
    assert not validate_submission([row(predictions=[{"frame": 0, "bboxes": [0, 0, math.nan]}])],
                                   {"v": metadata()}).valid


def test_size_required_and_consistent_on_final():
    index = {"v": metadata(), "w": metadata(video_id="w")}
    assert not validate_submission([row(), row("w")], index, stage="final").valid
    inconsistent = validate_submission([row(size=50), row("w", size=51)], index, stage="final")
    assert any(i.code == "inconsistent_size" for i in inconsistent.issues)
    assert not validate_submission([row(size=0)], {"v": metadata()}, stage="final").valid
    assert not validate_submission([row(size=9217)], {"v": metadata()}, stage="final").valid


def test_letterbox_forward_inverse_roundtrip():
    transform = LetterboxTransform.from_sizes(1920, 1080, 224, 224)
    box = (100.5, 20.25, 500.0, 200.0)
    mapped = transform.forward_box(box)
    recovered = transform.inverse_box(mapped)
    assert recovered == pytest.approx(box)
    assert transform.pad_top > 0


def test_center_crop_respects_target_ratio():
    crop = center_crop(1920, 1080, (9, 16))
    x, y, width = crop
    assert x >= 0 and y >= 0
    assert width * 16 / 9 <= 1080 + 1e-9
    assert x + width <= 1920 + 1e-9


def test_atomic_writer_validates_before_write(tmp_path):
    target = tmp_path / "out.jsonl"
    index = {"v": metadata()}
    write_submission(target, [row(predictions=[pred(1)], size=50)], index, stage="final", actual_model_size_mb=50)
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ContractError):
        write_submission(target, [row(predictions=[pred(2), pred(1)])], index)
    assert json.loads(target.read_text(encoding="utf-8"))["predictions"][0]["frame"] == 1
