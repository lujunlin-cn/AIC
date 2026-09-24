import pytest

from aic.contract import VideoMetadata
from aic.evaluation import box_iou, evaluate_submission, evaluate_video


def metadata(video_id="v"):
    return VideoMetadata(video_id, 100, 100, 20, (1, 1))


def row(video_id="v", predictions=None, size=None):
    result = {"video_id": video_id, "targetRatioWH": [1, 1],
              "predictions": [] if predictions is None else predictions}
    if size is not None:
        result["model_size_mb"] = size
    return result


def pred(frame, crop=(0, 0, 10)):
    return {"frame": frame, "bboxes": list(crop)}


def test_iou_exact_and_disjoint():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1)
    assert box_iou((0, 0, 10, 10), (20, 20, 1, 1)) == 0
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_exact_frame_matching_and_continuous_iou():
    index = {"v": metadata()}
    result = evaluate_submission([row(predictions=[pred(1, (0, 0, 10))])],
                                 [row(predictions=[pred(1, (0, 0, 10))])], index)
    assert result["f_video"] == pytest.approx(1)
    wrong_frame = evaluate_submission([row(predictions=[pred(2)])],
                                      [row(predictions=[pred(1)])], index)
    assert wrong_frame["f_video"] == 0


def test_macro_average_and_empty_cases():
    index = {"empty": metadata("empty"), "nonempty": metadata("nonempty")}
    result = evaluate_submission([row("empty"), row("nonempty", [pred(0)])],
                                 [row("empty"), row("nonempty", [pred(0)])], index)
    assert result["f_video"] == pytest.approx(1)
    only_empty = evaluate_submission([row()], [row()], {"v": metadata()})
    assert only_empty["f_video"] == 1
    one_side = evaluate_submission([row()], [row(predictions=[pred(0)])], {"v": metadata()})
    assert one_side["f_video"] == 0


def test_duplicate_scores_first_valid_but_all_predictions_denominator():
    index = {"v": metadata()}
    truth = [row(predictions=[pred(0)])]
    predictions = [pred(0), pred(0), pred(2)]
    metrics = evaluate_video(predictions, truth[0]["predictions"], index["v"])
    assert metrics["matched_frames"] == 1
    assert metrics["duplicate_valid_prediction_count"] == 1
    assert metrics["n_pred"] == 3
    assert metrics["f1"] == pytest.approx(2 / 4)


def test_invalid_predictions_are_diagnostic_but_strict_rejects():
    index = {"v": metadata()}
    truth = [row(predictions=[pred(0)])]
    invalid = [row(predictions=[pred(99)])]
    with pytest.raises(Exception):
        evaluate_submission(invalid, truth, index, strict=True)
    result = evaluate_submission(invalid, truth, index, strict=False)
    assert result["submission_valid"] is False
    assert result["diagnostic_score"] == 0
    assert result["videos"][0]["invalid_prediction_count"] == 1


def test_size_weighting_and_unverified_marker():
    index = {"v": metadata()}
    truth = [row(predictions=[pred(0)], size=200)]
    result = evaluate_submission(truth, truth, index, stage="final")
    assert result["size_coefficient"] == pytest.approx(0.95)
    assert result["score"] == pytest.approx(95)
    assert result["official_evaluator_verified"] is False
