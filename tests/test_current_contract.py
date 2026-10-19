"""Synthetic rule regression tests; no competition data or labels are read."""

import json
import math

import pytest

from aic.contract import (CURRENT_RULE_VERSION, LEGACY_RULE_VERSION, ContractError,
                          VideoMetadata, load_jsonl, main as contract_main,
                          validate_submission, validate_submission_file,
                          write_jsonl, write_submission)
from aic.evaluation import evaluate_submission, main as evaluation_main


def index(*video_ids):
    return {video_id: VideoMetadata(video_id, 100, 100, 20, (1, 1))
            for video_id in (video_ids or ("synthetic",))}


def row(video_id="synthetic", **fields):
    return {"video_id": video_id, "targetRatioWH": [1, 1],
            "predictions": [{"frame": 0, "bboxes": [0, 0, 10]}], **fields}


def codes(report):
    return {issue.code for issue in report.errors}


@pytest.mark.parametrize("stage", ["current", "semifinal", "semi-final"])
@pytest.mark.parametrize("params", [0, 0.5, 9])
def test_current_aliases_accept_full_inclusive_parameter_range(stage, params):
    report = validate_submission([row(model_params_b=params)], index(), stage=stage)
    assert report.valid
    assert report.stage == "current"
    assert report.model_params_b == params
    assert report.model_size_mb is None
    assert report.to_dict()["validator"] == CURRENT_RULE_VERSION
    assert report.to_dict()["official_validator_verified"] is False


@pytest.mark.parametrize("params", [-1e-9, 9.000000001, True, False, "8.5", None,
                                  math.nan, math.inf, -math.inf, 10 ** 1000])
def test_current_rejects_invalid_parameter_declarations(params):
    report = validate_submission([row(model_params_b=params)], index(), stage="current")
    assert not report.valid
    assert "model_params" in codes(report)


@pytest.mark.parametrize("fields", [{}, {"model_size_mb": 1}, {"model_size_mb": 18000}])
def test_storage_metadata_cannot_replace_required_parameter_count(fields):
    report = validate_submission([row(**fields)], index(), stage="semifinal")
    assert "model_params_missing" in codes(report)
    with pytest.raises(ContractError, match="model_params_missing"):
        evaluate_submission([row(**fields)], [row()], index(), stage="semifinal")


@pytest.mark.parametrize("size", [0, 9216, 18000, 36000])
def test_current_storage_bytes_do_not_control_eligibility(size):
    report = validate_submission([row(model_params_b=9, model_size_mb=size)], index(),
                                 stage="current", actual_model_size_mb=size)
    assert report.valid
    assert report.model_size_mb == size


@pytest.mark.parametrize("size", [-1, True, False, "18000", None, math.nan, math.inf])
def test_current_storage_metadata_still_requires_nonnegative_finite_number(size):
    report = validate_submission([row(model_params_b=9, model_size_mb=size)], index(),
                                 stage="current")
    assert "model_size" in codes(report)


def test_current_parameter_declaration_must_cover_all_videos_consistently():
    metadata = index("a", "b")
    report = validate_submission([row("a", model_params_b=8), row("b", model_params_b=9)],
                                 metadata, stage="current")
    assert "inconsistent_params" in codes(report)
    partial = validate_submission([row("a", model_params_b=8), row("b")], metadata,
                                  stage="current")
    assert {"partial_params", "model_params_missing"} <= codes(partial)
    consistent = validate_submission([row("a", model_params_b=8), row("b", model_params_b=8.0)],
                                     metadata, stage="current")
    assert consistent.valid


def test_current_optional_storage_metadata_requires_consistency_if_present():
    metadata = index("a", "b")
    rows = [row("a", model_params_b=9, model_size_mb=18000),
            row("b", model_params_b=9, model_size_mb=18001)]
    assert "inconsistent_size" in codes(validate_submission(rows, metadata, stage="current"))
    del rows[1]["model_size_mb"]
    assert "partial_size" in codes(validate_submission(rows, metadata, stage="current"))


def test_current_declared_values_must_match_audited_actual_values():
    valid = validate_submission([row(model_params_b=8.480646545, model_size_mb=18000)],
                                index(), stage="current", actual_model_params_b=8.480646545,
                                actual_model_size_mb=18000)
    assert valid.valid
    # One missing parameter is larger than the float round-trip tolerance.
    mismatch = validate_submission([row(model_params_b=8.480646544, model_size_mb=17999)],
                                   index(), stage="current", actual_model_params_b=8.480646545,
                                   actual_model_size_mb=18000)
    assert {"params_mismatch", "size_mismatch"} <= codes(mismatch)
    with pytest.raises(ContractError, match="params_mismatch"):
        evaluate_submission([row(model_params_b=8)], [row()], index(), stage="current",
                            actual_model_params_b=8.5)


@pytest.mark.parametrize("actual", [9.000000001, -1, True, False, math.nan, math.inf])
def test_invalid_measured_parameter_count_is_rejected_before_comparison(actual):
    with pytest.raises(ContractError, match="model_params_b"):
        validate_submission([row(model_params_b=9)], index(), stage="current",
                            actual_model_params_b=actual)


@pytest.mark.parametrize("size", [-1, True, math.inf])
def test_invalid_measured_storage_size_is_rejected(size):
    with pytest.raises(ContractError, match="model_size_mb"):
        validate_submission([row(model_params_b=9)], index(), stage="current",
                            actual_model_size_mb=size)


@pytest.mark.parametrize("stage", ["current", "semifinal", "semi-final"])
def test_current_perfect_prediction_at_9b_scores_100_without_size_penalty(stage):
    rows = [row(model_params_b=9, model_size_mb=36000)]
    # A synthetic truth exporter may include metadata. Ground-truth validation
    # must not accidentally apply the historical byte limit to those fields.
    result = evaluate_submission(rows, rows, index(), stage=stage, actual_model_params_b=9)
    assert result["f_video"] == pytest.approx(1)
    assert result["score"] == pytest.approx(100)
    assert result["size_coefficient"] == 1
    assert result["model_params_b"] == 9
    assert result["score_formula"] == "100*f_video"
    assert result["metric_version"] == CURRENT_RULE_VERSION
    assert result["official_evaluator_verified"] is False


def test_current_continuous_iou_exact_frames_and_video_macro_average():
    metadata = index("a", "b")
    truth = [row("a"), row("b")]
    rows = [row("a", model_params_b=9,
                predictions=[{"frame": 0, "bboxes": [5, 0, 10]}]),
            row("b", model_params_b=9,
                predictions=[{"frame": 1, "bboxes": [0, 0, 10]}])]
    result = evaluate_submission(rows, truth, metadata, stage="current")
    assert result["videos"][0]["f1"] == pytest.approx(1 / 3)
    assert result["videos"][1]["f1"] == 0
    assert result["score"] == pytest.approx(100 / 6)


def test_zero_parameter_method_can_be_valid_and_score_100():
    rows = [row(model_params_b=0, model_size_mb=0)]
    result = evaluate_submission(rows, rows, index(), stage="current", actual_model_params_b=0)
    assert result["score"] == 100


def test_invalid_current_parameters_have_no_eligible_score_in_diagnostic_mode():
    result = evaluate_submission([row(model_params_b=10)], [row()], index(),
                                 stage="current", strict=False)
    assert result["submission_valid"] is False
    assert result["score"] is None
    assert result["mode"] == "diagnostic_only"
    assert result["diagnostic_score"] == 100
    assert result["official_evaluator_verified"] is False


@pytest.mark.parametrize("stage,normalized,expected", [
    ("preliminary", "preliminary", 100), ("prelim", "preliminary", 100),
    ("initial", "preliminary", 100), ("final", "final", 95), ("replay", "final", 95),
])
def test_explicit_historical_stages_keep_their_original_score(stage, normalized, expected):
    rows = [row(model_size_mb=200)]
    result = evaluate_submission(rows, rows, index(), stage=stage)
    assert result["score"] == expected
    assert result["stage"] == normalized
    assert result["metric_version"] == LEGACY_RULE_VERSION


@pytest.mark.parametrize("stage", ["final", "replay"])
def test_historical_final_replay_still_require_positive_size_with_9216_cap(stage):
    assert "model_size_missing" in codes(validate_submission(
        [row(model_params_b=9)], index(), stage=stage))
    for invalid_size in (0, 9217):
        report = validate_submission([row(model_size_mb=invalid_size)], index(), stage=stage)
        assert "model_size" in codes(report)
    assert validate_submission([row(model_size_mb=9216)], index(), stage=stage).valid


def test_export_populates_audited_parameters_and_preserves_file_on_invalid_rewrite(tmp_path):
    target = tmp_path / "predictions.jsonl"
    report = write_submission(target, [row()], index(), stage="current",
                              actual_model_params_b=9, actual_model_size_mb=36000)
    assert report.valid
    assert load_jsonl(target)[0]["model_params_b"] == 9
    original_bytes = target.read_bytes()
    with pytest.raises(ContractError, match="params_mismatch"):
        write_submission(target, [row(model_params_b=8)], index(), stage="current",
                         actual_model_params_b=9)
    assert target.read_bytes() == original_bytes


def test_current_file_and_cli_entry_points_use_parameter_contract(tmp_path, capsys):
    metadata_path = tmp_path / "index.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    truth_path = tmp_path / "synthetic_truth.jsonl"
    write_jsonl(metadata_path, [{"video_id": "synthetic", "width": 100, "height": 100,
                                "frame_count": 20, "targetRatioWH": [1, 1]}])
    write_jsonl(prediction_path, [row(model_params_b=9)])
    write_jsonl(truth_path, [row()])
    assert validate_submission_file(prediction_path, metadata_path, stage="current",
                                    actual_model_params_b=9).valid
    common_args = ["--index", str(metadata_path), "--submission", str(prediction_path),
                   "--stage", "semifinal", "--actual-model-params-b", "9"]
    assert contract_main(["validate", *common_args]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["stage"] == "current"
    assert validation["model_params_b"] == 9
    assert evaluation_main([*common_args, "--ground-truth", str(truth_path)]) == 0
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["score"] == 100
    assert evaluation["official_evaluator_verified"] is False
