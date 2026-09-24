"""Local formula reconstruction; not verified against an official evaluator.

Only the same video and exact original frame can match. Valid repeated frames
score once (the first valid occurrence); every submitted prediction contributes
to N_pred. Strict mode rejects invalid submissions before producing scores.
Diagnostic mode exposes the arithmetic for invalid prediction entries, without
claiming the resulting number is a valid submission or official score.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

from .contract import (ContractError, VideoMetadata, crop_to_xywh, load_index,
                       load_jsonl, size_coefficient, validate_prediction,
                       validate_submission)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Continuous xywh IoU. Inputs must have finite coordinates and positive area."""
    from .contract import finite_number
    for box in (first, second):
        if len(box) != 4 or not all(finite_number(v) for v in box) or box[2] <= 0 or box[3] <= 0:
            raise ContractError("IoU expects finite [x,y,w,h] with positive w,h")
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
        0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union = w1 * h1 + w2 * h2 - intersection
    if not union or not (union > 0):
        # Positive finite boxes can still underflow when adversarially tiny;
        # treating their overlap as zero is safer than emitting NaN/Inf.
        return 0.0
    return intersection / union


def evaluate_video(predictions: Sequence[Any], ground_truth: Sequence[Mapping[str, Any]],
                   metadata: VideoMetadata) -> dict[str, Any]:
    """Diagnostic arithmetic; GT must be valid and contain no duplicate frames.

    Empty P/R denominators are reported as null because document 01 only defines
    the F1 empty-set convention. F1 is 1 for empty/empty, otherwise 2*S/(Np+Ng).
    Temporal metrics count geometrically valid exact-frame matches, once per frame.
    """
    truth: dict[int, Mapping[str, Any]] = {}
    for prediction in ground_truth:
        errors = validate_prediction(prediction, metadata)
        if errors:
            raise ContractError(f"invalid ground truth for {metadata.video_id}: {errors}")
        frame = prediction["frame"]
        if frame in truth:
            raise ContractError("ground truth may not repeat frame IDs")
        truth[frame] = prediction
    used_valid_frames: set[int] = set()
    matched = 0
    invalid = 0
    duplicates = 0
    iou_sum = 0.0
    for prediction in predictions:
        if validate_prediction(prediction, metadata):
            invalid += 1
            continue
        frame = prediction["frame"]
        if frame in used_valid_frames:
            duplicates += 1
            continue
        used_valid_frames.add(frame)
        if frame in truth:
            iou_sum += box_iou(crop_to_xywh(prediction["bboxes"], metadata.target_ratio),
                               crop_to_xywh(truth[frame]["bboxes"], metadata.target_ratio))
            matched += 1
    count_pred, count_gt = len(predictions), len(truth)
    count_total = count_pred + count_gt
    return {
        "video_id": metadata.video_id,
        "n_pred": count_pred, "n_gt": count_gt,
        "iou_sum": iou_sum,
        "precision": iou_sum / count_pred if count_pred else None,
        "recall": iou_sum / count_gt if count_gt else None,
        "f1": 2 * iou_sum / count_total if count_total else 1.0,
        "temporal_precision": matched / count_pred if count_pred else None,
        "temporal_recall": matched / count_gt if count_gt else None,
        "temporal_f1": 2 * matched / count_total if count_total else 1.0,
        "matched_frames": matched,
        "matched_crop_iou": iou_sum / matched if matched else None,
        "false_positive_count": count_pred - matched,
        "false_negative_count": count_gt - matched,
        "invalid_prediction_count": invalid,
        "duplicate_valid_prediction_count": duplicates,
    }


def evaluate_submission(
    rows: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    index: Mapping[str, VideoMetadata],
    *,
    stage: str = "preliminary",
    strict: bool = True,
    actual_model_size_mb: float | None = None,
) -> dict[str, Any]:
    """Macro average over every indexed video; never silently omit invalid rows.

    Diagnostic mode tolerates invalid prediction entries, duplicate prediction
    frames, and model-size errors. Broken video-level structure/coverage/ratio is
    rejected even in diagnostic mode because scoring it would require inventing
    a rule. Missing/invalid model size yields null final score in diagnostic mode.
    """
    # Ground-truth order has no semantic meaning; submission order is checked
    # separately because deterministic sorted output is part of our exporter.
    gt_report = validate_submission(ground_truth, index, stage="preliminary",
                                   require_sorted=False)
    gt_report.raise_for_errors()
    report = validate_submission(rows, index, stage=stage,
                                 actual_model_size_mb=actual_model_size_mb)
    stage = report.stage
    if strict:
        report.raise_for_errors()
    else:
        tolerated = {"invalid_prediction", "duplicate_frame", "nonfinite", "model_size",
                     "model_size_missing", "inconsistent_size", "partial_size", "size_mismatch"}
        for issue in report.issues:
            if issue.severity == "error" and issue.code not in tolerated:
                raise ContractError(f"cannot score invalid video-level structure: {issue.code}: {issue.message}")
    pred_by_id = {row["video_id"]: row for row in rows}
    truth_by_id = {row["video_id"]: row for row in ground_truth}
    video_metrics = [evaluate_video(pred_by_id[key]["predictions"], truth_by_id[key]["predictions"], meta)
                     for key, meta in index.items()]
    f_video = sum(row["f1"] for row in video_metrics) / len(video_metrics)
    size_errors = {"model_size", "model_size_missing", "inconsistent_size", "partial_size", "size_mismatch"}
    coefficient = 1.0 if stage == "preliminary" else None
    if stage == "final" and report.model_size_mb is not None and not any(
        issue.code in size_errors for issue in report.issues
    ):
        coefficient = size_coefficient(report.model_size_mb)
    return {
        "metric_version": "local_rules_2026-09-24_not_official",
        "official_evaluator_verified": False,
        "submission_valid": report.valid,
        "mode": "strict" if strict else "diagnostic_only",
        "stage": stage,
        "video_count": len(video_metrics),
        "f_video": f_video,
        "size_coefficient": coefficient,
        "model_size_mb": report.model_size_mb,
        "score": 100 * f_video * coefficient if coefficient is not None and report.valid else None,
        "diagnostic_score": 100 * f_video * coefficient if coefficient is not None else None,
        "videos": video_metrics,
        "validation": report.to_dict(),
        "conventions": {
            "spatial": "continuous_pixel_xywh_no_rounding",
            "temporal": "exact_original_zero_based_frame",
            "aggregation": "unweighted_video_macro_average",
            "empty_precision_recall": "null_when_denominator_zero",
            "empty_f1": "both_empty_1_exactly_one_empty_0",
            "invalidity": "official_prediction_vs_whole_submission_rejection_boundary_unconfirmed",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--stage", choices=["preliminary", "initial", "prelim", "final", "replay", "semifinal"], default="preliminary")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = evaluate_submission(load_jsonl(args.submission), load_jsonl(args.ground_truth),
                                     load_index(args.index), stage=args.stage, strict=not args.diagnostic)
        rendered = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0 if result["submission_valid"] else 2
    except (ContractError, OSError) as error:
        print(json.dumps({"submission_valid": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
