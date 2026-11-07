"""Leakage-resistant adapters for scale supervision and preference pairs."""

from __future__ import annotations

import math
from collections import defaultdict
import statistics
from typing import Iterable, Sequence

from aic.scale_geometry import (
    box_iou_xywh,
    map_box_to_scale_id,
    scale_crop_from_center,
    scale_crop_xywh_exact,
)


def assert_source_disjoint(source_splits: dict[str, Iterable[str]]) -> None:
    seen: dict[str, str] = {}
    for split, sources in source_splits.items():
        for source_id in sources:
            source_id = str(source_id)
            if source_id in seen and seen[source_id] != split:
                raise ValueError(f"source {source_id!r} leaks across {seen[source_id]} and {split}")
            seen[source_id] = split


def _required(row: dict, name: str):
    if name not in row or row[name] is None:
        raise ValueError(f"missing required field: {name}")
    return row[name]


def fixed_center_sft_target(
    row: dict,
    scale_bins: Sequence[float],
    min_deploy_iou: float,
) -> dict:
    """Choose the best scale action at a frozen position; never alter position."""
    if row.get("split") != "train":
        raise ValueError("SFT target construction is train-only")
    if not scale_bins:
        raise ValueError("scale_bins cannot be empty")
    if not math.isfinite(min_deploy_iou) or not 0 <= min_deploy_iou <= 1:
        raise ValueError("min_deploy_iou must be in [0, 1]")
    source_id = str(_required(row, "source_id"))
    dataset_source = str(_required(row, "dataset_source"))
    ratio = _required(row, "target_ratio")
    center = _required(row, "center_xy")
    gt = _required(row, "gt_crop_xywh")
    width = int(_required(row, "image_width"))
    height = int(_required(row, "image_height"))
    scored = []
    for scale_id, scale in enumerate(scale_bins):
        iou = box_iou_xywh(
            scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale)), gt
        )
        exact_box = scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale))
        scored.append((iou, scale_id, exact_box))
    best_iou, best_id, best_box = max(scored, key=lambda item: (item[0], -item[1]))
    best_submission_box = scale_crop_from_center(
        width, height, tuple(ratio), tuple(center), float(scale_bins[best_id])
    )
    return {
        "source_id": source_id,
        "image_id": str(row.get("image_id", source_id)),
        "dataset_source": dataset_source,
        "split": "train",
        "target_ratio": list(ratio),
        "position_source": "P1_frozen",
        "position_center_xy": list(center),
        "scale_id": best_id if best_iou >= min_deploy_iou else None,
        "crop_xywh": list(best_box),
        "aic_bbox_xyw": list(best_submission_box[:3]),
        "aic_bbox_xyw": list(best_box[:3]),
        "best_fixed_center_deploy_iou": best_iou,
        "label_status": "ok" if best_iou >= min_deploy_iou else "position_mismatch",
    }


def _pair_record(
    *, source_id: str, image_id: str, dataset_source: str, ratio: Sequence[int],
    position_center_xy: Sequence[float], chosen_id: int, rejected_id: int,
    chosen_crop: Sequence[float], rejected_crop: Sequence[float], chosen_aic_bbox: Sequence[int],
    rejected_aic_bbox: Sequence[int], chosen_score: float,
    rejected_score: float, preference_source: str, score_name: str,
    score_semantics: str, split: str, subject_id: str | None = None,
) -> dict:
    if split != "train":
        raise ValueError("preference construction is train-only")
    if str(source_id) == "" or str(image_id) == "":
        raise ValueError("source_id and image_id must be nonempty")
    if chosen_id == rejected_id or list(chosen_crop) == list(rejected_crop):
        raise ValueError("chosen and rejected must be distinct scale actions")
    return {
        "source_id": str(source_id),
        "image_id": str(image_id),
        "dataset_source": dataset_source,
        "split": split,
        "target_ratio": list(ratio),
        "position_source": "P1_frozen",
        "position_center_xy": list(position_center_xy),
        "subject_id": subject_id,
        "chosen": {"scale_id": chosen_id, "crop_xywh": list(chosen_crop),
                   "aic_bbox_xyw": list(chosen_aic_bbox)},
        "rejected": {"scale_id": rejected_id, "crop_xywh": list(rejected_crop),
                     "aic_bbox_xyw": list(rejected_aic_bbox)},
        "preference_source": preference_source,
        "chosen_score": float(chosen_score),
        "rejected_score": float(rejected_score),
        "score_name": score_name,
        "score_semantics": score_semantics,
        "score_delta": float(chosen_score - rejected_score),
        "valid_geometry": True,
    }


def geometry_preference_pairs(
    row: dict, scale_bins: Sequence[float], delta_iou: float, max_pairs: int
) -> list[dict]:
    """Build LIVE/GNMC preferences from same-source deployed IoU differences."""
    if row.get("split") != "train":
        raise ValueError("LIVE/GNMC preference construction is train-only")
    if not math.isfinite(delta_iou) or delta_iou <= 0:
        raise ValueError("delta_iou must be calibrated on training annotations")
    if not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    source_id = str(_required(row, "source_id"))
    image_id = str(row.get("image_id", source_id))
    ratio = _required(row, "target_ratio")
    center = _required(row, "center_xy")
    gt = _required(row, "gt_crop_xywh")
    width = int(_required(row, "image_width"))
    height = int(_required(row, "image_height"))
    actions = []
    for scale_id, scale in enumerate(scale_bins):
        crop = scale_crop_from_center(width, height, tuple(ratio), tuple(center), float(scale))
        score = box_iou_xywh(
            scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale)), gt
        )
        actions.append((scale_id, crop, score))
    pairs = []
    for index, chosen in enumerate(actions):
        for rejected in actions[index + 1:]:
            difference = chosen[2] - rejected[2]
            if difference >= delta_iou:
                preferred, dispreferred = chosen, rejected
            elif difference <= -delta_iou:
                preferred, dispreferred = rejected, chosen
            else:
                continue
            pairs.append((abs(difference), preferred, dispreferred))
    pairs.sort(key=lambda item: (-item[0], item[1][0], item[2][0]))
    return [
        _pair_record(
            source_id=source_id, image_id=image_id,
            dataset_source=str(_required(row, "dataset_source")), ratio=ratio,
            position_center_xy=center, chosen_id=chosen[0], rejected_id=rejected[0],
            chosen_crop=scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale_bins[chosen[0]])),
            rejected_crop=scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale_bins[rejected[0]])),
            chosen_aic_bbox=chosen[1][:3], rejected_aic_bbox=rejected[1][:3],
            chosen_score=chosen[2], rejected_score=rejected[2],
            preference_source="deploy_iou", score_name="IoU",
            score_semantics="same_source_fixed_center_aic_deployment_iou",
            split="train",
        )
        for _, chosen, rejected in pairs[:max_pairs]
    ]


def gaicd_scale_preferences(
    row: dict, scale_bins: Sequence[float], min_mos_delta: float, max_pairs: int,
    geometry_tolerance_px: float = 1.0,
) -> list[dict]:
    """Use MOS only when a rated crop is itself a fixed-center scale action."""
    if row.get("split") != "train":
        raise ValueError("GAICD preference construction is train-only")
    if row.get("release", "journal") != "journal":
        raise ValueError("GAICD must use the journal release split only")
    if not math.isfinite(min_mos_delta) or min_mos_delta <= 0:
        raise ValueError("min_mos_delta must be fixed from training-only noise analysis")
    if not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    candidates = _required(row, "candidates")
    width = int(_required(row, "image_width"))
    height = int(_required(row, "image_height"))
    ratio = _required(row, "target_ratio")
    center = _required(row, "center_xy")
    by_scale: dict[int, list[tuple[float, tuple[float, float, float, float]]]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.get("valid", True):
            continue
        mos = candidate.get("mos")
        if (not isinstance(mos, (int, float)) or isinstance(mos, bool)
                or not math.isfinite(mos) or mos in (-2, -1)):
            continue
        crop = candidate.get("crop_xywh")
        if not isinstance(crop, (list, tuple)) or len(crop) != 4:
            continue
        scale_id = map_box_to_scale_id(
            crop, width, height, tuple(ratio), tuple(center), scale_bins,
            tolerance_px=geometry_tolerance_px,
        )
        if scale_id is not None:
            projected = scale_crop_xywh_exact(
                width, height, tuple(ratio), tuple(center), float(scale_bins[scale_id])
            )
            by_scale[scale_id].append((float(mos), projected))
    # Repeated annotation rows for one action are aggregated, never treated as
    # independent examples.
    actions = []
    for scale_id, entries in by_scale.items():
        score = sum(item[0] for item in entries) / len(entries)
        actions.append((scale_id, entries[0][1], score))
    pairs = []
    for index, first in enumerate(actions):
        for second in actions[index + 1:]:
            difference = first[2] - second[2]
            if abs(difference) < min_mos_delta:
                continue
            chosen, rejected = (first, second) if difference > 0 else (second, first)
            pairs.append((abs(difference), chosen, rejected))
    pairs.sort(key=lambda item: (-item[0], item[1][0], item[2][0]))
    return [
        _pair_record(
            source_id=str(_required(row, "source_id")),
            image_id=str(_required(row, "image_id")), dataset_source="GAICD",
            ratio=ratio, position_center_xy=center,
            chosen_id=chosen[0], rejected_id=rejected[0],
            chosen_crop=chosen[1], rejected_crop=rejected[1],
            chosen_aic_bbox=scale_crop_from_center(width, height, tuple(ratio), tuple(center), float(scale_bins[chosen[0]]))[:3],
            rejected_aic_bbox=scale_crop_from_center(width, height, tuple(ratio), tuple(center), float(scale_bins[rejected[0]]))[:3],
            chosen_score=chosen[2], rejected_score=rejected[2],
            preference_source="mos", score_name="MOS",
            score_semantics="same_image_journal_candidate_mean_opinion_score_1_to_5",
            split="train",
        )
        for _, chosen, rejected in pairs[:max_pairs]
    ]


def sacd_scale_preferences(
    row: dict, scale_bins: Sequence[float], max_pairs: int,
    geometry_tolerance_px: float = 1.0,
) -> list[dict]:
    """Consume explicit same-image/subject ranked pairs; never infer rankings."""
    if row.get("split") != "train":
        raise ValueError("SACD preference construction is train-only")
    ranked_pairs = row.get("ranked_pairs")
    if not ranked_pairs:
        return []
    if row.get("subject_count") != 1 or not row.get("subject_id"):
        return []
    width = int(_required(row, "image_width"))
    height = int(_required(row, "image_height"))
    ratio = _required(row, "target_ratio")
    center = _required(row, "center_xy")
    result = []
    for pair in ranked_pairs:
        if pair.get("image_id") != row.get("image_id") or pair.get("subject_id") != row.get("subject_id"):
            raise ValueError("SACD ranked pair must retain the same image and subject")
        chosen_box = _required(pair, "chosen_box_xywh")
        rejected_box = _required(pair, "rejected_box_xywh")
        chosen_id = map_box_to_scale_id(
            chosen_box, width, height, tuple(ratio), tuple(center), scale_bins,
            tolerance_px=geometry_tolerance_px,
        )
        rejected_id = map_box_to_scale_id(
            rejected_box, width, height, tuple(ratio), tuple(center), scale_bins,
            tolerance_px=geometry_tolerance_px,
        )
        if chosen_id is None or rejected_id is None:
            continue
        chosen = scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale_bins[chosen_id]))
        rejected = scale_crop_xywh_exact(width, height, tuple(ratio), tuple(center), float(scale_bins[rejected_id]))
        result.append(_pair_record(
            source_id=str(_required(row, "source_id")), image_id=str(row["image_id"]),
            dataset_source="SACD", ratio=ratio, position_center_xy=center,
            chosen_id=chosen_id, rejected_id=rejected_id,
            chosen_crop=chosen, rejected_crop=rejected,
            chosen_aic_bbox=scale_crop_from_center(width, height, tuple(ratio), tuple(center), float(scale_bins[chosen_id]))[:3],
            rejected_aic_bbox=scale_crop_from_center(width, height, tuple(ratio), tuple(center), float(scale_bins[rejected_id]))[:3],
            chosen_score=1.0, rejected_score=0.0,
            preference_source="ranked_pair", score_name="human_rank",
            score_semantics="same_image_single_subject_explicit_ranked_pair",
            split="train", subject_id=str(row["subject_id"]),
        ))
    return result[:max_pairs]


def validate_preference_pair(pair: dict) -> None:
    required = (
        "source_id", "image_id", "dataset_source", "split", "target_ratio",
        "position_source", "position_center_xy", "chosen", "rejected",
        "preference_source", "score_name", "score_semantics",
    )
    missing = [field for field in required if field not in pair]
    if missing:
        raise ValueError(f"preference pair missing fields: {missing}")
    if pair["split"] != "train":
        raise ValueError("DPO training pair is not from train split")
    if pair["position_source"] != "P1_frozen":
        raise ValueError("scale preference must use the frozen P1 position")
    if pair["chosen"]["scale_id"] == pair["rejected"]["scale_id"]:
        raise ValueError("chosen and rejected scale ids must differ")
    for action in (pair["chosen"], pair["rejected"]):
        if len(action.get("crop_xywh", [])) != 4 or len(action.get("aic_bbox_xyw", [])) != 3:
            raise ValueError("preference action geometry does not match AIC contracts")
    if not pair.get("chosen_score", 0) > pair.get("rejected_score", 0):
        raise ValueError("chosen score must be greater than rejected score")
    if pair["score_name"] == "IoU" and pair["preference_source"] != "deploy_iou":
        raise ValueError("IoU score semantics are mislabeled")
    if pair["score_name"] == "MOS" and pair["preference_source"] != "mos":
        raise ValueError("MOS must remain labeled as MOS")
    if pair["preference_source"] == "ranked_pair" and not pair.get("subject_id"):
        raise ValueError("SACD preference must retain subject_id")


def audit_preference_pairs(
    pairs: Sequence[dict],
    source_splits: dict[str, Iterable[str]] | None = None,
    max_pairs_per_image: int | None = None,
) -> dict:
    split_by_source = None
    if source_splits is not None:
        normalized_splits = {split: {str(source) for source in sources}
                             for split, sources in source_splits.items()}
        assert_source_disjoint(normalized_splits)
        split_by_source = {
            source_id: split
            for split, source_ids in normalized_splits.items()
            for source_id in source_ids
        }
    counts_by_dataset: dict[str, int] = defaultdict(int)
    counts_by_preference_source: dict[str, int] = defaultdict(int)
    pairs_by_image: dict[tuple[str, str], int] = defaultdict(int)
    pairs_by_source: dict[str, int] = defaultdict(int)
    scale_coverage: dict[str, int] = defaultdict(int)
    deltas = []
    duplicate_keys = set()
    duplicate_count = 0
    for pair in pairs:
        validate_preference_pair(pair)
        counts_by_dataset[pair["dataset_source"]] += 1
        counts_by_preference_source[pair["preference_source"]] += 1
        source_id, image_id = str(pair["source_id"]), str(pair["image_id"])
        if split_by_source is not None and split_by_source.get(source_id) != "train":
            raise ValueError(f"preference source {source_id!r} is not in the training manifest")
        pairs_by_image[(source_id, image_id)] += 1
        pairs_by_source[source_id] += 1
        chosen_id, rejected_id = pair["chosen"]["scale_id"], pair["rejected"]["scale_id"]
        scale_coverage[f"{chosen_id}->{rejected_id}"] += 1
        deltas.append(float(pair["score_delta"]))
        key = (source_id, image_id, tuple(pair["target_ratio"]), pair["preference_source"],
               chosen_id, rejected_id)
        if key in duplicate_keys:
            duplicate_count += 1
        duplicate_keys.add(key)
    pair_limit_ok = (max_pairs_per_image is None
                     or all(count <= max_pairs_per_image for count in pairs_by_image.values()))
    ordered = sorted(deltas)
    return {
        "status": "audited",
        "total_pairs": len(pairs),
        "unique_sources": len(pairs_by_source),
        "source_counts_by_dataset": dict(sorted(counts_by_dataset.items())),
        "pair_counts_by_preference_source": dict(sorted(counts_by_preference_source.items())),
        "score_delta_summary": None if not ordered else {
            "min": ordered[0], "median": statistics.median(ordered), "max": ordered[-1]
        },
        "scale_bin_pair_coverage": dict(sorted(scale_coverage.items())),
        "duplicate_pair_count": duplicate_count,
        "geometry_duplicate_rate": duplicate_count / len(pairs) if pairs else 0.0,
        "max_pairs_per_image": max_pairs_per_image,
        "pair_limit_passed": pair_limit_ok,
        "all_pairs_train_split": all(pair["split"] == "train" for pair in pairs),
        "source_split_overlap_checked": source_splits is not None,
        "source_bootstrap_unit": "source",
        "sacd_subject_condition_preserved": all(
            pair.get("subject_id") for pair in pairs if pair["dataset_source"] == "SACD"
        ),
        "mos_semantics_preserved": all(
            pair["score_name"] == "MOS" and pair["preference_source"] == "mos"
            for pair in pairs if pair["dataset_source"] == "GAICD"
        ),
        "official_aic_test_data_used": False,
    }
