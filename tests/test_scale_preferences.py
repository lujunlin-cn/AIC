import pytest

from aic.scale_geometry import scale_crop_from_center
from aic.scale_preferences import (
    assert_source_disjoint,
    audit_preference_pairs,
    fixed_center_sft_target,
    gaicd_scale_preferences,
    geometry_preference_pairs,
    sacd_scale_preferences,
    validate_preference_pair,
)


def _row(gt=(160, 90, 320, 180)):
    return {
        "source_id": "source-a",
        "image_id": "image-a",
        "dataset_source": "LIVE_YT_VC",
        "split": "train",
        "image_width": 640,
        "image_height": 360,
        "target_ratio": [16, 9],
        "center_xy": [320, 180],
        "gt_crop_xywh": list(gt),
    }


def test_fixed_center_sft_selects_scale_without_position_drift():
    bins = [0.5, 0.75, 1.0]
    target = fixed_center_sft_target(_row(), bins, min_deploy_iou=0.9)
    assert target["scale_id"] == 0
    assert target["position_center_xy"] == [320, 180]
    assert target["crop_xywh"] == pytest.approx([160, 90, 320, 180])
    assert target["aic_bbox_xyw"] == [160, 90, 320]
    assert target["label_status"] == "ok"


def test_low_fixed_center_iou_is_diagnostic_not_scale_target():
    target = fixed_center_sft_target(_row(gt=(0, 0, 10, 10)), [0.5, 0.75], min_deploy_iou=0.2)
    assert target["scale_id"] is None
    assert target["label_status"] == "position_mismatch"


def test_geometry_preferences_use_same_train_source_and_deploy_iou():
    pairs = geometry_preference_pairs(_row(), [0.5, 0.75, 1.0], delta_iou=0.1, max_pairs=3)
    assert pairs
    assert all(pair["preference_source"] == "deploy_iou" for pair in pairs)
    assert all(pair["score_name"] == "IoU" for pair in pairs)
    assert all(pair["chosen_score"] > pair["rejected_score"] for pair in pairs)
    assert pairs[0]["chosen"]["aic_bbox_xyw"] == [160, 90, 320]
    validate_preference_pair(pairs[0])
    with pytest.raises(ValueError, match="train-only"):
        geometry_preference_pairs(dict(_row(), split="val"), [0.5, 1.0], 0.1, 1)


def test_source_group_split_fails_on_same_image_cross_split():
    with pytest.raises(ValueError, match="leaks"):
        assert_source_disjoint({"train": ["img-1"], "val": ["img-1"]})


def test_gaicd_only_uses_rated_same_ratio_fixed_center_scale_actions():
    bins = [0.5, 1.0]
    small = scale_crop_from_center(640, 360, (16, 9), (320, 180), 0.5)
    full = scale_crop_from_center(640, 360, (16, 9), (320, 180), 1.0)
    row = {
        "source_id": "gaicd:1", "image_id": "1", "dataset_source": "GAICD",
        "release": "journal", "split": "train", "image_width": 640,
        "image_height": 360, "target_ratio": [16, 9], "center_xy": [320, 180],
        "candidates": [
            {"crop_xywh": list(small), "mos": 4.7, "valid": True},
            {"crop_xywh": list(full), "mos": 2.0, "valid": True},
            {"crop_xywh": list(full), "mos": -1, "valid": False},
            {"crop_xywh": [0, 0, 320, 180], "mos": 5.0, "valid": True},
            {"crop_xywh": [0, 0, 320, 200], "mos": 5.0, "valid": True},
        ],
    }
    pairs = gaicd_scale_preferences(row, bins, min_mos_delta=0.5, max_pairs=2)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["scale_id"] == 0
    assert pairs[0]["score_name"] == "MOS"
    assert pairs[0]["chosen_score"] == 4.7
    validate_preference_pair(pairs[0])
    audit = audit_preference_pairs(pairs, {"train": ["gaicd:1"], "val": ["other"]}, 2)
    assert audit["source_counts_by_dataset"] == {"GAICD": 1}
    assert audit["mos_semantics_preserved"]
    assert audit["source_bootstrap_unit"] == "source"
    with pytest.raises(ValueError, match="training manifest"):
        audit_preference_pairs(pairs, {"train": ["other"], "val": ["gaicd:1"]}, 2)
    with pytest.raises(ValueError, match="journal"):
        gaicd_scale_preferences(dict(row, release="conference"), bins, 0.5, 2)


def test_sacd_does_not_infer_pairs_from_best_box_or_candidate_arrays():
    row = {
        "source_id": "sacd:1", "image_id": "sacd:1", "split": "train",
        "subject_id": "s0", "subject_count": 1, "best_box": [[10, 10, 300, 200]],
        "target_ratio": [16, 9], "image_width": 640, "image_height": 360,
        "center_xy": [320, 180],
    }
    assert sacd_scale_preferences(row, [0.5, 1.0], max_pairs=10) == []


def test_sacd_explicit_pair_requires_same_subject_and_scale_geometry():
    bins = [0.5, 1.0]
    chosen = scale_crop_from_center(640, 360, (16, 9), (320, 180), 0.5)
    rejected = scale_crop_from_center(640, 360, (16, 9), (320, 180), 1.0)
    row = {
        "source_id": "sacd:1", "image_id": "sacd:1", "split": "train",
        "subject_id": "s0", "subject_count": 1, "target_ratio": [16, 9],
        "image_width": 640, "image_height": 360, "center_xy": [320, 180],
        "ranked_pairs": [{"image_id": "sacd:1", "subject_id": "s0",
                          "chosen_box_xywh": list(chosen), "rejected_box_xywh": list(rejected)}],
    }
    pairs = sacd_scale_preferences(row, bins, max_pairs=2)
    assert len(pairs) == 1
    assert pairs[0]["dataset_source"] == "SACD"
    assert pairs[0]["preference_source"] == "ranked_pair"
    with pytest.raises(ValueError, match="same image and subject"):
        sacd_scale_preferences(dict(row, ranked_pairs=[dict(row["ranked_pairs"][0], subject_id="s1")]), bins, 2)
