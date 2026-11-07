import pytest

from aic.scale_candidates import build_scale_candidates, score_candidate_set, source_macro_candidate_metrics


def test_candidate_generation_deduplicates_projected_geometry_and_reports_oracle():
    bins = [0.4, 0.6, 0.8, 1.0]
    candidates = build_scale_candidates(
        source_id="src-1", image_id="src-1:0", dataset_source="GNMC", split="val",
        image_width=640, image_height=360, target_ratio=(16, 9), center_xy=(320, 180),
        scale_bins=bins, greedy_scale_id=0, stochastic_scale_ids=[1, 1, 2],
        grid_scale_ids=[0, 1, 2, 3], max_candidates=8,
    )
    assert len(candidates) == 4
    assert sum(row["requested_count"] for row in candidates) == 8
    gt = (128, 72, 384, 216)
    result = score_candidate_set(candidates, gt)
    assert result["greedy_deploy_iou"] < result["oracle_at_k_deploy_iou"]
    assert result["unique_candidate_ratio"] == 0.5
    assert result["position_fixed"]
    macro = source_macro_candidate_metrics([result, dict(result, source_id="src-2", oracle_gain=0.0)])
    assert macro["sources"] == 2
    assert macro["macro_unit"] == "source"


def test_candidate_oracle_rejects_test_split_labels():
    candidates = build_scale_candidates(
        source_id="src-1", image_id="src-1:0", dataset_source="GNMC", split="test",
        image_width=640, image_height=360, target_ratio=(16, 9), center_xy=(320, 180),
        scale_bins=[0.5, 1.0], greedy_scale_id=0, stochastic_scale_ids=[],
        grid_scale_ids=[1], max_candidates=2,
    )
    with pytest.raises(ValueError, match="test split"):
        score_candidate_set(candidates, (160, 90, 320, 180))
