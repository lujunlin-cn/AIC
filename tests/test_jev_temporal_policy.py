import numpy as np
import pytest

from aic.jev_temporal_policy import (
    expand_events_to_frames,
    half_open_indices,
    merge_events,
    phase_candidates,
    choose_phase,
    selected_frame_ids,
    validate_pts,
)


def test_pts_contract_and_half_open_mapping_use_original_axis():
    pts = [0.0, 0.04, 0.09, 0.20, 0.31]
    contract = validate_pts(pts)
    assert contract.pts.tolist() == pytest.approx(pts)
    assert half_open_indices(pts, 0.04, 0.20).tolist() == [1, 2]
    assert expand_events_to_frames(pts, [[0.04, 0.20], [0.20, 0.31]]).tolist() == [1, 2, 3]


def test_non_monotone_pts_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_pts([0.0, 0.2, 0.1])


def test_phase_candidates_keep_equal_budget_and_boundaries():
    pts = np.arange(20, dtype=float) * 0.5
    candidates = phase_candidates(pts, keep_start=6, budget=5, shift_seconds=2.0)
    assert candidates.budget == 5
    assert all(len(candidates.frame_ids[name]) == 5 for name in ("left", "keep", "right"))
    assert candidates.frame_ids["left"].tolist() == [2, 3, 4, 5, 6]
    assert candidates.frame_ids["keep"].tolist() == [6, 7, 8, 9, 10]
    assert candidates.frame_ids["right"].tolist() == [10, 11, 12, 13, 14]
    assert selected_frame_ids(candidates, "keep").tolist() == [6, 7, 8, 9, 10]


def test_merge_events_and_conservative_choice_fallback():
    assert merge_events([[0, 1], [0.9, 2], [3, 4]], 5) == [(0.0, 2.0), (3.0, 4.0)]
    selected, audit = choose_phase("left", {"left": .43, "keep": .40, "right": .17}, confidence_margin=.15)
    assert selected == "keep"
    assert audit["accepted"] is False
    selected, audit = choose_phase("left", {"left": .70, "keep": .20, "right": .10}, confidence_margin=.15)
    assert selected == "left"
    assert audit["accepted"] is True
    selected, audit = choose_phase("right", {"left": .05, "keep": .10, "right": .85}, order_stable=False)
    assert selected == "keep"
    assert audit["fallback"] is True
