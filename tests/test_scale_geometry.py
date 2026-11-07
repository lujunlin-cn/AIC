import math
import random

import pytest

from aic.scale_geometry import (
    box_iou_xywh,
    deployed_box_xyxy,
    fit_scale_bins_from_rows,
    maximum_window,
    scale_crop_from_center,
    transpose_xywh,
)


def test_one_thousand_fixed_center_scale_and_transpose_cases():
    rng = random.Random(20261108)
    cases = 1200
    for index in range(cases):
        width = rng.randint(240, 2400)
        height = rng.randint(240, 1800)
        ratio = (9, 16) if index % 2 else (16, 9)
        center = (width / 2, height / 2)
        max_width, _ = maximum_window(width, height, ratio)
        full = scale_crop_from_center(width, height, ratio, center, 1.0)
        assert full[2] == math.floor(max_width + 1e-9)
        assert 0 <= full[0] and full[0] + full[2] <= width
        x, y, crop_w = full[:3]
        assert 0 <= y and y + crop_w * ratio[1] / ratio[0] <= height + 1e-9
        smaller = scale_crop_from_center(width, height, ratio, center, 0.63)
        assert abs((smaller[0] + smaller[2] / 2) - center[0]) <= 1
        assert abs((smaller[1] + smaller[2] * ratio[1] / ratio[0] / 2) - center[1]) <= 1
        box = [x, y, crop_w, crop_w * ratio[1] / ratio[0]]
        transposed, dims = transpose_xywh(box, width, height)
        restored, restored_dims = transpose_xywh(transposed, *dims)
        assert restored_dims == (width, height)
        assert restored == pytest.approx(box)
        submission = (x, y, crop_w)
        exact = deployed_box_xyxy(submission, ratio)
        assert exact[2] <= width + 1e-9 and exact[3] <= height + 1e-9
        assert box_iou_xywh(box, box) == pytest.approx(1.0)


def test_scale_bins_are_fitted_only_from_train_source_distribution():
    rows = [{"split": "train", "scale_ratio": value} for value in (0.35, 0.5, 0.55, 0.7, 0.8, 0.9)]
    bins = fit_scale_bins_from_rows(rows, count=4)
    assert len(bins) == 4
    assert bins == sorted(set(bins))
    assert bins[-1] == 1.0
    with pytest.raises(ValueError, match="train rows"):
        fit_scale_bins_from_rows(rows + [{"split": "val", "scale_ratio": 0.2}], count=4)


@pytest.mark.parametrize("scale", [0, -0.1, 1.01, float("nan")])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        scale_crop_from_center(640, 360, (16, 9), (320, 180), scale)


def test_boundary_crop_is_legal_and_does_not_overrun():
    crop = scale_crop_from_center(1920, 1080, (9, 16), (0, 0), 0.7)
    assert crop[0] == crop[1] == 0
    assert crop[0] + crop[2] <= 1920
    assert crop[1] + crop[2] * 16 / 9 <= 1080 + 1e-9
