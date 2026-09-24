import numpy as np

from aic.spatial import gradient_saliency_center, legal_widths, place_crop, saliency_crop, smooth_centers


def test_legal_crop_candidates_and_clamping():
    widths = legal_widths(1280, 720, [9, 16])
    assert widths[0] == 405.0 and widths == sorted(widths, reverse=True)
    crop = place_crop(1280, 720, [9, 16], 0, 0, widths[-1])
    assert crop[0] == 0 and crop[1] == 0 and crop[2] == widths[-1]
    assert crop[1] + crop[2] * 16 / 9 <= 720 + 1e-6


def test_gradient_center_and_ema_are_finite():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, 48:] = 255
    cx, cy, confidence = gradient_saliency_center(image)
    assert .7 < cx < 1 and 0 <= cy <= 1 and confidence >= 0
    smoothed = smooth_centers([[.1, .5], [.9, .5], [.9, .5]], alpha=.5, reset=[False, False, True])
    assert np.allclose(smoothed[0], [.1, .5])
    assert np.allclose(smoothed[2], [.9, .5])
    assert np.all(np.isfinite(saliency_crop(image, [16, 9])))
