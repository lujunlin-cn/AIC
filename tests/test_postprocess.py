import numpy as np
import pytest

from aic.postprocess import (PostprocessConfig, hysteresis_select,
                             normalize_scores, select_temporal, smooth_scores)


def test_rank_normalization_is_tie_stable_and_label_free():
    values = normalize_scores([1., 2., 2., 4.], "rank")
    assert np.allclose(values, [0., .5, .5, 1.])
    assert np.all((values >= 0) & (values <= 1))


def test_smoothing_preserves_length_and_shot_reset():
    values = smooth_scores([0., 0., 1., 0., 0., 1.], "median", 3,
                           shot_boundaries=[3])
    assert len(values) == 6
    # A boundary gives exactly the same result as independently smoothing
    # each half (no cross-shot context enters the first segment).
    assert np.allclose(values, np.r_[smooth_scores([0., 0., 1.], "median", 3),
                                      smooth_scores([0., 0., 1.], "median", 3)])


def test_hysteresis_starts_high_and_continues_low():
    got = hysteresis_select([.1, .8, .5, .5, .1], high=.7, low=.4)
    assert got.tolist() == [False, True, True, True, False]


def test_select_temporal_gap_and_min_duration():
    cfg = PostprocessConfig(threshold=.5, gap=1, min_duration=2)
    got = select_temporal([0., .9, .1, .9, 0., .4, 0.], cfg)
    # First two peaks are joined and survive the minimum duration.
    assert got.tolist() == [False, True, True, True, False, False, False]


def test_invalid_smoothing_window():
    with pytest.raises(ValueError):
        smooth_scores([1., 2.], "median", 2)
