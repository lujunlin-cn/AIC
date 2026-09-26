import numpy as np
import torch

from scripts.probe_videomae_qvh import clip_spans, metrics, normalize_video


def test_video_normalization_is_channelwise_with_time_axis():
    x = torch.zeros((2, 3, 16, 2, 2))
    x[:, 0] = 255
    x[:, 1] = 127.5
    result = normalize_video(x, [0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    assert result.shape == x.shape
    torch.testing.assert_close(result[0, :, 12, 0, 0], torch.tensor([1.8, 0.6, -0.6]))


def test_spans_cover_tail_without_repeated_centers():
    spans = clip_spans(300, 16, 4)
    assert spans[0] == (0, 16)
    assert spans[-1] == (284, 300)
    assert all(stop - start == 16 for start, stop in spans)
    centers = [s + (e - s) // 2 for s, e in spans]
    assert len(centers) == len(set(centers))
    assert clip_spans(7, 16, 4) == [(0, 7)]


def test_spearman_uses_average_ranks_for_ties():
    from scipy.stats import spearmanr
    p = np.asarray([.2, .1, .4, .3, .5])
    target = np.asarray([0., 0., 0., 1., 1.])
    row = metrics(p, target, "synthetic")
    assert abs(row["spearman"] - spearmanr(p, target).statistic) < 1e-12
