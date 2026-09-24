import torch

from aic.train import (TVSUM_LABEL_PROTOCOL, TVSUM_TARGET_THRESHOLD,
                       aggregate_proxy_metrics, frame_f1, video_proxy_metrics)


def test_prediction_threshold_does_not_change_fixed_target():
    # Two positive targets under the versioned TVSum proxy; changing only the
    # prediction threshold changes predictions, never the ground-truth count.
    labels = torch.tensor([0.49, 0.50, 1.0, 0.0])
    logits = torch.tensor([-0.2, 0.0, 0.2, 1.0])
    mask = torch.ones(4, dtype=torch.bool)
    low = frame_f1(logits, labels, mask, threshold=0.4)
    high = frame_f1(logits, labels, mask, threshold=0.8)
    assert TVSUM_LABEL_PROTOCOL.endswith("ge_0.5_v1")
    assert TVSUM_TARGET_THRESHOLD == 0.5
    # Both calls have the same fixed FN/TP truth support; only prediction
    # selection changes.  In particular, the 0.49 target remains negative.
    assert low[1] + low[3] == high[1] + high[3] == 2


def test_video_macro_metrics_and_diagnostics():
    # Video A has one positive; video B is long and negative. Macro and micro
    # differ, making accidental micro-only model selection visible.
    labels_a = torch.tensor([1.0, 0.0])
    logits_a = torch.tensor([2.0, -2.0])
    labels_b = torch.zeros(10)
    logits_b = torch.full((10,), -2.0)
    mask_a = torch.ones(2, dtype=torch.bool)
    mask_b = torch.ones(10, dtype=torch.bool)
    rows = [video_proxy_metrics(logits_a, labels_a, mask_a, .5, "a"),
            video_proxy_metrics(logits_b, labels_b, mask_b, .5, "b")]
    out = aggregate_proxy_metrics(rows, tp=1, fp=0, fn=0, valid_frames=12)
    assert out["video_macro_f1"] == 1.0
    assert out["micro_f1_diagnostic"] == 1.0
    assert out["empty_prediction_rate"] == 0.5
    assert rows[1]["score_quantiles"]["q50"] < 0.2

