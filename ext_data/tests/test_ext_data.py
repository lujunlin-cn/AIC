"""Unit tests for the external-data helpers (no network, no real datasets).

Run on the remote:  cd /home/supie/AIC/ext_data && python -m pytest -q tests
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicext import ids  # noqa: E402
from aicext.adapters.base import collate_pad  # noqa: E402
from aicext.adapters.temporal import grid_scores, interval_label  # noqa: E402
from aicext.media import frames_at_times, uniform_sample  # noqa: E402
from aicext.schema import annotation_record, media_record, validate  # noqa: E402


def test_collate_pad_all_axes_and_masks():
    a = {"x": np.ones((2, 3), np.float32), "v": np.ones(3, bool), "id": "a"}
    b = {"x": np.ones((4, 1), np.float32), "v": np.ones(1, bool), "id": "b"}
    out = collate_pad([a, b], pad_keys=("x", "v"))
    assert out["x"].shape == (2, 4, 3)
    assert out["v"].dtype == bool and out["v"].shape == (2, 3)
    assert out["v"][1].tolist() == [True, False, False]  # padding is never valid
    assert out["x_shape"].tolist() == [[2, 3], [4, 1]]
    assert out["id"] == ["a", "b"]


def test_ids_and_groups():
    assert ids.youtube_id("dQw4w9WgXcQ") == "yt:dQw4w9WgXcQ"
    with pytest.raises(ValueError):
        ids.youtube_id("short")
    assert ids.qvh_youtube("a_b-c_d_e_f_150.0_300.0") == "yt:a_b-c_d_e_f"
    assert ids.dhf1k_id(7) == "dhf1k:007"
    b = ids.split_bucket("yt:dQw4w9WgXcQ")
    assert 0 <= b < 1 and b == ids.split_bucket("yt:dQw4w9WgXcQ")


def test_interval_label_masks_unlabelled():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    val, valid = interval_label(t, [(0.5, 1.5, 3.0)])
    assert valid.tolist() == [False, True, True, False, False]  # [t0, t1): 1.5 excluded
    assert val[1] == 3.0 and val[0] == 0.0  # zero only where invalid


def test_grid_scores_anchor_zero():
    t = np.array([0.0, 0.5, 1.0, 3.5])
    val, edge = grid_scores(t, np.array([0.0, 1.0, 0.0]), step_s=1.0, anchor_s=0.0)
    assert np.allclose(val[:3], [0.0, 0.5, 1.0])
    assert edge.tolist() == [False, False, False, True]


def test_frames_at_times_first_frame_at_or_after():
    t = np.array([0.0, 0.0333, 0.0667, 0.1])
    assert frames_at_times(t, np.array([0.0, 0.05, 0.1])).tolist() == [0, 2, 3]
    idx, _ = uniform_sample(np.arange(0, 3, 1 / 30), 2.0)
    assert idx[:3].tolist() == [0, 15, 30]


def test_schema_rejects_unknown_fields_and_types():
    with pytest.raises(KeyError):
        media_record(uid="x", bogus=1)
    with pytest.raises(ValueError):
        annotation_record(uid="a", annotation_type="highlight", annotation_source="human")
    m = media_record(uid="u", dataset="d", release="r", item_id="i", source_video_id="s", group_id="g",
                     official_split="train", media_kind="remote_only", media_path=None,
                     download_status="metadata_only", preprocess_status="pending")
    assert validate([m], ("uid", "media_path")) == []


class _Ix:
    def __init__(self, items):
        self.items = items


def _shot_adapter(tmp_path, subset, transitions, kinds, n=20):
    from aicext.adapters.temporal import ShotBoundaryAdapter

    p = tmp_path / f"{subset}.json"
    p.write_text(json.dumps({"frame_num": n, "transitions": transitions, "kinds": kinds, "subset": subset,
                             "negatives_reliable": subset != "only_gradual"}))
    ad = ShotBoundaryAdapter.__new__(ShotBoundaryAdapter)
    ad.items = [({"uid": "u", "group_id": "g", "timeline_path": None}, {"annotation_path": str(p)})]
    ad.window, ad.rng = None, np.random.default_rng(0)
    return ad


def test_shot_boundary_masks(tmp_path):
    ad = _shot_adapter(tmp_path, "train", [[4, 5], [10, 14], [16, 16]], ["cut", "gradual", "degenerate"])
    s = ad[0]
    assert s["label"][4] == 1 and s["label"][5] == 1 and s["label"][12] == 2
    assert s["valid"][0] and not s["valid"][16]  # degenerate row is an ignore region
    og = _shot_adapter(tmp_path, "only_gradual", [[10, 14]], ["gradual"])[0]
    assert og["valid"].sum() == 5 and not og["valid"][0]  # outside labelled gradual: unlabelled


def test_gaicd_row_order_conversion():
    # GAICD rows are y1 x1 y2 x2 MOS; the author loader reads xmin=col1, ymin=col0
    row = np.array([[29, 43, 486, 811, 2.0]], np.float32)
    y1, x1, y2, x2, _ = row.T
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
    assert xywh.tolist() == [[43.0, 29.0, 768.0, 457.0]]
