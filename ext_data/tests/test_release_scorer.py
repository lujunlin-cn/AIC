"""Window scorer, holdout rules and release immutability (no real datasets)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aicext import window_scorer as ws  # noqa: E402
from aicext.holdout import project_split, reserve_bucket  # noqa: E402


def upstream_iou(a, b):  # RetargetVid retargetvid_eval.bb_intersection_over_union, verbatim logic
    xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return inter / float((a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter)


def test_max_window_geometry():
    assert ws.max_window(640, 360, "1-3")[:2] == (120.0, 360.0) and ws.max_window(640, 360, "1-3")[2] == 0
    w, h, ax = ws.max_window(640, 360, [3, 1])
    assert (w, ax) == (640.0, 1) and abs(h - 213.333) < 1e-3
    w, h, ax = ws.max_window(1920, 1080, [9, 16])
    assert (w, h, ax) == (607.5, 1080.0, 0)
    assert ws.max_window(1280, 720, [16, 9])[2] is None


def test_candidates_legal_and_grid():
    c = ws.legal_candidates(640, 360, "1-3", n=65)
    assert len(c["offsets"]) == 65 and c["offsets"][0] == 0 and c["offsets"][-1] == 520
    chk = ws.check_candidates(640, 360, "1-3", c["boxes_xywh"])
    assert chk["legal"].all() and chk["is_max_window"].all()
    bad = np.array([[600, 0, 120, 360], [0, 0, 100, 360], [0, 0, 60, 180]], float)
    chk = ws.check_candidates(640, 360, "1-3", bad)
    assert chk["legal"].tolist() == [False, False, True] and chk["is_max_window"].tolist() == [False, False, False]


def test_iou_halfopen_and_inclusive_match_upstream():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = np.array([rng.integers(0, 500), rng.integers(0, 200), 0, 0], float)
        a[2:] = a[:2] + rng.integers(1, 140, 2)
        b = np.array([rng.integers(0, 500), rng.integers(0, 200), 0, 0], float)
        b[2:] = b[:2] + rng.integers(1, 140, 2)
        ax = np.r_[a[:2], a[2:] - a[:2]][None]
        bx = np.r_[b[:2], b[2:] - b[:2]][None]
        inc, _ = ws.iou_matrix(ax, bx, "inclusive_plus1")
        assert abs(inc[0, 0] - upstream_iou(a, b)) < 1e-12
        ho, _ = ws.iou_matrix(ax, bx, "halfopen")
        iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
        assert abs(ho[0, 0] - iw * ih / u) < 1e-12


def test_score_candidates_masks_and_aggregates():
    gt = np.array([[100, 0, 120, 360], [110, 0, 120, 360], [np.nan, np.nan, np.nan, np.nan]], float)
    cand = np.array([[100, 0, 120, 360], [600, 0, 120, 360]], float)
    out = ws.score_candidates(640, 360, "1-3", cand, gt, gt_source="human", gt_is_max_window=True, with_ceiling=True)
    assert out["valid"].tolist() == [[True, True, False], [False, False, False]]  # 2nd candidate illegal
    assert out["iou"][0, 0] == pytest.approx(1.0) and np.isnan(out["iou"][0, 2])
    assert out["iou_mean"][0] == pytest.approx((1 + 110 / 130) / 2, rel=1e-6)
    assert np.isnan(out["iou_mean"][1]) and out["n_valid"].tolist() == [2, 0]
    assert out["iou_ceiling"][0] == pytest.approx(1.0) and np.isnan(out["iou_ceiling"][2])
    with pytest.raises(ValueError):
        ws.score_candidates(640, 360, "1-3", cand, gt, gt_source="mixed", gt_is_max_window=True)


def test_variable_size_gt_ceiling_below_one():
    gt = np.array([[800, 300, 270, 480]], float)  # LIVE-like small 9:16 box
    out = ws.score_candidates(1920, 1080, [9, 16], ws.legal_candidates(1920, 1080, [9, 16])["boxes_xywh"], gt,
                              gt_source="human", gt_is_max_window=False, with_ceiling=True)
    assert out["iou_ceiling"][0] == pytest.approx(270 * 480 / (607.5 * 1080), rel=1e-6)
    assert not out["gt_is_max_window"]


def test_gt_boundary_never_mutates_input():
    gt = np.array([[1313, 0, 608, 1080]], float)  # right = 1921 as published
    keep = gt.copy()
    c1 = ws.apply_gt_boundary(gt, 1920, 1080, "clip_v1")
    c2 = ws.apply_gt_boundary(gt, 1920, 1080, "clip_exp_v1")
    assert (gt == keep).all()
    assert c1[0, 0] + c1[0, 2] == 1920 and c2[0, 0] + c2[0, 2] == 1919 and c2[0, 1] + c2[0, 3] == 1079


def test_project_split_rules():
    e = lambda role: {"experiment": "X", "split_name": "s", "role": role}  # noqa: E731
    assert project_split("train", "r", [e("fit")])[0] == "train"
    assert project_split("train", "r", [e("fit"), e("selection")])[0] == "holdout_dev_exposed"
    assert project_split("val", "r", [e("confirmation"), e("selection")])[0] == "holdout_confirm_exposed"
    assert project_split("train", "r", [e("in_progress_unregistered")])[0] == "holdout_dev_exposed"
    assert project_split("quarantine", "r", [e("confirmation")])[0] == "quarantine"
    assert project_split("train", "r", [], {"reason": "reserve"})[0] == "holdout_confirm_reserved"
    b = reserve_bucket("liveytvc:video1")
    assert 0 <= b < 1 and b == reserve_bucket("liveytvc:video1")


def test_release_writer_immutable(tmp_path, monkeypatch):
    from aicext import release as R

    monkeypatch.setattr(R, "CODE_PARTS", ("aicext",))
    w = R.ReleaseWriter("toy_v1", root=tmp_path)
    w.jsonl("manifests/train.jsonl", [{"a": 1}])
    w.npz("packed/x.npz", v=np.arange(3))
    w.snapshot_code()
    out = w.finalize({"kind": "toy"})
    rel = R.Release(path=out, verify="all")
    assert rel.manifest("train") == [{"a": 1}] and rel.meta["immutable"]
    assert R.make_readonly_check(out)
    with pytest.raises(FileExistsError):
        R.ReleaseWriter("toy_v1", root=tmp_path)
    with pytest.raises(PermissionError):
        (out / "manifests" / "train.jsonl").open("w")
    idx = [json.loads(l) for l in (tmp_path / "INDEX.jsonl").read_text().splitlines()]
    assert idx[0]["release"] == "toy_v1"
    for p in sorted(out.rglob("*"), key=lambda q: -len(q.parts)):  # let pytest clean up
        os.chmod(p, 0o755)
    os.chmod(out, 0o755)
