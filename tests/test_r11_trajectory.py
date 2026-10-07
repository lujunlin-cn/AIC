"""Synthetic tests for the R11 S1 score-field Viterbi audit (no data needed)."""
import numpy as np
import pytest

from aic.r11_trajectory import (cluster_ci, decode_pool, frame_positions,
                                paired_per_source, segments, viterbi_l1)


def test_positions_match_geometry():
    # '1-3': w = 360/3 = 120 full height -> free axis x, span 520, pos in [0,1].
    p = frame_positions("1-3")
    assert len(p) == 129 and p[0] == 0.0 and abs(p[-1] - 1.0) < 1e-9
    # '3-1': w = 640, h = 640/3 > 360? no: h = 640*(1/3) = 213.33 -> full width,
    # free axis y, span = 360 - 213.33 = 146.67 -> still normalised to [0,1].
    assert abs(frame_positions("3-1")[-1] - 1.0) < 1e-9
    # '16-9' on a 640x360 source is the full frame -> degenerate zero positions.
    assert np.all(frame_positions("16-9") == 0.0)


def test_lam_zero_equals_argmax_and_full_lam_is_monotone_path():
    rng = np.random.default_rng(0)
    T, NC = 40, 129
    pos = np.linspace(0, 1, NC)
    scores = rng.normal(0, 0.1, (T, NC))
    scores[:, ::24] += 0.5  # many argmax jumps
    assert (viterbi_l1(scores, pos, 0.0) == scores.argmax(1)).all()
    # large lam must select a piecewise-constant path (<= a few switches)
    path = viterbi_l1(scores, pos, 50.0)
    assert len(set(path.tolist())) <= 3


def test_viterbi_beats_argmax_on_smooth_ground_truth():
    # GT: box drifts slowly; scores = true utility + per-frame noise on the
    # argmax peak only (temporal-decoupled noise).  A smooth decoder wins.
    rng = np.random.default_rng(1)
    T, NC = 120, 129
    pos = np.linspace(0, 1, NC)
    centre = 0.3 + 0.4 * np.sin(np.linspace(0, np.pi, T))
    u = 0.6 - np.abs(pos[None, :] - centre[:, None])
    noisy = u + 0.10 * (rng.random((T, NC)) < 0.15) * rng.normal(0, 1, (T, NC))
    base = noisy.argmax(1)
    dp = viterbi_l1(noisy, pos, 1.5)
    assert u[np.arange(T), dp].mean() > u[np.arange(T), base].mean()


def test_segments_split_on_ratio_and_order():
    vid = np.array(["a", "a", "a", "b", "b"])
    ratio = np.array(["1-3", "3-1", "1-3", "1-3", "1-3"])
    order = np.array([0, 1, 2, 0, 1])
    segs = segments(vid, ratio, order)
    assert len(segs) == 4  # a:1-3 | a:3-1 | a:1-3 | b:1-3
    assert list(segs[1]) == [1]


def test_paired_delta_and_permutation_control():
    # Build 8 sources where DP beats argmax ONLY through frame order signal.
    rng = np.random.default_rng(2)
    T, NC = 60, 129
    pos = np.linspace(0, 1, NC)
    rows_pred, rows_u, vids, ratios, ords = [], [], [], [], []
    for s in range(8):
        centre = 0.2 + 0.6 * s / 7 + 0.2 * np.sin(np.linspace(0, np.pi, T))
        u = 0.6 - np.abs(pos[None, :] - centre[:, None])
        pred = u + 0.12 * (rng.random((T, NC)) < 0.2) * rng.normal(0, 1, (T, NC))
        rows_pred.append(pred)
        rows_u.append(u)
        vids += [f"v{s}"] * T
        ratios += ["1-3"] * T
        ords += list(range(T))
    pred = np.concatenate(rows_pred)
    u = np.concatenate(rows_u)
    vid = np.array(vids)
    ratio = np.array(ratios)
    order = np.array(ords)
    base, dp = decode_pool(pred, u, vid, ratio, order, lam=1.5)
    per = paired_per_source(u, vid, ratio, order, base, dp)
    mean, ci = cluster_ci(per)
    assert mean > 0 and ci[0] > 0  # clean temporal signal -> CI excludes 0
    # permutation control destroys the order signal
    _, dp_p = decode_pool(pred, u, vid, ratio, order, lam=1.5, permute_order=True)
    per_p = paired_per_source(u, vid, ratio, order, base, dp_p)
    mean_p, _ = cluster_ci(per_p)
    assert mean_p < mean  # shuffled order must not keep the full gain


def test_cluster_ci_shapes():
    d = {f"v{i}": 0.01 for i in range(5)}
    d.update({f"w{i}": -0.01 for i in range(5)})
    mean, ci = cluster_ci(d, n_boot=2000)
    assert abs(mean) < 1e-12 and ci[0] <= 0 <= ci[1]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
