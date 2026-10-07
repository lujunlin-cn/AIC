"""R11 S1: temporal-constrained decoding over the frozen B3 score field.

B3 picks, per frame, argmax over the 129-grid candidate utilities -- an
independent decision per frame.  S1 retests the P1A idea (L1-smooth Viterbi,
REJECTED on B0 observations, EXPERIMENTS.md L262) on a different input: the
B3 utility score field.  Pure offline analysis -- consumes a dumped score
table, no model, no media, no GT beyond the cached per-candidate IoU `u`.

Score-table contract (npz, allow_pickle=False):
    pred  (N, NC) float32   head scores per candidate
    u     (N, NC) float32   per-candidate IoU vs GT (evaluation only)
    vid   (N,)    <U str    source id
    ratio (N,)    <U str    '1-3' | '3-1' | '9-16' | '16-9'
    ord   (N,)    int       frame order within source (dump side: argsort of frame)
    pool  (N,)    <U str    pool tag, e.g. 'live_dev' / 'live_confirmation'

Decoding: per (vid, ratio) segment, Viterbi with
    cost_t(j) = -pred_t(j) + lam * |pos_t(j) - pos_{t-1}(i*)|,
pos = offset/span in [0, 1] (the same normalisation the head sees as the
`pos` feature).  lam = 0 reproduces per-frame argmax exactly (asserted in
tests).  Ratio changes inside a source split the sequence (geometry change).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .contract import parse_ratio

NC_DEFAULT = 129

# Same tag -> (rw, rh) map as scripts/v8_s_train_multidata.py.
RATIOS = {"1-3": (1, 3), "3-1": (3, 1), "9-16": (9, 16), "16-9": (16, 9)}


def frame_positions(ratio, n_cand=NC_DEFAULT, W=640.0, H=360.0):
    """Normalised free-axis offsets for one frame's candidate grid.

    Mirrors scripts/v8_s_train_multidata.py: offs = linspace(0, span, NC),
    pos = offs / span (zeros when the window covers the full frame).
    """
    rw, rh = RATIOS[ratio] if isinstance(ratio, str) else parse_ratio(ratio)
    w = min(W, H * rw / rh)
    h = w * rh / rw
    axis = 0 if w < W - 1e-9 else (1 if h < H - 1e-9 else None)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    if span <= 0:
        return np.zeros(n_cand, dtype=np.float64)
    return (np.linspace(0.0, span, n_cand) / span).astype(np.float64)


def viterbi_l1(scores, pos, lam):
    """Decode one segment.  scores (T, NC), pos (NC,) normalised offsets.

    Returns chosen candidate indices (T,).  O(T * NC^2), vectorised per step.
    """
    T, NC = scores.shape
    if T == 1 or lam == 0.0:
        return scores.argmax(1).copy()
    move = np.abs(pos[:, None] - pos[None, :])  # move[i, j] = |pos_i - pos_j|
    cost = -scores[0].astype(np.float64)
    back = np.zeros((T, NC), dtype=np.int32)
    for t in range(1, T):
        trans = cost[:, None] + lam * move  # (i, j): best from i, landing at j
        back[t] = trans.argmin(0)
        cost = trans.min(0) - scores[t]
    out = np.zeros(T, dtype=np.int64)
    out[-1] = int(cost.argmin())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def segments(vid, ratio, order):
    """Yield (row_indices) for each contiguous (vid, ratio) block, ordered."""
    vid = np.asarray(vid)
    ratio = np.asarray(ratio)
    order = np.asarray(order)
    out = []
    for v in sorted(set(vid.tolist())):
        m = vid == v
        idx = np.flatnonzero(m)
        idx = idx[np.argsort(order[idx], kind="stable")]
        start = 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or ratio[idx[k]] != ratio[idx[start]]:
                out.append(idx[start:k])
                start = k
    return out


def segment_iou(u_seg, idx_seg):
    return float(u_seg[np.arange(len(idx_seg)), idx_seg].mean())


def paired_per_source(u, vid, ratio, order, base_idx, dp_idx):
    """Per-source mean-IoU delta (dp - base), the clustering unit for CIs."""
    per = {}
    for seg in segments(vid, ratio, order):
        v = str(vid[seg[0]])
        b = segment_iou(u[seg], base_idx[seg])
        d = segment_iou(u[seg], dp_idx[seg])
        per.setdefault(v, []).append(d - b)
    return {v: float(np.mean(ds)) for v, ds in per.items()}


def cluster_ci(deltas, n_boot=10000, seed=0):
    """Source-level bootstrap CI95 of the mean paired delta."""
    d = np.array(sorted(deltas.values()), dtype=np.float64)
    if len(d) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), (float(np.percentile(means, 2.5)),
                             float(np.percentile(means, 97.5)))


def decode_pool(pred, u, vid, ratio, order, lam, pos=None, permute_order=False):
    """Decode every segment; optionally destroy temporal structure first
    (content-permutation control: shuffle ord WITHIN source).  `pos` is the
    optional (N, NC) per-row normalised offset matrix from the dump (correct
    for any W/H); falls back to the 640x360 reconstruction per ratio.
    Returns (argmax_idx, dp_idx) over all rows."""
    base = np.zeros(len(pred), dtype=np.int64)
    dp = np.zeros(len(pred), dtype=np.int64)
    ord_eff = order
    if permute_order:
        rng = np.random.default_rng(20261007)
        ord_eff = np.array(order, dtype=np.int64).copy()
        for v in sorted(set(np.asarray(vid).tolist())):
            m = np.asarray(vid) == v
            idx = np.flatnonzero(m)
            ord_eff[idx] = rng.permutation(ord_eff[idx])
    for seg in segments(vid, ratio, ord_eff):
        p = pos[seg[0]] if pos is not None else frame_positions(str(ratio[seg[0]]), pred.shape[1])
        base[seg] = pred[seg].argmax(1)
        dp[seg] = viterbi_l1(pred[seg], np.asarray(p, dtype=np.float64), lam)
    return base, dp


def sweep(pred, u, vid, ratio, order, lams, pos=None, permute=False):
    """Per-lam per-source paired deltas (and the lam=0 identity check)."""
    out = {}
    for lam in lams:
        base, dp = decode_pool(pred, u, vid, ratio, order, lam, pos=pos,
                               permute_order=permute)
        per = paired_per_source(u, vid, ratio, order, base, dp)
        mean, ci = cluster_ci(per)
        out[lam] = {"mean": mean, "ci95": ci, "n_src": len(per)}
    return out


def load_table(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", type=Path, required=True, help="score-table npz (see contract)")
    ap.add_argument("--dev-pool", required=True, help="pool tag used for lam selection")
    ap.add_argument("--eval-pools", nargs="+", required=True,
                    help="locked-out pools; the selected lam is applied ONCE")
    ap.add_argument("--lams", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tab = load_table(args.table)
    pred, u = tab["pred"], tab["u"]
    vid, ratio, order, pool = tab["vid"], tab["ratio"], tab["ord"], tab["pool"]
    pos = tab.get("pos")  # (N, NC) dump-side matrix; None -> 640x360 fallback
    is_dev = pool == args.dev_pool
    res = {"protocol": "R11 S1 score-field Viterbi; lam tuned on dev, applied once to eval",
           "dev_pool": args.dev_pool, "eval_pools": args.eval_pools,
           "lam_grid": args.lams, "n_rows": int(len(pred)),
           "pos_source": "table" if pos is not None else "ratio-640x360-fallback"}

    kw = dict(pred=pred[is_dev], u=u[is_dev], vid=vid[is_dev],
              ratio=ratio[is_dev], order=order[is_dev],
              pos=(pos[is_dev] if pos is not None else None))
    res["dev_sweep"] = {str(k): v for k, v in sweep(**kw, lams=args.lams).items()}
    res["dev_permute_control"] = {str(k): v for k, v in sweep(
        **kw, lams=args.lams, permute=True).items()}
    best_lam = max(args.lams, key=lambda k: res["dev_sweep"][str(k)]["mean"])
    res["selected_lam"] = float(best_lam)

    res["eval"] = {}
    for tag in args.eval_pools:
        m = pool == tag
        base, dp = decode_pool(pred[m], u[m], vid[m], ratio[m], order[m], best_lam,
                               pos=(pos[m] if pos is not None else None))
        per = paired_per_source(u[m], vid[m], ratio[m], order[m], base, dp)
        mean, ci = cluster_ci(per)
        base_only = paired_per_source(u[m], vid[m], ratio[m], order[m], base, base)
        bmean, _ = cluster_ci(base_only)
        res["eval"][tag] = {"lam": float(best_lam), "mean_delta": mean, "ci95": ci,
                            "n_src": len(per), "base_mean_iou_delta_identity": bmean}
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["eval"], indent=1))


if __name__ == "__main__":
    main()
