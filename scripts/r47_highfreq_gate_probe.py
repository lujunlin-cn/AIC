#!/usr/bin/env python3
"""Source-held-out probe for a fixed event-boundary high-frequency gate.

The D3 residual proved that sub-second state can help, but its full ranking
also perturbs quiet frames.  This experiment tests one predeclared mechanism:
only a frame in the top quartile of within-video transition strength, with a
state change that persists across the local window, may receive the D3
residual.  Quiet frames keep the R26 center score.  The quartile and eta are
fixed before reading the proxy results; there is no parameter sweep.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import sys
import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import GRADE, _rank01, proxy_f, select


def read_rows(paths):
    out = []
    for p in paths:
        with Path(p).open(newline="") as f:
            out.extend(dict(x) for x in csv.DictReader(f))
    return out


def strength_gate(h: np.ndarray) -> np.ndarray:
    """Fixed, label-free boundary condition on the D3 feature contract.

    Feature 15 is the before/after mean change, 16 the before/after peak
    change, and 17/18 are local state displacement before/after.  A persistent
    transition has a large aggregate change and post-window displacement at
    least as large as the pre-window displacement.  The top quartile is a
    within-video normalization, so no absolute video-domain threshold is fit.
    """
    asym = np.abs(h[:, 15]) + np.abs(h[:, 16])
    persistence = h[:, 18] - h[:, 17]
    strength = asym + 0.5 * (h[:, 17] + h[:, 18])
    q = float(np.quantile(strength, 0.75)) if len(strength) else 0.0
    return (strength >= q) & (persistence >= 0.0), strength, q


def fit_oof(feats, labels):
    by = {}
    for r in labels:
        by.setdefault(r["video_id"], []).append(r)
    pred = {}
    gate = {}
    for vid in sorted(by):
        train_rows = [r for r in labels if r["video_id"] != vid]
        X, Y, C = [], [], []
        for r in train_rows:
            v = r["video_id"]
            t = min(int(float(r["time_sec"])), len(feats[v]) // 4 - 1)
            j = min(round(t * 4), len(feats[v]) - 1)
            X.append(feats[v][j]); Y.append(GRADE[r["generic_highlight"]])
            C.append(1.0 - abs(t - 14.5) / 14.5)
        X, Y, C = np.asarray(X, np.float32), np.asarray(Y, float), np.asarray(C, float)
        sc = StandardScaler().fit(X)
        A = np.c_[np.ones(len(C)), C]
        coef = np.linalg.lstsq(A, Y, rcond=None)[0]
        m = Ridge(alpha=10.0).fit(sc.transform(X), Y - A @ coef)
        pred[vid] = m.predict(sc.transform(feats[vid]))
        gate[vid] = strength_gate(feats[vid])
    return by, pred, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eta", type=float, default=0.20)
    args = ap.parse_args()
    labels = read_rows(args.labels)
    vids = sorted({r["video_id"] for r in labels})
    feats = {}
    for v in vids:
        frames, _, _ = decode_highfreq(args.videos / f"{v}.mp4")
        feats[v] = highfreq_features(frames)
    by, pred, gate = fit_oof(feats, labels)
    source = []
    for v in vids:
        rs = by[v]
        times, yi = np.unique(np.asarray([int(float(r["time_sec"])) for r in rs]), return_index=True)
        # A source is represented once per second, matching the fixed proxy.
        y = np.asarray([GRADE[rs[i]["generic_highlight"]] for i in yi], float)
        h = feats[v]; p4 = pred[v]; g4, strength, q = gate[v]
        ix = np.minimum(np.rint(times * 4).astype(int), len(h) - 1)
        p = p4[ix]; g = g4[ix]
        n = len(y); k = int(round(.85 * n))
        c = 1.0 - np.abs(np.arange(n) - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
        rc, rr = _rank01(c), _rank01(p)
        # Full D3 arm is retained as the direct control.  The proposed gate
        # applies the signed residual only to high-confidence transitions.
        full = rc + args.eta * (rr - 0.5)
        gated = rc + args.eta * g.astype(float) * (rr - 0.5)
        kc, kf, kg = select(rc, k), select(full, k), select(gated, k)
        fc, ff, fg = proxy_f(kc, y, True), proxy_f(kf, y, True), proxy_f(kg, y, True)
        bc, bf, bg = proxy_f(kc, y, False), proxy_f(kf, y, False), proxy_f(kg, y, False)
        source.append({"video_id": v, "n": n, "k": k, "q75_strength": q,
                       "gated_label_count": int(g.sum()),
                       "center_soft": fc, "full_d3_soft": ff, "gate_soft": fg,
                       "delta_full_soft": ff - fc, "delta_gate_soft": fg - fc,
                       "center_binary": bc, "full_d3_binary": bf, "gate_binary": bg,
                       "delta_full_binary": bf - bc, "delta_gate_binary": bg - bc,
                       "full_swaps": int(len(set(kc.tolist()) ^ set(kf.tolist())) // 2),
                       "gate_swaps": int(len(set(kc.tolist()) ^ set(kg.tolist())) // 2),
                       "gate_times": times[g].tolist()})
    out = {"experiment_id": "R47_HIGHFREQ_PERSISTENT_BOUNDARY_GATE",
           "status": "PUBLIC_SOURCE_HELD_OUT_COMPLETED", "rows": len(labels),
           "sources": len(vids), "eta": args.eta, "ridge_alpha": 10.0,
           "gate": "within-video top quartile of abs(before-after mean)+abs(before-after peak)+0.5*state displacement; after displacement >= before; positive residual only",
           "official_gt": False, "source": source,
           "macro_center_soft": float(np.mean([z["center_soft"] for z in source])),
           "macro_full_d3_soft": float(np.mean([z["full_d3_soft"] for z in source])),
           "macro_gate_soft": float(np.mean([z["gate_soft"] for z in source])),
           "macro_center_binary": float(np.mean([z["center_binary"] for z in source])),
           "macro_full_d3_binary": float(np.mean([z["full_d3_binary"] for z in source])),
           "macro_gate_binary": float(np.mean([z["gate_binary"] for z in source])),
           "positive_full_soft": int(sum(z["delta_full_soft"] > 0 for z in source)),
           "positive_gate_soft": int(sum(z["delta_gate_soft"] > 0 for z in source)),
           "positive_full_binary": int(sum(z["delta_full_binary"] > 0 for z in source)),
           "positive_gate_binary": int(sum(z["delta_gate_binary"] > 0 for z in source))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("experiment_id", "sources", "macro_center_soft", "macro_full_d3_soft", "macro_gate_soft", "macro_center_binary", "macro_full_d3_binary", "macro_gate_binary", "positive_full_soft", "positive_gate_soft")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
