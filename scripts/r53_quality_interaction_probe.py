#!/usr/bin/env python3
"""R53: conditional local-state residual diagnostics.

This probe asks a different question from the stopped high-frequency and
whole-video experiments: does the useful local image residual depend on an
observable frame-quality/state regime?  It compares the existing local-state
readout with fixed quality/state interactions under source-held-out fitting.

The human event fields are included only as a mechanism diagnostic.  They are
not available at official inference time and are therefore never used to make
an upload package.  All decisions use the fixed center budget, eta=.20 and
Ridge alpha=10 inherited from the audited R44 protocol; there is no sweep.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

GRADE = {"0普通": 0.0, "1有信息": 1.0, "2值得保留": 2.0, "3高光核心": 3.0}


def read_rows(paths):
    out = []
    for stream, path in enumerate(paths):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                q = dict(r); q["_stream"] = stream; out.append(q)
    return out


def descriptor(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    return np.asarray([
        gray.mean() / 255., gray.std() / 255.,
        cv2.Laplacian(gray, cv2.CV_32F).var() / 10000.,
        cv2.Canny(im, 80, 160).mean() / 255.,
        im[:, :, 0].mean() / 255., im[:, :, 1].mean() / 255.,
        im[:, :, 2].mean() / 255., hsv[:, :, 1].mean() / 255.,
        hsv[:, :, 2].std() / 255.,
    ], dtype=np.float32)


def features(root: Path, vid: str) -> np.ndarray:
    ims = []
    for i in range(30):
        im = cv2.imread(str(root / vid / f"{i:02d}.jpg"), cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(root / vid / f"{i:02d}.jpg")
        ims.append(im)
    base = np.stack([descriptor(im) for im in ims])
    gray = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
                     for im in ims])
    prev = np.zeros(30, np.float32); nxt = np.zeros(30, np.float32)
    for i in range(30):
        if i: prev[i] = np.mean(np.abs(gray[i] - gray[i-1])) / 255.
        if i + 1 < 30: nxt[i] = np.mean(np.abs(gray[i+1] - gray[i])) / 255.
    return np.c_[base, prev, nxt, np.abs(prev-nxt)].astype(np.float32)


def quality_interactions(x: np.ndarray) -> np.ndarray:
    # These are observable proxies: contrast, sharpness, edge density,
    # saturation/value spread, and local motion.  They do not use annotations.
    q = x[:, [1, 2, 3, 7, 8, 9, 10, 11]]
    # Centering is deliberately not learned from labels; the final scaler is
    # fitted separately inside each source-held-out training fold.
    return np.c_[x, (x[:, :, None] * q[:, None, :]).reshape(len(x), -1)]


def cat_interactions(x: np.ndarray, rows: list[dict], field: str) -> np.ndarray:
    vals = sorted(set(r.get(field, "") or "<MISSING>" for r in rows))
    idx = {v:i for i,v in enumerate(vals)}
    oh = np.zeros((len(rows), len(vals)), np.float32)
    for i,r in enumerate(rows): oh[i, idx[r.get(field, "") or "<MISSING>"]] = 1.
    # Diagnostic only: an annotator field is not a legal inference input.
    return np.c_[x, oh, (x[:, :, None] * oh[:, None, :]).reshape(len(x), -1)]


def center_rank(n):
    t = np.arange(n, dtype=float)
    c = 1. - np.abs(t - (n-1)/2.) / max((n-1)/2., 1.)
    return np.argsort(np.argsort(c, kind="mergesort"), kind="mergesort").astype(float) / max(n-1, 1)


def select(score, k):
    return np.argsort(-score, kind="mergesort")[:max(1, min(int(k), len(score)))]


def proxy_f(ix, y, binary=False):
    z = (y >= 2.).astype(float) if binary else y / 3.
    m = float(z.sum()); s = float(z[ix].sum())
    return 2*s/(len(ix)+m) if len(ix)+m else 1.


def fit_predict(xtr, ytr, ctr, xte, cte):
    scaler = StandardScaler().fit(xtr)
    A = np.c_[np.ones(len(ctr)), ctr]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    base_tr = A @ coef
    model = Ridge(alpha=10.).fit(scaler.transform(xtr), ytr - base_tr)
    # Return signed residual; center rank remains the deployment anchor.
    residual = model.predict(scaler.transform(xte))
    # Keep the residual for ranking, and reconstruct the center trend for the
    # calibration diagnostic reported below.
    target = np.c_[np.ones(len(cte)), cte] @ coef + residual
    return residual, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = read_rows(args.labels)
    vids = sorted({r["video_id"] for r in rows})
    cache = {v: features(args.frames, v) for v in vids}
    X = np.stack([cache[r["video_id"]][int(float(r["time_sec"]))] for r in rows])
    y = np.asarray([GRADE[r["generic_highlight"]] for r in rows], float)
    g = np.asarray([r["video_id"] for r in rows]); t = np.asarray([int(float(r["time_sec"])) for r in rows])
    c = 1. - np.abs(t - 14.5) / 14.5
    arms = {"D2_LOCAL": X, "Q_INTERACTION": quality_interactions(X)}
    for field in ("event_phase", "event_type", "needs_context"):
        arms[f"HUMAN_{field}"] = cat_interactions(X, rows, field)

    # OOF predictions are source-held-out, including all duplicate streams of
    # a source in the held-out fold.
    pred = {name: np.full(len(rows), np.nan) for name in arms}
    target_pred = {name: np.full(len(rows), np.nan) for name in arms}
    fold = {name: [] for name in arms}
    for v in vids:
        tr, te = g != v, g == v
        for name, xx in arms.items():
            p, yp = fit_predict(xx[tr], y[tr], c[tr], xx[te], c[te]); pred[name][te] = p; target_pred[name][te] = yp
            fold[name].append({"video_id": v, "mae": float(np.mean(np.abs(y[te] - yp))),
                               "rho": float(spearmanr(y[te], p).statistic)
                               if len(np.unique(y[te])) > 1 else None})

    source_views = []
    deltas = {name: [] for name in arms}
    for v in vids:
        for stream in sorted({int(r["_stream"]) for r in rows if r["video_id"] == v}):
            ix = np.asarray([i for i,r in enumerate(rows) if r["video_id"] == v and int(r["_stream"]) == stream])
            # protect against duplicate frame rows: one observation per time.
            order = np.argsort(t[ix]); ix = ix[order]
            _, keep = np.unique(t[ix], return_index=True); ix = ix[np.sort(keep)]
            yy, cc = y[ix], center_rank(len(ix)); k = int(round(.85*len(ix)))
            base = proxy_f(select(cc, k), yy)
            rec = {"video_id": v, "stream": stream, "base": base,
                   "base_binary": proxy_f(select(cc, k), yy, True)}
            for name in arms:
                score = cc + .20 * (np.argsort(np.argsort(pred[name][ix], kind="mergesort"), kind="mergesort") / max(len(ix)-1,1) - .5)
                val = proxy_f(select(score, k), yy); valb = proxy_f(select(score, k), yy, True)
                rec[name] = val; rec[name+"_delta"] = val-base
                rec[name+"_binary"] = valb; rec[name+"_binary_delta"] = valb-rec["base_binary"]
                deltas[name].append(val-base)
            source_views.append(rec)
    result = {
        "experiment_id": "R53_QUALITY_CONDITIONED_RESIDUAL_PROBE",
        "status": "PUBLIC_SOURCE_HELD_OUT_COMPLETED",
        "rows": len(rows), "sources": len(vids), "views": len(source_views),
        "fixed_eta": .20, "fixed_ridge_alpha": 10.,
        "hypothesis": "local residual utility is conditional on observable quality/state regime; human fields are diagnostic separators only",
        "arms": {n: {"macro_mae": float(np.mean([z["mae"] for z in fold[n]])),
                     "macro_rho": float(np.nanmean([z["rho"] for z in fold[n]])),
                     "positive_views": int(sum(d > 0 for d in deltas[n])),
                     "macro_proxy_delta": float(np.mean(deltas[n])),
                     "folds": fold[n]} for n in arms},
        "feature_contract": {"D2_LOCAL": "9 RGB descriptors + prev/next gray state",
                             "Q_INTERACTION": "D2 plus fixed products with contrast/sharpness/edge/saturation/motion",
                             "HUMAN_*": "label-field one-hot and local interactions; diagnostic only, not deployable"},
        "source_views": source_views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"experiment_id": result["experiment_id"],
                      "arms": {k:{z:v for z,v in d.items() if z in ("macro_mae","macro_rho","macro_proxy_delta","positive_views")} for k,d in result["arms"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
