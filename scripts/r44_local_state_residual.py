#!/usr/bin/env python3
"""R44: source-held-out human-label probe and frozen-center residual scorer.

The labels are a small mechanism diagnostic, not official temporal ground
truth.  This script therefore keeps the center position rank as the primary
decision and adds only a capacity-limited residual predicted from local image
state.  It can run both the source-held-out public probe and the official
426-video frame scorer used to build the candidate package.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


GRADE = {"0普通": 0.0, "1有信息": 1.0, "2值得保留": 2.0, "3高光核心": 3.0}


def read_labels(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="") as f:
            rows.extend(dict(r) for r in csv.DictReader(f))
    return rows


def frame_descriptor(im: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    edge = cv2.Canny(im, 80, 160).mean() / 255.0
    # Deliberately small, interpretable descriptor.  No pretrained model is
    # fit to the human labels, reducing the chance of fitting source identity.
    return np.asarray([
        gray.mean() / 255.0,
        gray.std() / 255.0,
        cv2.Laplacian(gray, cv2.CV_32F).var() / 10000.0,
        edge,
        im[:, :, 0].mean() / 255.0,
        im[:, :, 1].mean() / 255.0,
        im[:, :, 2].mean() / 255.0,
        hsv[:, :, 1].mean() / 255.0,
        hsv[:, :, 2].std() / 255.0,
    ], dtype=np.float32)


def public_features(frame_root: Path, video_id: str) -> np.ndarray:
    ims = []
    for i in range(30):
        p = frame_root / video_id / f"{i:02d}.jpg"
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(p)
        ims.append(im)
    base = np.stack([frame_descriptor(im) for im in ims])
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in ims]
    prev = np.zeros(30, dtype=np.float32)
    nxt = np.zeros(30, dtype=np.float32)
    for i in range(30):
        if i:
            prev[i] = np.mean(np.abs(gray[i] - gray[i - 1])) / 255.0
        if i + 1 < 30:
            nxt[i] = np.mean(np.abs(gray[i + 1] - gray[i])) / 255.0
    local = np.stack([prev, nxt, np.abs(prev - nxt)], axis=1)
    return np.concatenate([base, local], axis=1).astype(np.float32)


def center_rank(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    c = 1.0 - np.abs(t - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
    return _rank01(c)


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / max(len(x) - 1, 1)


def fit_residual(x_train: np.ndarray, y_train: np.ndarray, c_train: np.ndarray,
                 alpha: float = 10.0) -> tuple[StandardScaler, Ridge, float, float]:
    # Fit the content head to a center-detrended target.  The center itself is
    # never learned away; it remains the deployed anchor rank.
    scaler = StandardScaler().fit(x_train)
    # Map the center rank to the observed target range on the training fold.
    A = np.c_[np.ones(len(c_train)), c_train]
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    base = A @ coef
    model = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train - base)
    return scaler, model, float(coef[0]), float(coef[1])


def predict_residual(scaler: StandardScaler, model: Ridge, intercept: float,
                     slope: float, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return model.predict(scaler.transform(x))


def select(score: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(score)))
    # Stable tie handling keeps original frame order deterministic.
    return np.argsort(-score, kind="mergesort")[:k]


def proxy_f(selected: np.ndarray, y: np.ndarray, soft: bool) -> float:
    z = y / 3.0 if soft else (y >= 2.0).astype(np.float64)
    s = float(z[selected].sum())
    m = float(z.sum())
    return 2.0 * s / (len(selected) + m) if len(selected) + m else 1.0


def public_probe(args: argparse.Namespace) -> dict:
    rows = read_labels(args.labels)
    by_vid: dict[str, list[dict]] = {}
    feats: dict[str, np.ndarray] = {}
    for row in rows:
        vid = row["video_id"]
        by_vid.setdefault(vid, []).append(row)
        if vid not in feats:
            feats[vid] = public_features(args.frames, vid)
    # Keep repeated observations in the source-held-out diagnostic.  This
    # makes the uncertainty visible instead of silently creating a consensus.
    x, y, g, t = [], [], [], []
    for vid, rs in by_vid.items():
        for row in rs:
            i = int(float(row["time_sec"]))
            x.append(feats[vid][i])
            y.append(GRADE[row["generic_highlight"]])
            g.append(vid)
            t.append(i)
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float64)
    g = np.asarray(g); t = np.asarray(t)
    c = np.concatenate([center_rank(30)[np.asarray([int(i) for i in np.flatnonzero(g == vid)])]
                        for vid in by_vid]) if False else np.asarray(
                            [1.0 - abs(i - 14.5) / 14.5 for i in t], np.float64)
    # Fold-wise residuals and a frozen center ranking.  Evaluate a predeclared
    # small eta set only as a robustness table; deployment uses eta=0.10.
    pred = np.full(len(y), np.nan)
    fold_rows = []
    for vid in sorted(by_vid):
        tr, te = g != vid, g == vid
        sc, model, itc, sl = fit_residual(x[tr], y[tr], c[tr], alpha=args.alpha)
        pred[te] = predict_residual(sc, model, itc, sl, x[te], c[te])
        fold_rows.append({"video_id": vid, "n": int(te.sum()),
                          "mae_center": float(np.mean(np.abs(y[te] - (itc + sl * c[te])))),
                          "mae_residual": float(np.mean(np.abs(y[te] - (itc + sl*c[te] + pred[te])))),
                          "rho_residual": float(spearmanr(y[te], pred[te]).statistic)
                          if len(np.unique(y[te])) > 1 else None})
    source_effect = []
    eta_table = []
    for eta in args.etas:
        all_f_soft_c, all_f_soft_r, all_f_bin_c, all_f_bin_r = [], [], [], []
        for vid in sorted(by_vid):
            ix = np.flatnonzero(g == vid)
            # The source has 30 frames, but duplicate rows for overlap are
            # evaluated separately; use the first occurrence per time index.
            uniq, keep = np.unique(t[ix], return_index=True)
            yi = y[ix][keep]
            ci = c[ix][keep]
            ri = pred[ix][keep]
            n = len(yi); k = int(round(args.keep * n))
            sr = _rank01(ci) + float(eta) * (_rank01(ri) - 0.5)
            kc = select(ci, k); kr = select(sr, k)
            fc = proxy_f(kc, yi, True); fr = proxy_f(kr, yi, True)
            bc = proxy_f(kc, yi, False); br = proxy_f(kr, yi, False)
            all_f_soft_c.append(fc); all_f_soft_r.append(fr)
            all_f_bin_c.append(bc); all_f_bin_r.append(br)
            if eta == args.etas[0]:
                source_effect.append({"video_id": vid, "center_soft": fc,
                                      "residual_soft": fr,
                                      "delta_soft": fr-fc,
                                      "center_binary": bc,
                                      "residual_binary": br,
                                      "delta_binary": br-bc})
        eta_table.append({"eta": eta,
                          "soft_center": float(np.mean(all_f_soft_c)),
                          "soft_residual": float(np.mean(all_f_soft_r)),
                          "soft_delta": float(np.mean(np.asarray(all_f_soft_r)-np.asarray(all_f_soft_c))),
                          "binary_center": float(np.mean(all_f_bin_c)),
                          "binary_residual": float(np.mean(all_f_bin_r)),
                          "binary_delta": float(np.mean(np.asarray(all_f_bin_r)-np.asarray(all_f_bin_c))),
                          "positive_source_count": int(sum(x["delta_soft"] > 0 for x in source_effect)) if eta == args.etas[0] else None})
    return {"experiment_id": "R44_RESIDUAL_001", "rows": len(rows),
            "unique_sources": len(by_vid), "alpha": args.alpha, "keep": args.keep,
            "deployment_eta": 0.10, "folds": fold_rows,
            "macro_mae_center": float(np.mean([r["mae_center"] for r in fold_rows])),
            "macro_mae_residual": float(np.mean([r["mae_residual"] for r in fold_rows])),
            "eta_table": eta_table, "source_effect_eta_first": source_effect,
            "feature_contract": "9 static descriptors + prev/next gray state; source-held-out Ridge residual"}


def decode_features(video: Path, fps: float, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / native_fps if total > 0 else 0.0
    times = np.linspace(0.0, max(duration - 1.0 / native_fps, 0.0), n)
    ims = []
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec * 1000.0))
        ok, im = cap.read()
        if not ok:
            # Sequential fallback near the end of VFR/H264 streams.
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(int(round(sec*native_fps)), 0), max(total-1, 0)))
            ok, im = cap.read()
        if not ok:
            im = ims[-1].copy() if ims else np.zeros((224, 224, 3), np.uint8)
        ims.append(im)
    cap.release()
    base = np.stack([frame_descriptor(im) for im in ims])
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in ims]
    prev = np.zeros(n, np.float32); nxt = np.zeros(n, np.float32)
    for i in range(n):
        if i: prev[i] = np.mean(np.abs(gray[i]-gray[i-1]))/255.0
        if i+1<n: nxt[i] = np.mean(np.abs(gray[i+1]-gray[i]))/255.0
    return np.c_[base, prev, nxt, np.abs(prev-nxt)].astype(np.float32)


def official_score(args: argparse.Namespace) -> dict:
    rows = [json.loads(x) for x in args.index.read_text().splitlines() if x.strip()]
    labels = read_labels(args.labels)
    by_vid = {}
    X, Y, C = [], [], []
    for r in labels:
        vid = r["video_id"]; by_vid.setdefault(vid, []).append(r)
        i = int(float(r["time_sec"]))
        X.append(public_features(args.frames, vid)[i])
        Y.append(GRADE[r["generic_highlight"]])
        C.append(1.0-abs(i-14.5)/14.5)
    X = np.asarray(X,np.float32); Y=np.asarray(Y,np.float64); C=np.asarray(C,np.float64)
    sc, model, itc, sl = fit_residual(X, Y, C, alpha=args.alpha)
    out_rows=[]; changed=0; total=0; videos=[]
    for r in rows:
        vid = str(r["video_id"]); n = int(r.get("frame_count", 0))
        if not n:
            # Derive from the parent prediction count and the exact budget.
            n = int(round(len(r["predictions"])/args.keep))
        video = args.video_root / f"{vid}.mp4"
        # Score every decoded 1-fps anchor and interpolate to each original
        # frame.  This keeps compute modest and uses only content near anchors.
        cap = cv2.VideoCapture(str(video)); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0); cap.release()
        m = max(4, int(round(total_frames/fps)))
        feat = decode_features(video, fps, m)
        cr = center_rank(m)
        rr = model.predict(sc.transform(feat))
        # Convert anchor residual to each original frame using nearest-second
        # lookup; candidate frame ordering remains over original frame IDs.
        frame_ids = np.arange(total_frames, dtype=np.int64)
        a = np.minimum(np.rint(frame_ids / fps).astype(int), m-1)
        residual = rr[a]
        center = 1.0 - np.abs(frame_ids - (total_frames-1)/2.0)/max((total_frames-1)/2.0,1.0)
        fused = _rank01(center) + float(args.eta) * (_rank01(residual)-0.5)
        k = int(round(args.keep * total_frames))
        chosen = set(select(fused, k).tolist())
        old = {int(p["frame"]): p for p in r["predictions"]}
        preds = [old[i] for i in sorted(chosen) if i in old]
        # Parent predictions should cover exactly k frames.  Select only from
        # the parent frame universe so spatial boxes remain byte-identical.
        if len(preds) != len(r["predictions"]):
            parent_ids = np.asarray(sorted(old), np.int64)
            fs = fused[parent_ids]
            sel = parent_ids[select(fs, len(parent_ids))]
            preds = [old[int(i)] for i in sorted(sel)]
        if [p["frame"] for p in preds] != [p["frame"] for p in r["predictions"]]: changed += 1
        total += len(preds); videos.append({"video_id":vid,"parent_count":len(old),"new_count":len(preds)})
        q = dict(r); q["predictions"] = preds; out_rows.append(q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"experiment_id":"R44_RESIDUAL_001_OFFICIAL", "videos":len(out_rows),
            "predictions":total,"changed_videos":changed,"eta":args.eta,"keep":args.keep,
            "feature_contract":"1fps handcrafted local-state residual; boxes copied from R26 parent",
            "output":str(args.output)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--labels", nargs=2, type=Path, required=True); p.add_argument("--frames", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--alpha", type=float, default=10.0); p.add_argument("--keep", type=float, default=.85); p.add_argument("--etas", type=float, nargs="+", default=[.05,.10,.15,.20])
    o = sub.add_parser("official"); o.add_argument("--labels", nargs=2, type=Path, required=True); o.add_argument("--frames", type=Path, required=True); o.add_argument("--index", type=Path, required=True); o.add_argument("--video-root", type=Path, required=True); o.add_argument("--parent", type=Path, required=True); o.add_argument("--output", type=Path, required=True); o.add_argument("--eta", type=float, default=.10); o.add_argument("--keep", type=float, default=.85); o.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    if args.cmd == "probe": out = public_probe(args)
    else:
        # official() reads parent itself to avoid accidentally using a stale
        # package.  The explicit parent argument is copied into the index path.
        args.index = args.parent
        out = official_score(args)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
