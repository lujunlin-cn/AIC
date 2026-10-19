#!/usr/bin/env python3
"""R45: boundary protected human calibrated residual.

R44 showed a small positive official delta while occasionally replacing a
large block on a long video.  R45 keeps the R26 center-85 parent as a hard
anchor: the central 75 percent of that parent window is forced to remain, and
the residual can choose only the two local boundary bands (five percent of
the full video on each side).  The remaining ten percent of the output
budget is selected from those bands.  This is a single predeclared rule, not
an eta/alpha sweep.

The fitting and feature contract intentionally match R44 exactly.  B3 boxes
are copied byte-for-byte from the full-frame parent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_semantic_calibrated_residual import (
    decode_anchors,
    fit,
    public_features,
    read_rows,
    _rank01,
    select,
)


def make_set(ids: np.ndarray, residual: np.ndarray, eta: float,
             keep: float = 0.85, band_fraction: float = 0.05):
    """Return the exact-K boundary protected selection and diagnostics."""
    n = len(ids)
    k = int(round(keep * n))
    lo = (n - k) // 2
    hi = lo + k  # exclusive; this is exactly the R26 center parent window
    band = max(1, int(round(band_fraction * n)))

    # Protect the central portion of the R26 window.  The free candidate set
    # is local to both R26 boundaries and excludes the remote video tails.
    p0 = min(hi, lo + band)
    p1 = max(p0, hi - band)
    protected = np.arange(p0, p1, dtype=np.int64)
    c0 = max(0, lo - band)
    c1 = min(n, hi + band)
    allowed = np.arange(c0, c1, dtype=np.int64)
    free = np.setdiff1d(allowed, protected, assume_unique=True)
    need = k - len(protected)
    if need < 0 or need > len(free):
        raise RuntimeError(
            f"invalid boundary geometry n={n} k={k} band={band} "
            f"protected={len(protected)} free={len(free)}"
        )

    center = 1.0 - np.abs(ids / max(float(n - 1), 1.0) - 0.5) * 2.0
    fused = _rank01(center) + float(eta) * (_rank01(residual) - 0.5)
    order = free[select(fused[free], need)] if need else np.empty(0, dtype=np.int64)
    chosen = np.sort(np.concatenate([protected, order]))
    if len(chosen) != k or len(np.unique(chosen)) != k:
        raise RuntimeError(f"selection contract failed n={n} k={k} got={len(chosen)}")
    parent = np.arange(lo, hi, dtype=np.int64)
    return chosen, {
        "n": n, "k": k, "band": band, "parent_lo": lo, "parent_hi": hi,
        "protected": int(len(protected)), "free": int(len(free)),
        "changed": int(len(set(parent.tolist()) ^ set(chosen.tolist())) // 2),
        "outside_added": int(np.sum((chosen < lo) | (chosen >= hi))),
    }


def official(args: argparse.Namespace) -> dict:
    parent_rows = [json.loads(x) for x in args.parent.read_text().splitlines() if x.strip()]
    labels = read_rows(args.labels)
    X, Y, C, S = [], [], [], []
    cache: dict[str, np.ndarray] = {}
    sem = json.loads(args.public_scores.read_text())
    for r in labels:
        vid = r["video_id"]
        if vid not in cache:
            cache[vid] = public_features(args.frames, vid)
        t = int(float(r["time_sec"]))
        X.append(cache[vid][t])
        from scripts.r44_local_state_residual import GRADE
        Y.append(GRADE[r["generic_highlight"]])
        C.append(1.0 - abs(t - 14.5) / 14.5)
        S.append(sem["videos"][vid]["event_score"][min(t, len(sem["videos"][vid]["event_score"]) - 1)])
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, float)
    C = np.asarray(C, float)
    S = np.asarray(S, float)
    sx, head = fit(np.c_[X, S], Y, C, args.alpha)

    report = json.loads(args.r43_report.read_text())
    diagnostics = {str(x["video_id"]): x for x in report["diagnostics"]}
    out = []
    stats = []
    total = 0
    for row in parent_rows:
        vid = str(row["video_id"])
        anchors = np.asarray(diagnostics[vid]["event_score"], float)
        feat, fps, n = decode_anchors(args.video_root / f"{vid}.mp4", len(anchors))
        local = np.asarray(feat, np.float32)
        rr_anchor = head.predict(sx.transform(np.c_[local, anchors]))
        times = np.arange(n, dtype=float) / max(fps, 1e-6)
        anchor_times = np.linspace(0.0, max((n - 1) / max(fps, 1e-6), 0.0), len(anchors))
        rr = np.interp(times, anchor_times, rr_anchor)

        parent = {int(p["frame"]): p for p in row["predictions"]}
        ids = np.asarray(sorted(parent), dtype=np.int64)
        if len(ids) != n or not np.array_equal(ids, np.arange(n, dtype=np.int64)):
            raise RuntimeError(f"full B3 coverage mismatch for {vid}: {len(ids)} vs {n}")
        chosen, st = make_set(ids, rr, args.eta, args.keep, args.band_fraction)
        preds = [parent[int(i)] for i in chosen]
        q = dict(row)
        q["model_size_mb"] = 450.283377
        q["model_params_b"] = 0.450283377
        q["predictions"] = preds
        out.append(q)
        stats.append({"video_id": vid, **st})
        total += len(preds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "experiment_id": "R45_BOUNDARY_PROTECTED_RESIDUAL",
        "videos": len(out), "predictions": total, "eta": args.eta,
        "alpha": args.alpha, "keep": args.keep,
        "band_fraction": args.band_fraction,
        "parent": "R26_POS_CENTER_EXACT085_B3",
        "boxes": "byte copied from full-frame B3 parent",
        "stats": {
            "changed_videos": int(sum(s["changed"] > 0 for s in stats)),
            "mean_changed": float(np.mean([s["changed"] for s in stats])),
            "max_changed": int(max(s["changed"] for s in stats)),
            "mean_outside_added": float(np.mean([s["outside_added"] for s in stats])),
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False))
    args.summary.write_text(json.dumps({**summary, "per_video": stats}, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs=2, type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--public-scores", type=Path, required=True)
    p.add_argument("--r43-report", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--eta", type=float, default=0.20)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--keep", type=float, default=0.85)
    p.add_argument("--band-fraction", type=float, default=0.05)
    official(p.parse_args())


if __name__ == "__main__":
    main()
