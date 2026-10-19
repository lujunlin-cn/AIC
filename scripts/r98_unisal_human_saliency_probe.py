#!/usr/bin/env python3
"""R98: test a frozen temporal human-saliency path on native crop GT.

This is a deliberately narrow transfer test inspired by AIM 2024 Video
Saliency Prediction (arXiv:2409.14827).  UNISAL is used only as a frozen
visual observation path; audio and saliency-as-highlight labels are excluded.
For each LIVE keyframe, the model produces a temporal saliency map and we
compare legal crop windows centred at its saliency mass against the existing
centre and Qwen-point controls.  The target is native crop IoU on source held
out tables, not a saliency metric.

The script runs on the remote environment because the existing UNISAL package
and weights live under /data/aic.  It writes only compact map statistics and
source-level metrics, never videos or weights.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def crop_box(w: int, h: int, ratio: tuple[float, float], cx: float, cy: float = .5):
    rw, rh = ratio
    # Largest crop with the requested width/height ratio inside the source.
    if w / h >= rw / rh:
        ch = float(h)
        cw = ch * rw / rh
    else:
        cw = float(w)
        ch = cw * rh / rw
    x = float(np.clip(cx * w - cw / 2., 0., w - cw))
    y = float(np.clip(cy * h - ch / 2., 0., h - ch))
    return np.array([x, y, x + cw, y + ch], dtype=np.float64)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0])); y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2])); y1 = min(float(a[3]), float(b[3]))
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    aa = max(0., float(a[2] - a[0])) * max(0., float(a[3] - a[1]))
    bb = max(0., float(b[2] - b[0])) * max(0., float(b[3] - b[1]))
    return inter / max(aa + bb - inter, 1e-12)


def saliency_stats(smap: np.ndarray) -> dict[str, float]:
    m = np.asarray(smap, dtype=np.float64)
    m -= np.nanmin(m)
    m = np.nan_to_num(m, nan=0., posinf=0., neginf=0.)
    if float(m.sum()) <= 1e-12:
        m[...] = 1.
    den = float(m.sum())
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    cx = float((m * xx).sum() / den / max(m.shape[1] - 1, 1))
    cy = float((m * yy).sum() / den / max(m.shape[0] - 1, 1))
    # A robust alternative to a single peak: centre of the middle 50% mass.
    flat = m.ravel(); order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]) / den
    keep = order[csum <= .50]
    if len(keep) < 1:
        keep = order[:1]
    ym, xm = np.unravel_index(keep, m.shape)
    qx = float(np.mean(xm) / max(m.shape[1] - 1, 1))
    qy = float(np.mean(ym) / max(m.shape[0] - 1, 1))
    return {"cx": cx, "cy": cy, "q50x": qx, "q50y": qy,
            "peak": float(np.max(m) / den),
            "entropy": float(-(m / den * np.log(m / den + 1e-12)).sum())}


def load_table(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def run_table(table: list[dict], keyframe_root: Path, unisal_handler, limit: int | None):
    rows = []
    for ui, u in enumerate(table[:limit]):
        vid = str(u["vid"])
        folder = keyframe_root / vid
        paths = []
        for frame in u["keys"]:
            p = folder / f"{int(frame)}.png"
            if not p.is_file():
                raise FileNotFoundError(str(p))
            paths.append(p)
        ims = []
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f"decode failed: {p}")
            ims.append(im)
        maps = unisal_handler.predictions_from_frames_in_memory(
            unisal_handler._R98_TRAINER, ims, [], "")
        stats = [saliency_stats(x) for x in maps]
        # Only keyframes with a native GT row are used.  This avoids silently
        # treating the intervening unlabelled frames as negatives.
        gpos = {int(f): i for i, f in enumerate(np.asarray(u["gf"]).tolist())}
        vals = {k: [] for k in ("center", "qwen", "unisal", "unisal_x", "mass50")}
        for j, frame in enumerate(u["keys"]):
            if j >= len(np.asarray(u["Y"])):
                continue
            if int(frame) not in gpos:
                continue
            gt = np.asarray(u["gt"])[0, gpos[int(frame)]]
            qv = float(np.asarray(u["Y"])[j, -1])
            if not np.isfinite(qv):
                continue
            vals["center"].append(iou(crop_box(u["W"], u["H"], tuple(u["ratio"]), .5), gt))
            vals["qwen"].append(qv)
            s = stats[j]
            vals["unisal"].append(iou(crop_box(u["W"], u["H"], tuple(u["ratio"]), s["cx"], s["cy"]), gt))
            vals["unisal_x"].append(iou(crop_box(u["W"], u["H"], tuple(u["ratio"]), s["cx"]), gt))
            vals["mass50"].append(iou(crop_box(u["W"], u["H"], tuple(u["ratio"]), s["q50x"], s["q50y"]), gt))
        if vals["center"]:
            rows.append({"vid": vid, "stats": stats,
                         "metrics": {k: float(np.mean(v)) for k, v in vals.items()},
                         "n": len(vals["center"])})
        if (ui + 1) % 10 == 0:
            print(json.dumps({"processed": ui + 1, "sources": len(rows)}), flush=True)
    return rows


def summarize(rows: list[dict], seed: int = 20261019):
    methods = ("center", "qwen", "unisal", "unisal_x", "mass50")
    per = {m: np.asarray([r["metrics"][m] for r in rows], dtype=np.float64) for m in methods}
    rng = np.random.default_rng(seed); out = {"sources": len(rows), "methods": {}}
    for m in methods:
        out["methods"][m] = {"mean": float(per[m].mean()), "median": float(np.median(per[m]))}
    out["paired_vs_qwen"] = {}
    for m in methods:
        if m == "qwen":
            continue
        d = per[m] - per["qwen"]
        boot = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        out["paired_vs_qwen"][m] = {"mean": float(d.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--keyframes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "/data/aic/datasets/RetargetVid/3rd_party_libs/unisal")
    import unisal_handler
    # Keep the model loaded exactly once; this attribute avoids a second model
    # construction for each source while retaining the upstream inference code.
    unisal_handler._R98_TRAINER = unisal_handler.init_unisal_for_frames()
    rows = run_table(load_table(args.table), args.keyframes, unisal_handler, args.limit)
    out = {"protocol": "R98_UNISAL_HUMAN_SALIENCY_NATIVE_CROP",
           "paper": "arXiv:2409.14827",
           "audio": "excluded_due_to_project_negative_transfer_evidence",
           "summary": summarize(rows), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
