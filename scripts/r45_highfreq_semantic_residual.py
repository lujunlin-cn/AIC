#!/usr/bin/env python3
"""R45 4 fps local-state + frozen event-anchor residual candidate.

This is the deployment counterpart of ``r45_highfreq_state_probe.py``.  It
keeps the R26 full-frame B3 parent, exact 85 percent budget, and centre rank;
only a fixed eta=.20 residual changes temporal ordering.  Public labels are
the twelve annotated source clips and the InternVideo2 event score is used as
the frozen semantic anchor already used by R44.  No official labels are read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, frame_descriptor, _rank01, select


def rows_from(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        with p.open(newline="") as f:
            out.extend(dict(x) for x in csv.DictReader(f))
    return out


def decode_highfreq(video: Path) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release(); raise RuntimeError(f"empty video {video}")
    duration = max((n - 1) / fps, 0.0)
    m = max(1, int(np.floor(duration * 4.0 + 1e-7)) + 1)
    target = np.minimum(np.rint(np.arange(m) * fps / 4.0).astype(np.int64), n - 1)
    wanted = {}
    for j, idx in enumerate(target.tolist()):
        wanted.setdefault(int(idx), []).append(j)
    ims = [None] * m
    i = 0
    while i <= int(target[-1]):
        ok, im = cap.read()
        if not ok:
            break
        if i in wanted:
            for j in wanted[i]:
                ims[j] = im.copy()
        i += 1
    cap.release()
    fallback = next((x for x in ims if x is not None), np.zeros((224, 224, 3), np.uint8))
    ims = [x if x is not None else fallback.copy() for x in ims]
    return np.stack(ims), fps, n


def highfreq_features(frames: np.ndarray) -> np.ndarray:
    desc = np.stack([frame_descriptor(x) for x in frames]).astype(np.float32)
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                     for x in frames])
    diff = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
    out = []
    for c in range(len(frames)):
        lo, hi = max(0, c - 2), min(len(frames), c + 3)
        local = diff[max(0, lo - 1):min(len(diff), hi)]
        before = diff[max(0, c - 2):c]
        after = diff[c:min(len(diff), c + 2)]
        bm = float(np.mean(before)) if before.size else 0.0
        am = float(np.mean(after)) if after.size else 0.0
        bx = float(np.max(before)) if before.size else 0.0
        ax = float(np.max(after)) if after.size else 0.0
        im = desc[c]
        ib = desc[max(0, c - 2)]
        ia = desc[min(len(desc) - 1, c + 2)]
        out.append(np.r_[im,
                         float(np.max(local)) if local.size else 0.0,
                         float(np.mean(local)) if local.size else 0.0,
                         bm, am, bx, ax, am - bm, ax - bx,
                         float(np.mean(np.abs(im - ib))),
                         float(np.mean(np.abs(ia - im))),
                         float(np.mean(np.abs(ia - im))) - float(np.mean(np.abs(im - ib)))])
    return np.asarray(out, dtype=np.float32)


def fit(x: np.ndarray, y: np.ndarray, center: np.ndarray, alpha: float = 10.0):
    scaler = StandardScaler().fit(x)
    A = np.c_[np.ones(len(center)), center]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y - A @ coef)
    return scaler, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--public-scores", type=Path, required=True)
    ap.add_argument("--r43-report", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--public-video-root", type=Path, default=None)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eta", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--keep", type=float, default=0.85)
    args = ap.parse_args()

    labels = rows_from(args.labels)
    public_video_root = args.public_video_root or args.video_root
    ps = json.loads(args.public_scores.read_text())
    public_cache = {}
    x, y, c = [], [], []
    for row in labels:
        vid = row["video_id"]
        if vid not in public_cache:
            frames, _, _ = decode_highfreq(public_video_root / f"{vid}.mp4")
            public_cache[vid] = highfreq_features(frames)
        t = int(float(row["time_sec"]))
        j = min(int(round(t * 4.0)), len(public_cache[vid]) - 1)
        sem = ps["videos"][vid]["event_score"]
        x.append(np.r_[public_cache[vid][j], sem[min(t, len(sem) - 1)]])
        y.append(GRADE[row["generic_highlight"]])
        c.append(1.0 - abs(t - 14.5) / 14.5)
    x = np.asarray(x, np.float32); y = np.asarray(y, float); c = np.asarray(c, float)
    scaler, model = fit(x, y, c, args.alpha)

    parent = [json.loads(line) for line in args.parent.read_text().splitlines() if line.strip()]
    report = json.loads(args.r43_report.read_text())
    diagnostics = {str(v["video_id"]): v for v in report["diagnostics"]}
    out, stats = [], []
    for row in parent:
        vid = str(row["video_id"])
        frames, fps, n = decode_highfreq(args.video_root / f"{vid}.mp4")
        hf = highfreq_features(frames)
        anchor = np.asarray(diagnostics[vid]["event_score"], dtype=float)
        hf_times = np.arange(len(hf), dtype=float) / 4.0
        anchor_times = np.linspace(0.0, max((n - 1) / max(fps, 1e-6), 0.0), len(anchor))
        sem_hf = np.interp(hf_times, anchor_times, anchor)
        rr_hf = model.predict(scaler.transform(np.c_[hf, sem_hf]))
        frame_times = np.arange(n, dtype=float) / max(fps, 1e-6)
        residual = np.interp(frame_times, hf_times, rr_hf)
        ids = np.asarray(sorted(int(p["frame"]) for p in row["predictions"]), dtype=np.int64)
        if len(ids) != n:
            raise RuntimeError(f"parent must cover all frames for {vid}: {len(ids)} != {n}")
        center = 1.0 - np.abs(frame_times / max(frame_times[-1], 1e-6) - 0.5) * 2.0
        fused = _rank01(center[ids]) + args.eta * (_rank01(residual[ids]) - 0.5)
        k = int(round(args.keep * n))
        chosen = ids[select(fused, k)]
        old = {int(p["frame"]): p for p in row["predictions"]}
        preds = [old[int(i)] for i in sorted(chosen)]
        stats.append({"video_id": vid, "n": n, "k": k,
                      "changed": int([p["frame"] for p in preds] != [p["frame"] for p in row["predictions"]])})
        q = dict(row); q["predictions"] = preds
        q["model_size_mb"] = 450.283377; q["model_params_b"] = 0.450283377
        out.append(q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"experiment_id": "R45_HIGHFREQ_SEMANTIC_RESIDUAL_B3",
               "parent": "R26_POS_CENTER_EXACT085_B3", "videos": len(out),
               "predictions": int(sum(len(r["predictions"]) for r in out)),
               "changed_videos": int(sum(x["changed"] for x in stats)),
               "eta": args.eta, "alpha": args.alpha, "keep": args.keep,
               "feature_contract": "4fps local state + interpolated frozen IV2 event score",
               "boxes": "copied from parent", "output": str(args.output)}
    args.output.with_suffix(".summary.json").write_text(json.dumps({**summary, "per_video": stats}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
