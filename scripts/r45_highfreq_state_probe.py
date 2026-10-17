#!/usr/bin/env python3
"""R45 high-frequency local-state diagnostic on the public annotation sources.

The R44 residual uses one-second local descriptors.  This probe asks a narrower
question: do sub-second state changes add source-held-out information for the
same annotated second?  It decodes the twelve public 30 s clips at 4 fps and
builds a small, fixed feature contract from the centre frame and its +/-0.25
and +/-0.5 s neighbours.  It never reads official test labels or official
scores and does not sweep deployment coefficients.

This is a mechanism diagnostic, not an AIC score estimate.  A positive result
only justifies a later fixed 4 fps boundary residual; it does not itself make a
submission package.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, frame_descriptor, _rank01, select, proxy_f


def load_labels(paths: list[Path]) -> list[dict]:
    rows = []
    for stream, path in enumerate(paths):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["_stream"] = stream
                rows.append(row)
    return rows


def decode_4fps(video: Path, seconds: int = 30) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    # Sequential decode avoids VFR seek errors.  The clips are short, so this
    # is cheap and gives an auditable native-frame -> 4 fps contract.
    native = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        native.append(im)
    cap.release()
    if not native:
        raise RuntimeError(f"empty video {video}")
    n = min(seconds * 4, int(np.floor((len(native) - 1) / fps * 4)) + 1)
    frames = []
    for j in range(n):
        idx = min(int(round(j * fps / 4.0)), len(native) - 1)
        frames.append(native[idx])
    return np.stack(frames), fps


def feature_track(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Return 1 fps-compatible and sub-second feature tracks."""
    desc = np.stack([frame_descriptor(im) for im in frames]).astype(np.float32)
    gray = np.stack([
        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for im in frames
    ])
    diff = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
    # 4 fps centre index for each labelled second.  Keep boundary handling
    # explicit; no wraparound or fabricated frames.
    centre = np.arange(30, dtype=int) * 4
    centre = np.minimum(centre, len(frames) - 1)
    dprev = np.r_[0.0, diff]
    dnext = np.r_[diff, 0.0]
    # Compact local-state contract: 9 static values plus robust sub-second
    # peaks/asymmetry and before/after state displacement (13 dimensions).
    out = []
    for c in centre:
        lo, hi = max(0, c - 2), min(len(frames), c + 3)
        local = diff[max(0, lo - 1):min(len(diff), hi)]
        before = diff[max(0, c - 2):c] if c > 0 else np.zeros(1)
        after = diff[c:min(len(diff), c + 2)]
        bmean = float(np.mean(before)) if before.size else 0.0
        amean = float(np.mean(after)) if after.size else 0.0
        bmax = float(np.max(before)) if before.size else 0.0
        amax = float(np.max(after)) if after.size else 0.0
        # Frame-state displacement at half and one-second horizons.
        im = desc[c]
        ihalf = desc[max(0, c - 2)]
        ifwd = desc[min(len(desc) - 1, c + 2)]
        state_before = float(np.mean(np.abs(im - ihalf)))
        state_after = float(np.mean(np.abs(ifwd - im)))
        vals = np.r_[im,
                     float(np.max(local)) if local.size else 0.0,
                     float(np.mean(local)) if local.size else 0.0,
                     bmean, amean, bmax, amax,
                     amean - bmean, amax - bmax,
                     state_before, state_after, state_after - state_before]
        out.append(vals)
    return {"highfreq": np.asarray(out, dtype=np.float32),
            "static": desc,
            "diff": diff.astype(np.float32)}


def fit_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
            alpha: float = 10.0) -> tuple[np.ndarray, dict]:
    pred = np.full(len(y), np.nan)
    source = {}
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        scaler = StandardScaler().fit(x[tr])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[tr]), y[tr])
        pred[te] = model.predict(scaler.transform(x[te]))
        source[g] = {
            "n": int(te.sum()),
            "mae": float(np.mean(np.abs(pred[te] - y[te]))),
            "spearman": float(spearmanr(y[te], pred[te]).statistic)
            if len(np.unique(y[te])) > 1 else None,
        }
    vals = [v["mae"] for v in source.values()]
    rho = [v["spearman"] for v in source.values() if v["spearman"] is not None]
    return pred, {"source": source, "macro_mae": float(np.mean(vals)),
                  "macro_spearman": float(np.mean(rho)) if rho else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--public-scores", type=Path,
                    help="optional frozen public InternVideo2 scores")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = load_labels(args.labels)
    by_vid = {}
    for row in rows:
        by_vid.setdefault(row["video_id"], []).append(row)
    tracks = {}
    for vid in sorted(by_vid):
        frames, fps = decode_4fps(args.videos / f"{vid}.mp4")
        tracks[vid] = feature_track(frames)
    sem = None
    if args.public_scores:
        raw = json.loads(args.public_scores.read_text())
        sem = {v: np.asarray(d["event_score"], dtype=np.float32)
               for v, d in raw["videos"].items()}
    ys, groups, xs, xsem, streams, ordered_rows = [], [], [], [], [], []
    for vid, rs in by_vid.items():
        for row in rs:
            t = min(int(float(row["time_sec"])), 29)
            ys.append(GRADE[row["generic_highlight"]]); groups.append(vid)
            streams.append(int(row["_stream"]))
            ordered_rows.append(row)
            xs.append(tracks[vid]["highfreq"][t])
            if sem is not None:
                xsem.append(float(sem[vid][min(t, len(sem[vid]) - 1)]))
    y = np.asarray(ys, dtype=np.float32)
    groups = np.asarray(groups)
    streams = np.asarray(streams, dtype=int)
    x_hf = np.asarray(xs, dtype=np.float32)
    # Arms are nested so the comparison isolates sub-second information.
    # 1fps static is the existing R44 image contract; HF adds only fixed local
    # change features.  A full feature arm is included to test complementarity.
    x_static = x_hf[:, :9]
    # The three 1 fps state channels used by R44, reconstructed from the 4 fps
    # track at the exact one-second anchors.  This makes the increment over the
    # deployed contract explicit rather than comparing against a weaker arm.
    x_local = np.c_[x_static, x_hf[:, 11], x_hf[:, 12],
                    np.abs(x_hf[:, 15])]
    x_delta = x_hf[:, 9:]
    arms = {"D1_STATIC_1FPS": x_static,
            "D2_R44_LOCAL_1FPS": x_local,
            "D3_R44_LOCAL_PLUS_HF": x_hf}
    if sem is not None:
        sx = np.asarray(xsem, dtype=np.float32)[:, None]
        arms["D4_R44_LOCAL_PLUS_HF_IV2"] = np.c_[x_hf, sx]
    out = {"experiment_id": "R45_HIGHFREQ_STATE_PROBE",
           "status": "PUBLIC_SOURCE_HELD_OUT_COMPLETED",
           "rows": int(len(y)), "sources": int(len(by_vid)),
           "decode": {"target_fps": 4, "native_fps":
                      {v: float(decode_4fps(args.videos / f"{v}.mp4")[1])
                       for v in sorted(by_vid)}},
           "alpha": 10.0, "public_scores_used": bool(sem is not None),
           "feature_contract":
           "9 1fps descriptors + fixed +/-0.25/0.5s diff/state features",
           "arms": {}}
    oof = {}
    for name, x in arms.items():
        pred, metrics = fit_oof(x, y, groups, alpha=10.0)
        out["arms"][name] = metrics
        oof[name] = pred
    # Keep a paired source table for decisions; no optimisation on this table.
    base = out["arms"]["D1_STATIC_1FPS"]["source"]
    full = out["arms"]["D3_R44_LOCAL_PLUS_HF"]["source"]
    out["paired_delta_mae_full_minus_static"] = {
        g: float(full[g]["mae"] - base[g]["mae"]) for g in sorted(base)
    }
    # Fixed descriptive strata: on a 30 s public source the R26 parent is
    # [2,28), while its two 5% edge bands are [0,2) and [26,30).
    strata = {}
    for i, row in enumerate(ordered_rows):
        t = int(float(row["time_sec"]))
        if t < 2 or t >= 28:
            region = "outside_or_parent_edge_band"
        elif t < 4 or t >= 26:
            region = "near_parent_boundary"
        else:
            region = "center_core"
        key = f"{region}|{row.get('event_phase', '')}"
        strata.setdefault(key, []).append(i)
    out["strata"] = {}
    for key, ix in sorted(strata.items()):
        ix = np.asarray(ix, dtype=int)
        rec = {"n": int(len(ix)), "mean_grade": float(np.mean(y[ix]))}
        for name in ("D2_R44_LOCAL_1FPS", "D3_R44_LOCAL_PLUS_HF",
                     "D4_R44_LOCAL_PLUS_HF_IV2"):
            if name in oof:
                rec[name + "_mae"] = float(np.mean(np.abs(oof[name][ix] - y[ix])))
        out["strata"][key] = rec
    # One fixed boundary-gated decision (eta=.20, the R44 deployment value)
    # tests whether the HF signal is useful precisely where it survives the
    # R26 centre parent.  This is a diagnostic proxy, not an official metric.
    boundary = {}
    for name in ("D2_R44_LOCAL_1FPS", "D3_R44_LOCAL_PLUS_HF",
                 "D4_R44_LOCAL_PLUS_HF_IV2"):
        if name not in oof:
            continue
        vals = []
        for vid in sorted(set(groups)):
            ixv = np.flatnonzero(groups == vid)
            for stream in sorted(set(streams[ixv].tolist())):
                ix = ixv[streams[ixv] == stream]
                times, keep = np.unique(np.asarray([int(float(ordered_rows[i]["time_sec"])) for i in ix]), return_index=True)
                ix = ix[keep]; yy = y[ix]; n = len(times); k = int(round(.85 * n))
                lo, hi = (n - k) // 2, (n - k) // 2 + k
                band = max(1, int(round(.05 * n)))
                protected = np.arange(min(hi, lo + band), max(min(hi, lo + band), hi - band), dtype=int)
                allowed = np.arange(max(0, lo - band), min(n, hi + band), dtype=int)
                free = np.setdiff1d(allowed, protected, assume_unique=True)
                need = k - len(protected)
                center = 1.0 - np.abs(times - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
                fused = _rank01(center) + .20 * (_rank01(oof[name][ix]) - .5)
                chosen = np.sort(np.r_[protected, free[select(fused[free], need)]])
                base = np.arange(lo, hi, dtype=int)
                vals.append({"video_id": vid, "stream": stream,
                             "center_soft": proxy_f(base, yy, True),
                             "boundary_soft": proxy_f(chosen, yy, True),
                             "delta_soft": proxy_f(chosen, yy, True) - proxy_f(base, yy, True),
                             "center_binary": proxy_f(base, yy, False),
                             "boundary_binary": proxy_f(chosen, yy, False),
                             "delta_binary": proxy_f(chosen, yy, False) - proxy_f(base, yy, False)})
        boundary[name] = {"n_views": len(vals),
                          "macro_delta_soft": float(np.mean([v["delta_soft"] for v in vals])),
                          "positive_views_soft": int(sum(v["delta_soft"] > 0 for v in vals)),
                          "macro_delta_binary": float(np.mean([v["delta_binary"] for v in vals])),
                          "views": vals}
    out["fixed_boundary_proxy"] = boundary
    full_proxy = {}
    for name in oof:
        vals = []
        for vid in sorted(set(groups)):
            ixv = np.flatnonzero(groups == vid)
            for stream in sorted(set(streams[ixv].tolist())):
                ix = ixv[streams[ixv] == stream]
                times, keep = np.unique(np.asarray([int(float(ordered_rows[i]["time_sec"])) for i in ix]), return_index=True)
                ix = ix[keep]; yy = y[ix]; n = len(times); k = int(round(.85 * n))
                center = 1.0 - np.abs(times - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
                fused = _rank01(center) + .20 * (_rank01(oof[name][ix]) - .5)
                chosen = select(fused, k); base = select(center, k)
                vals.append(proxy_f(chosen, yy, True) - proxy_f(base, yy, True))
        full_proxy[name] = {"n_views": len(vals), "macro_delta_soft": float(np.mean(vals)),
                            "positive_views_soft": int(sum(v > 0 for v in vals))}
    out["fixed_full_proxy"] = full_proxy
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(y),
                      "sources": len(by_vid), "arms": out["arms"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
