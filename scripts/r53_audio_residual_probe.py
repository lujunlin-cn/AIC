#!/usr/bin/env python3
"""R53: source-held-out audio residual probe.

The V1.3 audit found a non-trivial subset of high-grade frames whose label
explicitly depended on sound or language.  This probe asks a narrower
question: do cheap, time-aligned audio descriptors add held-out ranking
signal beyond the frozen center and 4-fps visual local-state residual?

It is a diagnostic only.  It does not read official labels and it never
constructs a submission package.  Audio is extracted from the public 30 s
clips with ffmpeg; missing audio is represented by a mask and zero features.
All fits are source-held-out, and repeated project exports are evaluated as
separate views to expose annotation uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import stft
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


GRADE = {"0普通": 0.0, "1有信息": 1.0, "2值得保留": 2.0, "3高光核心": 3.0}


def read_rows(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for stream, path in enumerate(paths):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                q = dict(row)
                q["_stream"] = stream
                out.append(q)
    return out


def decode_audio(path: Path, sr: int = 16000) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
           "-ac", "1", "-ar", str(sr), "pipe:1"]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)


def audio_features(path: Path, seconds: int = 30, sr: int = 16000) -> np.ndarray:
    """Return one robust, low-dimensional descriptor per integer second."""
    audio = decode_audio(path, sr)
    out: list[np.ndarray] = []
    # A 0.75 s centered window sees speech and applause without smearing a
    # short onset over the whole 1 s label bin.
    half = int(round(0.375 * sr))
    for t in range(seconds):
        c = int(round(t * sr))
        lo, hi = max(0, c - half), min(len(audio), c + half)
        x = audio[lo:hi]
        if x.size < sr // 20:
            out.append(np.zeros(18, dtype=np.float32))
            continue
        # A slice of the ffmpeg buffer is read-only; make a writable copy
        # before centering and sanitising it.
        x = np.nan_to_num(x, copy=True)
        x = x - float(np.mean(x))
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        peak = float(np.max(np.abs(x)))
        zcr = float(np.mean(x[1:] * x[:-1] < 0)) if x.size > 1 else 0.0
        # STFT statistics are deterministic and do not use a learned audio
        # model.  Missing/no-audio clips remain all zeros plus the mask.
        _, _, z = stft(x, fs=sr, nperseg=min(512, len(x)), noverlap=256,
                       boundary=None, padded=False)
        p = np.mean(np.abs(z) ** 2, axis=1) if z.size else np.zeros(257)
        freqs = np.linspace(0.0, sr / 2.0, len(p), dtype=np.float32)
        total = float(np.sum(p) + 1e-12)
        centroid = float(np.sum(freqs * p) / total) / (sr / 2.0)
        cumsum = np.cumsum(p) / total
        rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cumsum, .85)))]) / (sr / 2.0)
        bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 8000)]
        band = [float(np.sum(p[(freqs >= a) & (freqs < b)]) / total) for a, b in bands]
        flatness = float(np.exp(np.mean(np.log(p + 1e-12))) / (np.mean(p) + 1e-12))
        # loudness in log units makes the ridge less sensitive to recording
        # gain; retain raw rms as a separate cue.
        out.append(np.asarray([np.log1p(1000.0 * rms), rms, peak, zcr,
                               centroid, rolloff, *band, flatness,
                               float(np.mean(np.abs(np.diff(x))))], np.float32))
    base = np.stack(out).astype(np.float32)
    # Add centered one-second change and a local peak so an onset is not
    # mistaken for a level offset.  This yields 18 dimensions including the
    # explicit audio-present mask.
    d = np.zeros_like(base)
    if len(base) > 1:
        d[1:] = base[1:] - base[:-1]
    peak = np.max(base[:, 0:1] if len(base) else np.zeros((0, 1)), axis=1, keepdims=True)
    present = (np.max(np.abs(base), axis=1, keepdims=True) > 1e-8).astype(np.float32)
    return np.c_[base, d[:, 0:4], present].astype(np.float32)


def frame_descriptor(im: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    return np.asarray([
        gray.mean() / 255.0, gray.std() / 255.0,
        cv2.Laplacian(gray, cv2.CV_32F).var() / 10000.0,
        cv2.Canny(im, 80, 160).mean() / 255.0,
        im[:, :, 0].mean() / 255.0, im[:, :, 1].mean() / 255.0,
        im[:, :, 2].mean() / 255.0, hsv[:, :, 1].mean() / 255.0,
        hsv[:, :, 2].std() / 255.0,
    ], np.float32)


def visual_features(path: Path, seconds: int = 30) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    for t in range(seconds):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ok, im = cap.read()
        if not ok:
            frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), np.uint8))
        else:
            frames.append(im)
    cap.release()
    desc = np.stack([frame_descriptor(x) for x in frames]).astype(np.float32)
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                     for x in frames])
    diff = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
    prev = np.r_[0.0, diff]
    nxt = np.r_[diff, 0.0]
    return np.c_[desc, prev, nxt, np.abs(prev - nxt)].astype(np.float32)


def rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) / max(len(x) - 1, 1)


def select(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-score, kind="mergesort")[:max(1, min(int(k), len(score)))]


def proxy_f(ix: np.ndarray, y: np.ndarray, soft: bool) -> float:
    z = y / 3.0 if soft else (y >= 2.0).astype(float)
    s, m = float(z[ix].sum()), float(z.sum())
    return 2.0 * s / (len(ix) + m) if len(ix) + m else 1.0


def fit_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        scaler = StandardScaler().fit(x[tr])
        model = Ridge(alpha=10.0).fit(scaler.transform(x[tr]), y[tr])
        pred[te] = model.predict(scaler.transform(x[te]))
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = read_rows(args.labels)
    by: dict[str, list[dict]] = {}
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        by.setdefault(row["video_id"], []).append(row)
        if row["video_id"] not in cache:
            path = args.videos / f"{row['video_id']}.mp4"
            cache[row["video_id"]] = (visual_features(path), audio_features(path))
    x_v, x_a, y, groups, streams, times = [], [], [], [], [], []
    for vid in sorted(by):
        vf, af = cache[vid]
        for row in by[vid]:
            t = min(int(float(row["time_sec"])), 29)
            x_v.append(vf[t]); x_a.append(af[t]); y.append(GRADE[row["generic_highlight"]])
            groups.append(vid); streams.append(int(row["_stream"])); times.append(t)
    x_v = np.asarray(x_v, np.float32); x_a = np.asarray(x_a, np.float32)
    y = np.asarray(y, float); groups = np.asarray(groups); streams = np.asarray(streams); times = np.asarray(times)
    # The local visual arm is a fixed low-capacity control; audio is tested
    # alone and as an additive residual so a gain cannot be attributed to
    # merely changing the visual feature contract.
    pred = {
        "VISUAL": fit_oof(x_v, y, groups),
        "AUDIO": fit_oof(x_a, y, groups),
        "VISUAL_AUDIO": fit_oof(np.c_[x_v, x_a], y, groups),
    }
    deltas: dict[str, list[float]] = {k: [] for k in pred}
    binary: dict[str, list[float]] = {k: [] for k in pred}
    per_view: list[dict] = []
    for vid in sorted(by):
        iv = np.flatnonzero(groups == vid)
        for stream in sorted(set(streams[iv].tolist())):
            ix = iv[streams[iv] == stream]
            t, keep = np.unique(times[ix], return_index=True)
            ix = ix[keep]
            order = np.argsort(t)
            ix = ix[order]
            yy = y[ix]; n = len(ix); k = int(round(.85 * n))
            center = 1.0 - np.abs(np.arange(n) - (n - 1) / 2.0) / max((n - 1) / 2.0, 1.0)
            base = proxy_f(select(center, k), yy, True)
            rec = {"video_id": vid, "stream": int(stream), "n": n, "base_soft": base}
            for name, pp in pred.items():
                score = rank01(center) + .20 * (rank01(pp[ix][order]) - .5)
                val = proxy_f(select(score, k), yy, True)
                bval = proxy_f(select(score, k), yy, False)
                rec[f"{name}_delta_soft"] = val - base
                rec[f"{name}_binary"] = bval
                rec[f"{name}_delta_binary"] = bval - proxy_f(select(center, k), yy, False)
                deltas[name].append(val - base); binary[name].append(bval - proxy_f(select(center, k), yy, False))
            per_view.append(rec)
    result = {
        "experiment_id": "R53_AUDIO_RESIDUAL_PROBE",
        "status": "PUBLIC_SOURCE_HELD_OUT_COMPLETED",
        "hypothesis": "time-aligned cheap audio descriptors add held-out utility beyond visual local state",
        "rows": int(len(rows)), "sources": int(len(by)), "views": int(len(per_view)),
        "feature_contract": {"visual": "9 static + prev/next 1fps grayscale", "audio": "RMS/peak/ZCR/STFT bands/changes/missing mask", "eta": .20, "ridge_alpha": 10.0},
        "macro_delta_soft": {k: float(np.mean(v)) for k, v in deltas.items()},
        "positive_views_soft": {k: int(sum(x > 0 for x in v)) for k, v in deltas.items()},
        "macro_delta_binary": {k: float(np.mean(v)) for k, v in binary.items()},
        "positive_views_binary": {k: int(sum(x > 0 for x in v)) for k, v in binary.items()},
        "audio_present_sources": int(sum(np.max(np.abs(cache[v][1])) > 1e-8 for v in by)),
        "per_view": per_view,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("experiment_id", "macro_delta_soft", "positive_views_soft", "macro_delta_binary", "positive_views_binary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
