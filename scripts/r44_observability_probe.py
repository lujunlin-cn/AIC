#!/usr/bin/env python3
"""Small source-held-out observability probe for the completed V1.3 labels.

The result is a mechanism screen, not an AIC score estimate.  It uses only the
12 public annotation sources, leaves one original source out at a time, and
keeps missing dependency labels masked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


GRADE = {"0普通": 0.0, "1有信息": 1.0, "2值得保留": 2.0, "3高光核心": 3.0}
POS = {"2值得保留", "3高光核心"}
PRIMARY = {"GtJXHRFhm0E_210.0_360.0", "bGU97Zw6Jfg_60.0_210.0", "eLlUVxVLiLU_60.0_210.0", "m6U-5Hck5z0_60.0_210.0"}


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open(newline="") as f:
            rows.extend(dict(r) for r in csv.DictReader(f))
    return rows


def image_features(path: Path) -> np.ndarray:
    imgs, feats = [], []
    for i in range(30):
        im = cv2.imread(str(path / f"{i:02d}.jpg"), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path / f"{i:02d}.jpg")
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge = cv2.Canny(im, 80, 160).mean() / 255.0
        feats.append([gray.mean() / 255.0, gray.std() / 255.0,
                      cv2.Laplacian(gray, cv2.CV_32F).var() / 10000.0,
                      edge, im[:, :, 0].mean() / 255.0,
                      im[:, :, 1].mean() / 255.0, im[:, :, 2].mean() / 255.0])
        imgs.append(gray)
    feats = np.asarray(feats, dtype=np.float32)
    prev = np.zeros(30, dtype=np.float32)
    nxt = np.zeros(30, dtype=np.float32)
    for i in range(30):
        if i:
            prev[i] = np.mean(np.abs(imgs[i] - imgs[i - 1])) / 255.0
        if i + 1 < 30:
            nxt[i] = np.mean(np.abs(imgs[i + 1] - imgs[i])) / 255.0
    local = np.stack([prev, nxt, np.abs(prev - nxt)], axis=1)
    return np.concatenate([feats, local], axis=1)


def audio_features(path: Path) -> np.ndarray:
    # Decode one second bins at 16 kHz.  Missing audio is represented by zeros
    # plus a separate availability feature, never as evidence of silence.
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        raw = b""
    x = np.frombuffer(raw, dtype=np.float32)
    out = np.zeros((30, 3), dtype=np.float32)
    if x.size == 0:
        return out
    for i in range(30):
        seg = x[i * 16000:(i + 1) * 16000]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-10))
        zcr = float(np.mean(np.signbit(seg[1:]) != np.signbit(seg[:-1]))) if seg.size > 1 else 0.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, 1 / 16000)
        centroid = float((spec * freqs).sum() / (spec.sum() + 1e-8)) / 8000.0
        out[i] = [math.log1p(rms * 100.0), zcr, centroid]
    return out


def fit_predict(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float = 10.0) -> tuple[np.ndarray, dict]:
    pred = np.full(len(y), np.nan, dtype=np.float64)
    source_metrics = {}
    for test_group in sorted(set(groups)):
        tr, te = groups != test_group, groups == test_group
        scaler = StandardScaler().fit(x[tr])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[tr]), y[tr])
        pred[te] = model.predict(scaler.transform(x[te]))
        source_metrics[test_group] = {
            "mae": float(np.mean(np.abs(pred[te] - y[te]))),
            "spearman": float(spearmanr(y[te], pred[te]).statistic) if len(np.unique(y[te])) > 1 else None,
            "n": int(te.sum()),
        }
    vals = [v["mae"] for v in source_metrics.values()]
    cors = [v["spearman"] for v in source_metrics.values() if v["spearman"] is not None]
    return pred, {"source": source_metrics, "macro_mae": float(np.mean(vals)), "macro_spearman": float(np.mean(cors)) if cors else None}


def run_target(name: str, mask: np.ndarray, y: np.ndarray, groups: np.ndarray, arms: dict[str, np.ndarray]) -> dict:
    out = {"n": int(mask.sum()), "arms": {}}
    for arm, x in arms.items():
        p, metrics = fit_predict(x[mask], y[mask], groups[mask])
        out["arms"][arm] = metrics
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = load_rows(args.labels)
    # For overlap sources, preserve both repeated labels; all other sources have
    # one row per frame.  This is an uncertainty diagnostic, not a consensus.
    by_source = {}
    feature_cache = {}
    audio_cache = {}
    for row in rows:
        vid = row["video_id"]
        if vid not in by_source:
            by_source[vid] = []
            feature_cache[vid] = image_features(args.frames / vid)
            audio_cache[vid] = audio_features(args.videos / f"{vid}.mp4")
        by_source[vid].append(row)
    all_x, all_a, grades, groups, phases, deps, ordered_rows = [], [], [], [], [], [], []
    for vid, group in by_source.items():
        for row in group:
            t = int(float(row["time_sec"]))
            pos = 1.0 - abs(t - 14.5) / 14.5
            all_x.append(np.r_[pos, feature_cache[vid][t]])
            all_a.append(audio_cache[vid][t])
            grades.append(GRADE[row["generic_highlight"]])
            groups.append(vid)
            phases.append(row["event_phase"])
            deps.append(row.get("needs_context", ""))
            ordered_rows.append(row)
    x = np.asarray(all_x, dtype=np.float32)
    aud = np.asarray(all_a, dtype=np.float32)
    y = np.asarray(grades, dtype=np.float32)
    groups = np.asarray(groups)
    # D0 position; D1 static descriptors; D2 adds local frame changes; D3 adds audio.
    arms = {
        "D0_CENTER": x[:, :1],
        "D1_STATIC": x[:, :8],
        "D2_LOCAL_CONTEXT": x[:, :11],
        "D3_AUDIO_GATE": np.concatenate([x[:, :11], aud], axis=1),
    }
    out = {
        "experiment_id": "R44_DIAG_001",
        "status": "COMPLETED_CPU",
        "sources": sorted(by_source),
        "n_rows": len(rows),
        "n_unique_sources": len(by_source),
        "targets": {},
        "feature_contract": "JPEG 1fps descriptors plus ffmpeg 16kHz one-second audio bins; source-held-out Ridge alpha=10",
    }
    out["targets"]["generic_grade"] = run_target("generic_grade", np.ones(len(y), dtype=bool), y, groups, arms)
    high = np.array([r["generic_highlight"] in POS for r in ordered_rows])
    ongoing_result = np.array([phase in {"进行中", "完成或结果", "结果后反应"} and hi for phase, hi in zip(phases, high)])
    y_phase = np.array([1.0 if phase == "进行中" else 0.0 for phase in phases], dtype=np.float32)
    phase_mask = high & np.array([phase in {"进行中", "完成或结果", "结果后反应"} for phase in phases])
    out["targets"]["highlight_ongoing"] = run_target("highlight_ongoing", phase_mask, y_phase, groups, arms)
    dep_mask = high & np.array([d in {"当前帧足够", "需要前后过程", "需要声音或语言"} for d in deps])
    y_dep = np.array([1.0 if d in {"需要前后过程", "需要声音或语言"} else 0.0 for d in deps], dtype=np.float32)
    out["targets"]["highlight_context_or_audio_dependency"] = run_target("dependency", dep_mask, y_dep, groups, arms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "sources": len(by_source)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
