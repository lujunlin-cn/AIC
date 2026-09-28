"""Temporal supervision readers.

Label alignment contract (fixes the ~1 s anchor/bin offset seen earlier):

* A label defined on an interval ``[t0, t1)`` of source time is assigned to a
  sampled frame iff the frame's *real* display time ``t`` satisfies
  ``t0 <= t < t1``.  We never map a frame to ``floor(t / bin)`` and then treat
  the bin start as the frame time.
* Per-step scores on a regular grid are attached at ``anchor + k * step``
  where ``anchor`` is recorded per annotation (``coverage.grid_anchor_s``).
  Mr.HiSum ``gtscore[k]`` is the Most-Replayed intensity of the heat-marker
  bin containing instant ``k`` s (upstream ``floor(k*1000/chunk_ms)``), so its
  anchor is 0.0, not the cell centre.  Values are linearly interpolated onto
  real frame times; frames outside the first/last anchor are ``edge=True``.
* Segment labels keep their original meaning. Positions not covered by any
  labelled segment have ``valid=False`` — they are *not* negatives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .base import Index, media_times, sample_indices


def interval_label(times: np.ndarray, segments: list[tuple[float, float, float]],
                   reduce: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """segments: (t0, t1, value). Returns (value per time, valid mask)."""
    val = np.full(len(times), np.nan, np.float32)
    for t0, t1, v in segments:
        sel = (times >= t0) & (times < t1)
        if reduce == "max":
            val[sel] = np.where(np.isnan(val[sel]), v, np.maximum(val[sel], v))
        else:  # overlapping segments averaged
            val[sel] = np.where(np.isnan(val[sel]), v, (val[sel] + v) / 2)
    valid = ~np.isnan(val)
    return np.nan_to_num(val, nan=0.0), valid


def grid_scores(times: np.ndarray, scores: np.ndarray, step_s: float,
                anchor_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    centers = anchor_s + np.arange(len(scores)) * step_s
    val = np.interp(times, centers, scores).astype(np.float32)
    edge = (times < centers[0]) | (times > centers[-1])
    return val, edge


class SegmentLabelAdapter:
    """YouTube Highlights / PHD2 / QVH style labelled segments on real times.

    ``fields``: which label field to read from each segment record
    (e.g. ``mturk_score`` or ``match_label``).  Returns per-sample values,
    ``valid`` (covered by a labelled segment) and the raw segments.
    """

    def __init__(self, dataset: str, annotation_type: str = "temporal_segment_label",
                 split=None, field: str = "value", fps: Optional[float] = 2.0,
                 max_frames: Optional[int] = None, source: Optional[str] = None,
                 variant: Optional[str] = None, exclude_flags: Iterable[str] = ("misaligned_suspect",)):
        self.ix = Index.load(dataset)
        excl = set(exclude_flags)

        def where(m, a):
            if source and a["annotation_source"] != source:
                return False
            if variant and a.get("variant") != variant:
                return False
            return not (excl & set(m.get("anomalies") or []))

        self.items = self.ix.select(annotation_type, split=split, where=where)
        self.field, self.fps, self.max_frames = field, fps, max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        m, a = self.items[i]
        d = json.loads(Path(a["annotation_path"]).read_text())
        segs = d["segments"]
        idx, t = sample_indices(m, self.fps, self.max_frames)
        end = float(media_times(m)[-1]) if m.get("timeline_path") else float("inf")
        # segments starting after the media ends cannot be aligned; drop rather than guess
        tri = [(s["t0"], s["t1"], s[self.field]) for s in segs
               if s.get(self.field) is not None and s["t0"] <= end]
        val, valid = interval_label(t, tri)
        return {"uid": m["uid"], "group_id": m["group_id"], "annotation_uid": a["uid"], "frame_idx": idx,
                "time_s": t, "value": val, "valid": valid, "segments": segs, "field": self.field,
                "source": a["annotation_source"]}


class GridScoreAdapter:
    """Mr.HiSum ``gtscore`` (1 value per second): features and/or real-time labels.

    ``mode='features'`` returns the pre-extracted feature matrix aligned to the
    score grid (no media needed).  ``mode='video'`` maps scores onto real frame
    times of a downloaded source video.
    """

    def __init__(self, dataset: str = "MrHiSum", split=None, mode: str = "features",
                 fps: Optional[float] = 1.0, max_steps: Optional[int] = None, with_audio: bool = False):
        self.ix = Index.load(dataset)
        self.mode = mode
        if mode == "features":
            where = lambda m, a: bool((a.get("coverage") or {}).get("feature_path"))  # noqa: E731
        else:
            where = lambda m, a: bool(m.get("media_path")) and m.get("media_kind") == "video_file"  # noqa: E731
        self.items = self.ix.select("temporal_score_1d", split=split, ready_only=(mode == "video"), where=where)
        self.fps, self.max_steps, self.with_audio = fps, max_steps, with_audio
        self._files: dict[str, object] = {}

    def __len__(self):
        return len(self.items)

    def _file(self, path):
        import h5py

        if path not in self._files:
            if len(self._files) > 32:
                for f in self._files.values():
                    f.close()
                self._files.clear()
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def __getitem__(self, i: int) -> dict:
        m, a = self.items[i]
        cov = a["coverage"]
        h = self._file(a["annotation_path"])[cov["h5_key"]]
        score = np.asarray(h["gtscore"], np.float32)
        step, anchor = cov["step_s"], cov.get("grid_anchor_s", 0.0)
        out = {"uid": m["uid"], "group_id": m["group_id"], "gtscore": score, "step_s": step}
        if self.mode == "features":
            g = self._file(cov["feature_path"])[cov["h5_key"]]
            # YouTube-8M quantized bytes -> float, same dequantization as upstream Mr.HiSum
            feats = np.asarray(g["rgb"], np.float32) * (4.0 / 255.0) - 2.0
            if self.with_audio:
                aud = np.asarray(g["audio"], np.float32) * (4.0 / 255.0) - 2.0
                feats = np.concatenate([feats, aud[:len(feats)]], 1)
            n = max(len(feats), len(score))
            valid = np.zeros(n, bool)
            valid[:min(len(feats), len(score))] = True  # both a feature row and a label exist
            f_pad = np.zeros((n, feats.shape[1]), np.float32)
            f_pad[:len(feats)] = feats
            s_pad = np.zeros(n, np.float32)
            s_pad[:len(score)] = score
            out.update(features=f_pad, gtscore=s_pad, valid=valid,
                       grid_time_s=anchor + np.arange(n) * step)
            if self.max_steps and n > self.max_steps:
                for k in ("features", "gtscore", "valid", "grid_time_s"):
                    out[k] = out[k][:self.max_steps]
        else:
            idx, t = sample_indices(m, self.fps)
            val, edge = grid_scores(t, score, step, anchor)
            # frames past the last score cell (video longer than the label grid) are unlabelled
            covered = t < len(score) * cov["step_s"]
            out.update(frame_idx=idx, time_s=t, value=val, valid=covered, edge=edge)
        return out


class ShotBoundaryAdapter:
    """ClipShots transitions: per-frame labels with explicit ignore regions.

    Label codes per frame: 0 = no transition (only where the subset lists all
    transitions), 1 = hard cut (``e - s == 1``: frames s and s+1), 2 = gradual
    (``e - s > 1``: frames s..e) — same split as upstream ``tools/evaluate.py``.
    Degenerate (``e == s``) / malformed rows are ignore regions (valid=False).
    For ``only_gradual`` negatives are unreliable: every frame outside a
    labelled transition is ``valid=False``.  Labels cover
    ``min(frame_num, decoded frames)``; ``label_frames`` reports it.
    """

    def __init__(self, dataset: str = "ClipShots", split=None, subset: Optional[str] = None,
                 window: Optional[int] = 64, seed: int = 0, ready_only: bool = True):
        self.ix = Index.load(dataset)
        where = (lambda m, a: a["variant"] == subset) if subset else None
        self.items = self.ix.select("shot_boundary", split=split, where=where, ready_only=ready_only)
        self.window = window
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        m, a = self.items[i]
        d = json.loads(Path(a["annotation_path"]).read_text())
        n = int(d["frame_num"])
        t = media_times(m) if m.get("timeline_path") else None
        if t is not None:
            n = min(n, len(t))
        lab = np.zeros(n, np.int64)
        valid = np.full(n, bool(d["negatives_reliable"]))
        for (s, e), kind in zip(d["transitions"], d.get("kinds") or [None] * len(d["transitions"])):
            s, e = int(s), int(e)
            kind = kind or ("malformed" if e < s else "degenerate" if e == s else "cut" if e - s == 1 else "gradual")
            lo, hi = max(0, min(s, e)), min(n, max(s, e) + 1)
            if lo >= hi:
                continue
            if kind in ("cut", "gradual"):
                lab[lo:hi] = 1 if kind == "cut" else 2
                valid[lo:hi] = True
            else:
                lab[lo:hi] = 0
                valid[lo:hi] = False
        idx = np.arange(n)
        if self.window and n > self.window:
            s0 = int(self.rng.integers(0, n - self.window + 1))
            idx = idx[s0:s0 + self.window]
        out = {"uid": m["uid"], "group_id": m["group_id"], "frame_idx": idx, "label": lab[idx],
               "valid": valid[idx], "subset": d["subset"], "negatives_reliable": d["negatives_reliable"],
               "transitions": d["transitions"], "label_frames": n}
        out["time_s"] = t[idx] if t is not None else None
        return out


class UserSelectionAdapter:
    """PHD2 personal selections on real frame times (one sample per video x user).

    ``selected`` is True where this user's GIF interval covers the frame; ``valid``
    equals ``selected`` (only positives are known).  ``others_selected`` counts how
    many *other* users in the same csv selected the frame — a separate, weaker
    signal that callers may use explicitly; it never turns into a negative.
    """

    def __init__(self, dataset: str = "PHD2", split=None, csv_split: Optional[str] = None,
                 fps: Optional[float] = 2.0, is_last_only: bool = False,
                 exclude_flags: Iterable[str] = ("duration_mismatch",)):
        self.ix = Index.load(dataset)
        excl = set(exclude_flags)

        def where(m, a):
            if csv_split and a["variant"] != csv_split:
                return False
            return not (excl & set(m.get("anomalies") or []))

        pairs = self.ix.select("temporal_user_selection", split=split, where=where)
        self._files: dict[str, dict] = {}
        self.items = []
        for m, a in pairs:
            users = self._load(a["annotation_path"])[a["coverage"]["key"]]
            for u, segs in users.items():
                if is_last_only and not any(s["is_last"] for s in segs):
                    continue
                self.items.append((m, a, u))
        self.fps = fps

    def _load(self, path):
        if path not in self._files:
            self._files[path] = json.loads(Path(path).read_text())
        return self._files[path]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        m, a, user = self.items[i]
        users = self._load(a["annotation_path"])[a["coverage"]["key"]]
        idx, t = sample_indices(m, self.fps)
        end = float(t[-1]) if len(t) else 0.0
        sel = np.zeros(len(t), bool)
        for s in users[user]:
            if s["t0"] <= end:
                sel |= (t >= s["t0"]) & (t < s["t1"])
        others = np.zeros(len(t), np.int32)
        for u, segs in users.items():
            if u == user:
                continue
            hit = np.zeros(len(t), bool)
            for s in segs:
                hit |= (t >= s["t0"]) & (t < s["t1"])
            others += hit
        return {"uid": m["uid"], "group_id": m["group_id"], "user_id": user, "frame_idx": idx, "time_s": t,
                "selected": sel, "valid": sel.copy(), "others_selected": others,
                "is_last": any(s["is_last"] for s in users[user]), "segments": users[user]}
