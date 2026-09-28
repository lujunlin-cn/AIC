"""Common adapter machinery.

Every adapter reads the unified index (``processed/media.jsonl`` +
``processed/annotations.jsonl`` of one dataset), filters by split / annotation
type, and yields plain-Python/numpy samples.  ``torch_dataset()`` wraps any
adapter for a DataLoader.  Unlabelled positions are always expressed through a
``valid`` mask; missing labels are never filled with 0 silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from ..common import EXT_ROOT, read_jsonl
from ..media import load_timeline, read_frames, uniform_sample


@dataclass
class Index:
    dataset: str
    media: dict[str, dict] = field(default_factory=dict)
    annotations: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, dataset: str, root: Path = EXT_ROOT) -> "Index":
        d = root / dataset / "processed"
        media = {m["uid"]: m for m in read_jsonl(d / "media.jsonl")}
        return cls(dataset, media, list(read_jsonl(d / "annotations.jsonl")))

    def select(self, annotation_type: str | Iterable[str], split: Optional[str | Iterable[str]] = None,
               split_field: str = "aic_split", ready_only: bool = True,
               where: Optional[Callable[[dict, dict], bool]] = None) -> list[tuple[dict, dict]]:
        types = {annotation_type} if isinstance(annotation_type, str) else set(annotation_type)
        splits = None if split is None else ({split} if isinstance(split, str) else set(split))
        out = []
        for a in self.annotations:
            if a["annotation_type"] not in types:
                continue
            m = self.media.get(a["media_uid"])
            if m is None:
                continue
            if ready_only and (m.get("media_kind") not in ("feature_only",)) and \
                    m.get("download_status") not in ("verified", "downloaded", "referenced_existing",
                                                      "sample_only"):
                continue
            if splits is not None and m.get(split_field) not in splits:
                continue
            if where and not where(m, a):
                continue
            out.append((m, a))
        return out


def media_times(m: dict) -> np.ndarray:
    """Real per-frame display times (s) from the stored timeline."""
    if m.get("timeline_path"):
        return load_timeline(m["timeline_path"])["time_s"]
    if m.get("fps") and m.get("frame_count"):
        # Frame folders without container timestamps: nominal CFR clock, labelled as such.
        return np.arange(int(m["frame_count"]), dtype=np.float64) / float(m["fps"])
    raise ValueError(f"no timeline for {m['uid']}")


def sample_indices(m: dict, fps: Optional[float], max_frames: Optional[int] = None,
                   rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices and their real times; fps=None means every frame."""
    t = media_times(m)
    if fps is None:
        idx = np.arange(len(t))
    else:
        idx, _ = uniform_sample(t, fps)
    if max_frames and len(idx) > max_frames:
        start = 0 if rng is None else int(rng.integers(0, len(idx) - max_frames + 1))
        idx = idx[start:start + max_frames]
    return idx, t[idx]


def load_video_frames(m: dict, idx: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """Decode frames by display index for video files, or read image files for frame folders."""
    if m["media_kind"] == "video_file":
        got = read_frames(m["media_path"], idx, size=size)
        missing = [int(i) for i in idx if int(i) not in got]
        if missing:
            raise IndexError(f"{m['uid']}: frames not decodable {missing[:5]}")
        return np.stack([got[int(i)] for i in idx])
    if m["media_kind"] in ("frame_folder", "image"):
        import cv2

        files = frame_files(m)
        imgs = []
        for i in idx:
            img = cv2.cvtColor(cv2.imread(str(files[int(i)]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if size:
                h, w = img.shape[:2]
                s = size / max(h, w)
                img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            imgs.append(img)
        return np.stack(imgs)
    raise ValueError(f"cannot decode media kind {m['media_kind']}")


_FRAME_CACHE: dict[str, list[Path]] = {}


def frame_files(m: dict) -> list[Path]:
    key = m["media_path"]
    if key not in _FRAME_CACHE:
        p = Path(key)
        if m["media_kind"] == "image":
            _FRAME_CACHE[key] = [p]
        else:
            pat = m.get("frame_file_pattern") or "*.jpg"
            _FRAME_CACHE[key] = sorted(p.glob(pat))
    return _FRAME_CACHE[key]


def json_field(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v


class TorchDataset:
    """Minimal map-style dataset wrapper (torch imported lazily)."""

    def __init__(self, adapter):
        self.adapter = adapter

    def __len__(self):
        return len(self.adapter)

    def __getitem__(self, i):
        return self.adapter[i]


def collate_pad(batch: list[dict], pad_keys: Iterable[str] = ()) -> dict:
    """Pad every axis of variable-shape arrays to the batch max; adds ``<key>_shape``.

    Padding is 0 / False, so any ``valid`` mask is False on padded positions and
    padded labels are never read as real zeros.
    """
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    pad_keys = set(pad_keys)
    for k in keys:
        vals = [b[k] for b in batch]
        if k in pad_keys:
            arrs = [np.asarray(v) for v in vals]
            shape = tuple(max(a.shape[d] for a in arrs) for d in range(arrs[0].ndim))
            fill = False if arrs[0].dtype == bool else 0
            padded = np.full((len(arrs), *shape), fill, dtype=arrs[0].dtype)
            for i, a in enumerate(arrs):
                padded[(i, *[slice(0, s) for s in a.shape])] = a
            out[k] = padded
            out[k + "_shape"] = np.asarray([a.shape for a in arrs])
        else:
            out[k] = vals
    return out
