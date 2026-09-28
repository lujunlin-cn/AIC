"""Media probing and the frame <-> time contract used by every adapter.

Conventions (identical to ``aic.video``):

* ``frame_index`` is the 0-based index of a displayed frame in presentation
  order, counting every decoded frame.
* ``time_s`` is ``pts * time_base`` minus the first displayed frame's PTS time,
  so the first frame is at 0.0 s.  Sampling picks real frames by this clock and
  never computes an index as ``round(t * fps)``.
* Pixel coordinates are coded pixels without auto-rotation; ``rotation`` is
  recorded so callers can decide.

A timeline file (``.npz``) stores ``pts`` (int64), ``time_s`` (float64) and the
time base, so annotation frame numbers can be mapped to real timestamps.
"""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np


def ffprobe(path: str | Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                          str(path)], check=True, capture_output=True, text=True, timeout=300)
    return json.loads(out.stdout)


def _video_stream(data: dict) -> dict:
    vids = [s for s in data.get("streams", []) if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic", 0)]
    if not vids:
        raise ValueError("no video stream")
    return vids[0]


def probe(path: str | Path) -> dict:
    """Container metadata only (fast, no decode)."""
    data = ffprobe(path)
    s = _video_stream(data)
    rotation = float(s.get("tags", {}).get("rotate", 0) or 0)
    for item in s.get("side_data_list", []) or []:
        if "rotation" in item:
            rotation = float(item["rotation"])
    fr = s.get("avg_frame_rate", "0/0")
    rfr = s.get("r_frame_rate", "0/0")
    fps = float(Fraction(fr)) if fr not in ("0/0", "") else None
    duration = s.get("duration") or data.get("format", {}).get("duration")
    nb = s.get("nb_frames")
    return {
        "width": int(s["width"]), "height": int(s["height"]), "codec": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"), "fps_avg": fps,
        "fps_r": float(Fraction(rfr)) if rfr not in ("0/0", "") else None,
        "time_base": s.get("time_base"), "rotation": rotation,
        "container_duration": float(duration) if duration not in (None, "N/A") else None,
        "container_nb_frames": int(nb) if nb not in (None, "N/A") else None,
        "has_audio": any(x.get("codec_type") == "audio" for x in data.get("streams", [])),
        "format": data.get("format", {}).get("format_name"),
        "sar": s.get("sample_aspect_ratio"),
    }


def timeline(path: str | Path, decode: bool = True) -> dict:
    """Real display timeline.

    ``decode=True`` decodes every frame (exact displayed-frame count).
    ``decode=False`` demuxes packets only; faster but may disagree with the
    decoder for streams with empty/dropped packets, so it is labelled.
    """
    import av

    pts: list[int] = []
    with av.open(str(path)) as c:
        st = next(s for s in c.streams.video
                  if not (s.disposition & av.stream.Disposition.attached_pic))
        tb = st.time_base
        if decode:
            st.thread_type = "AUTO"
            st.codec_context.thread_count = 2
            missing = 0
            for fr in c.decode(st):
                if fr.pts is None:
                    missing += 1
                    pts.append(pts[-1] + 1 if pts else 0)
                else:
                    pts.append(int(fr.pts))
        else:
            missing = 0
            for pkt in c.demux(st):
                if pkt.size and pkt.pts is not None:
                    pts.append(int(pkt.pts))
                elif pkt.size:
                    missing += 1
            pts.sort()
    arr = np.asarray(pts, dtype=np.int64)
    if arr.size == 0:
        raise ValueError(f"no frames: {path}")
    tsec = (arr - arr[0]).astype(np.float64) * float(tb)
    diffs = np.diff(tsec)
    return {
        "pts": arr, "time_s": tsec, "time_base": f"{tb.numerator}/{tb.denominator}",
        "method": "decode" if decode else "demux_packets",
        "frame_count": int(arr.size), "missing_pts": int(missing),
        "duration_s": float(tsec[-1] + (np.median(diffs) if diffs.size else 0.0)),
        "fps_measured": float(1.0 / np.median(diffs)) if diffs.size and np.median(diffs) > 0 else None,
        "non_increasing": int(np.sum(diffs <= 0)) if diffs.size else 0,
        "max_gap_s": float(diffs.max()) if diffs.size else 0.0,
    }


def save_timeline(tl: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, pts=tl["pts"], time_s=tl["time_s"], time_base=np.array(tl["time_base"]),
                        method=np.array(tl["method"]))
    tmp.replace(path)


def load_timeline(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"pts": z["pts"], "time_s": z["time_s"], "time_base": str(z["time_base"]),
            "method": str(z["method"])}


def frames_at_times(time_s: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """First real frame at/after each requested time (same rule as aic.video)."""
    idx = np.searchsorted(time_s, np.asarray(sample_times) - 1e-9, side="left")
    return np.clip(idx, 0, len(time_s) - 1)


def uniform_sample(time_s: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample at ``fps`` on the real clock; returns (frame_indices, their times)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    grid = np.arange(0.0, float(time_s[-1]) + 1e-9, 1.0 / fps)
    idx = np.unique(frames_at_times(time_s, grid))
    return idx, time_s[idx]


def read_frames(path: str | Path, indices, size: Optional[int] = None) -> dict[int, np.ndarray]:
    """Decode selected display-order frames (RGB uint8). Sequential decode."""
    import av

    want = sorted(set(int(i) for i in indices))
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    ptr = 0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i == want[ptr]:
                img = fr.to_ndarray(format="rgb24")
                if size:
                    import cv2
                    h, w = img.shape[:2]
                    s = size / max(h, w)
                    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                                     interpolation=cv2.INTER_AREA)
                out[i] = img
                ptr += 1
                if ptr == len(want):
                    break
    return out
