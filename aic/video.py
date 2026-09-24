"""Original decoder indices and PTS, with an explicit coded-pixel convention.

No FPS multiplication or frame-seeking is used for submission indices. Rotation
metadata is retained; pixels are not autorotated until the official convention
is known. Times passed to the model are relative to the first displayed PTS.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    frame_count: int
    fps: float
    time_base: str
    rotation: float
    has_audio: bool
    duration: float
    start_time: float
    coordinate_convention: str = "coded_pixels_no_autorotate"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FrameStamp:
    index: int
    pts: int
    time_seconds: float


def probe_video(path: str | Path) -> VideoInfo:
    """Count actual decoded frames instead of trusting container nb_frames."""
    from fractions import Fraction

    path = Path(path)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams",
         "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    videos = [s for s in data["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic", 0)]
    if len(videos) != 1:
        raise ValueError(f"Expected exactly one video stream: {path}")
    stream = videos[0]
    count = stream.get("nb_read_frames")
    if count in (None, "N/A") or int(count) <= 0:
        raise ValueError(f"No decodable frames: {path}")
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for item in stream.get("side_data_list", []):
        rotation = float(item.get("rotation", rotation))
    raw_fps = stream.get("avg_frame_rate", "0/1")
    fps = float(Fraction(raw_fps)) if raw_fps != "0/0" else 0.0
    duration = float(stream.get("duration", data.get("format", {}).get("duration", 0)))
    return VideoInfo(
        str(path.resolve()), int(stream["width"]), int(stream["height"]),
        int(count), fps, stream["time_base"], rotation,
        any(s["codec_type"] == "audio" for s in data["streams"]),
        duration, float(stream.get("start_time", 0)),
    )


def _decoded(path):
    import av

    with av.open(str(path)) as container:
        stream = next(s for s in container.streams.video
                      if not (s.disposition & av.stream.Disposition.attached_pic))
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        first_time = None
        previous = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"Missing PTS at original frame {index}: {path}")
            absolute = float(frame.pts * frame.time_base)
            if previous is not None and absolute <= previous:
                raise ValueError(f"Non-increasing display PTS at frame {index}: {path}")
            if first_time is None:
                first_time = absolute
            previous = absolute
            yield FrameStamp(index, frame.pts, absolute - first_time), frame


def frame_timeline(path: str | Path) -> list[FrameStamp]:
    return [stamp for stamp, _ in _decoded(path)]


def letterbox_rgb(rgb, size: int = 224):
    """Full image letterbox; ImageNet-normalized later by the feature encoder."""
    import numpy as np
    from PIL import Image

    if size <= 0:
        raise ValueError("size must be positive")
    height, width = rgb.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = Image.fromarray(rgb).resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(resized, ((size - resized_w) // 2, (size - resized_h) // 2))
    return np.asarray(canvas)


def iter_sampled_frames(path: str | Path, sample_fps: float = 2.0,
                        size: int = 224) -> Iterator[tuple[int, float, object]]:
    """First actual frame at/after each sample time, without duplicated frames."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    next_sample = 0.0
    for stamp, frame in _decoded(path):
        if stamp.time_seconds + 1e-9 < next_sample:
            continue
        yield stamp.index, stamp.time_seconds, letterbox_rgb(frame.to_ndarray(format="rgb24"), size)
        next_sample = (math.floor((stamp.time_seconds + 1e-9) * sample_fps) + 1) / sample_fps


def expand_scores(timeline: list[FrameStamp], sampled_times, scores, threshold: float):
    """Linear probability interpolation on real PTS, selecting all original frames.

    This is a baseline interpolation policy, not a claim of precise boundaries.
    """
    import numpy as np

    times, values = np.asarray(sampled_times), np.asarray(scores)
    if not timeline or times.ndim != 1 or values.shape != times.shape or len(times) == 0:
        raise ValueError("Nonempty matching 1D time and score arrays are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite time or score")
    if np.any(np.diff(times) <= 0) or not math.isfinite(threshold):
        raise ValueError("Times must increase and threshold must be finite")
    dense = np.interp([s.time_seconds for s in timeline], times, values)
    return [s.index for s, value in zip(timeline, dense) if value >= threshold]
