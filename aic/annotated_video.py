"""Explicit ordinal-frame reader for audited CFR public annotation benchmarks.

This is NOT a fallback for competition videos. Some SumMe mirror MP4s contain
broken container PTS, while annotations refer to decoded frame order and FPS.
Callers must verify both annotation count and annotation FPS before use.
"""
import math
from .video import letterbox_rgb

PROTOCOL = "ANNOTATED_CFR_DECODE_ORDINAL_V1"

def iter_annotated_cfr_frames(path, *, annotation_fps, annotation_count,
                              sample_fps=2.0, size=224):
    import av
    if not all(math.isfinite(x) and x > 0 for x in (annotation_fps, sample_fps)):
        raise ValueError("annotation/sample FPS must be finite and positive")
    if annotation_count <= 0:
        raise ValueError("annotation count must be positive")
    count = 0
    next_sample = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if not math.isclose(float(stream.average_rate), annotation_fps, rel_tol=1e-6):
            raise ValueError("container and annotation FPS disagree")
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if count > annotation_count:
                raise ValueError("decoded/annotation frame count mismatch")
            seconds = index / annotation_fps
            if seconds + 1e-9 < next_sample:
                continue
            yield index, seconds, letterbox_rgb(frame.to_ndarray(format="rgb24"), size)
            next_sample = (math.floor((seconds + 1e-9) * sample_fps) + 1) / sample_fps
    if count != annotation_count:
        raise ValueError("decoded/annotation frame count mismatch")
