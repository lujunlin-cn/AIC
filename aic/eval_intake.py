"""Strict, reproducible intake for a newly released evaluation video set.

This module only prepares an enriched metadata index. It never reads labels,
changes model thresholds, or creates predictions. Every source video is fully
decoded once so frame count and display PTS are checked before inference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .index import load_compact_index
from .video import _decoded, probe_video


INTAKE_VERSION = "AIC_EVAL_INTAKE_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                        prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _expected_int(row: Mapping[str, Any], key: str) -> int | None:
    if key not in row:
        return None
    value = row[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{row.get('video_id')}: supplied {key} must be a positive integer")
    return value


def prepare_eval_set(compact_index: str | Path, video_root: str | Path,
                     output_index: str | Path, output_manifest: str | Path,
                     *, extension: str = ".mp4", force: bool = False) -> dict[str, Any]:
    """Probe and hash every evaluation video, returning the written manifest.

    The compact index may provide a relative `video_path`, `path`, `video`,
    `file_name`, or `filename`; otherwise `<video_root>/<video_id><extension>`
    is used. Existing output is refused unless ``force=True``. Expected
    dimensions/frame_count supplied by the release index are checked against
    the actual decode instead of silently replaced.
    """
    compact = Path(compact_index)
    out_index, out_manifest = Path(output_index), Path(output_manifest)
    if not force and (out_index.exists() or out_manifest.exists()):
        raise FileExistsError("evaluation intake outputs already exist; use --force for a new version")
    root = Path(video_root).resolve()
    rows = load_compact_index(compact)
    enriched: list[dict[str, Any]] = []
    seen_paths: dict[Path, str] = {}
    for row in rows:
        video_id = row["video_id"]
        raw_path = next((row.get(key) for key in
                         ("video_path", "path", "video", "file_name", "filename")
                         if isinstance(row.get(key), str) and row.get(key)), None)
        path = Path(raw_path) if raw_path is not None else Path(f"{video_id}{extension}")
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation video for {video_id}: {path}")
        if path in seen_paths:
            raise ValueError(f"video path reused by {seen_paths[path]} and {video_id}: {path}")
        seen_paths[path] = video_id
        info = probe_video(path)
        # Do not retain one FrameStamp object per frame for long evaluation
        # videos. The decoder itself rejects missing/non-increasing display PTS.
        decoded_count = sum(1 for _ in _decoded(path))
        if decoded_count != info.frame_count:
            raise ValueError(f"{video_id}: timeline/probe frame count mismatch")
        for key, actual in (("width", info.width), ("height", info.height),
                            ("frame_count", info.frame_count)):
            expected = _expected_int(row, key)
            if expected is not None and expected != actual:
                raise ValueError(f"{video_id}: index {key}={expected} != decoded {actual}")
        enriched.append({**row, "index_schema_version": "aic.input_index.v1",
                         "video_path": str(path), "width": info.width,
                         "height": info.height, "frame_count": info.frame_count,
                         "fps": info.fps, "duration": info.duration,
                         "rotation": info.rotation, "has_audio": info.has_audio,
                         "time_base": info.time_base,
                         "coordinate_convention": info.coordinate_convention,
                         "source_sha256": _sha256(path),
                         "pts_verified": True, "intake_version": INTAKE_VERSION})
    index_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                          for row in enriched)
    _write_atomic(out_index, index_text)
    canonical = json.dumps(enriched, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    manifest = {"intake_version": INTAKE_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "compact_index": str(compact.resolve()),
                "compact_index_sha256": _sha256(compact),
                "video_root": str(root), "videos": len(enriched),
                "index_path": str(out_index.resolve()),
                "index_sha256": _sha256(out_index),
                "eval_set_id": hashlib.sha256(canonical.encode()).hexdigest(),
                "rows": [{"video_id": row["video_id"], "video_path": row["video_path"],
                          "source_sha256": row["source_sha256"],
                          "frame_count": row["frame_count"], "width": row["width"],
                          "height": row["height"], "targetRatioWH": row["targetRatioWH"]}
                         for row in enriched],
                "labels_present": False, "official_f_video": None,
                "competition_score": None}
    _write_atomic(out_manifest, json.dumps(manifest, ensure_ascii=False,
                                            indent=2, sort_keys=True) + "\n")
    return manifest
