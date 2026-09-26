"""Convert the public baseline's compact index into an audited local index."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .video import probe_video


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _assert_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("NaN/Infinity is not allowed in an input index")


def _strict_json(text: str) -> Any:
    value = json.loads(text, parse_constant=_reject_constant,
                       object_pairs_hook=_unique_object)
    _assert_finite(value)
    return value


def load_compact_index(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        lines = text.splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError("compact JSONL index must not contain blank lines")
        rows = [_strict_json(line) for line in lines]
    else:
        value = _strict_json(text)
        rows = value if isinstance(value, list) else [value]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("compact index must contain one object per sample")
    seen: set[str] = set()
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError(f"invalid or duplicate video_id: {video_id!r}")
        seen.add(video_id)
        ratio = row.get("targetRatioWH")
        if (not isinstance(ratio, list) or len(ratio) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(float(x)) or float(x) <= 0 for x in ratio)):
            raise ValueError(f"invalid targetRatioWH for {video_id}")
    return rows


def enrich_index(compact_index: str | Path, video_dir: str | Path,
                 output: str | Path, *, extension: str = ".mp4") -> dict[str, Any]:
    rows = load_compact_index(compact_index)
    root = Path(video_dir)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        video_id = row["video_id"]
        raw_path = next((row.get(key) for key in
                         ("video_path", "path", "video", "file_name", "filename")
                         if isinstance(row.get(key), str) and row.get(key)), None)
        path = Path(raw_path) if raw_path is not None else root / f"{video_id}{extension}"
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"missing video for {video_id}: {path}")
        info = probe_video(path)
        enriched.append({**row, "index_schema_version": "aic.input_index.v1",
                         "video_path": str(path.resolve()),
                         "width": info.width, "height": info.height,
                         "frame_count": info.frame_count, "fps": info.fps,
                         "duration": info.duration, "rotation": info.rotation,
                         "has_audio": info.has_audio, "time_base": info.time_base,
                         "coordinate_convention": info.coordinate_convention})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            for row in enriched:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"input": str(compact_index), "output": str(output), "videos": len(enriched),
            "coordinate_convention": "coded_pixels_no_autorotate"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-index", required=True,
                        help="JSON/JSONL with video_id, targetRatioWH, and optional video_path")
    parser.add_argument("--video-dir", required=True,
                        help="root for relative video_path or <video_id><extension>")
    parser.add_argument("--output", required=True, help="enriched JSONL output path")
    parser.add_argument("--extension", default=".mp4")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(enrich_index(args.compact_index, args.video_dir, args.output,
                                      extension=args.extension), ensure_ascii=False,
                         indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
