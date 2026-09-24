"""Convert the public baseline's compact index into an audited local index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .video import probe_video


def load_compact_index(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
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
        if not isinstance(ratio, list) or len(ratio) != 2 or any(float(x) <= 0 for x in ratio):
            raise ValueError(f"invalid targetRatioWH for {video_id}")
    return rows


def enrich_index(compact_index: str | Path, video_dir: str | Path,
                 output: str | Path, *, extension: str = ".mp4") -> dict[str, Any]:
    rows = load_compact_index(compact_index)
    root = Path(video_dir)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        video_id = row["video_id"]
        raw_path = row.get("video_path")
        path = Path(raw_path) if isinstance(raw_path, str) else root / f"{video_id}{extension}"
        if not path.is_file():
            raise FileNotFoundError(f"missing video for {video_id}: {path}")
        info = probe_video(path)
        enriched.append({**row, "video_path": str(path.resolve()),
                         "width": info.width, "height": info.height,
                         "frame_count": info.frame_count, "fps": info.fps,
                         "duration": info.duration, "rotation": info.rotation,
                         "has_audio": info.has_audio, "time_base": info.time_base,
                         "coordinate_convention": info.coordinate_convention})
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in enriched:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"input": str(compact_index), "output": str(output), "videos": len(enriched),
            "coordinate_convention": "coded_pixels_no_autorotate"}
