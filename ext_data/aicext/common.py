"""Paths, append-only JSONL registries, hashing and logging helpers."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

EXT_ROOT = Path(os.environ.get("AIC_EXT_ROOT", "/data/aic/external_datasets"))
REGISTRY_DIR = EXT_ROOT / "_registry"
# Every dataset keeps the same five sub-directories; existing data is referenced
# by path from the index instead of being copied.
DATASET_SUBDIRS = ("raw", "annotations", "processed", "splits", "logs")
# Keep this much free space on the data disk for running experiments.
MIN_FREE_BYTES = int(float(os.environ.get("AIC_EXT_MIN_FREE_GB", "700")) * 1e9)


def now() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).isoformat(timespec="seconds")


def dataset_dir(name: str, create: bool = True) -> Path:
    root = EXT_ROOT / name
    if create:
        for sub in DATASET_SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def free_bytes(path: Path = EXT_ROOT) -> int:
    return shutil.disk_usage(path).free


def ensure_space(needed: int, path: Path = EXT_ROOT, reserve: int | None = None) -> None:
    """Refuse a download/extract whose peak usage would eat the reserve."""
    reserve = MIN_FREE_BYTES if reserve is None else reserve
    free = free_bytes(path)
    if free - needed < reserve:
        raise RuntimeError(
            f"insufficient space: need {needed/1e9:.1f} GB, free {free/1e9:.1f} GB, "
            f"reserve {reserve/1e9:.1f} GB")


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _clean(value: Any) -> Any:
    """JSON-safe values: NaN/Inf become None so readers never see bare NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        return _clean(value.item())
    return value


def dumps(row: dict) -> str:
    return json.dumps(_clean(row), ensure_ascii=False, sort_keys=False)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Atomic rewrite of a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps(row) + "\n")
            n += 1
    tmp.replace(path)
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(dumps(row) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def log(dataset: str, event: str, **fields: Any) -> None:
    """Structured event log per dataset plus a global stream."""
    row = {"time": now(), "dataset": dataset, "event": event, **fields}
    append_jsonl(EXT_ROOT / dataset / "logs" / "events.jsonl", row)
    append_jsonl(EXT_ROOT / "_logs" / "events.jsonl", row)
    print(dumps(row), flush=True)
