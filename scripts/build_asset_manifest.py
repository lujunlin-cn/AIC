#!/usr/bin/env python3
"""Create an atomic, resumable inventory of recoverable AIC assets.

The script never copies or deletes data and never follows directory symlinks.
Each input root is explicitly assigned one mode:

* ``--hash-root``: hash every regular file (weights, features, checkpoints,
  experiment metadata) and record size/mtime/SHA-256.
* ``--metadata-root``: record files, directories and symlink targets without
  reading file contents. Use this for received/evaluation datasets so no video
  bytes leave the server.

Outputs are written to ``.part`` then atomically renamed. A changing file is
retained in the manifest with ``status=changed_during_hash`` instead of being
silently treated as valid. Existing outputs are never overwritten unless
``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CHUNK = 8 * 1024 * 1024


def _sha256(path: Path) -> tuple[str, int, bool]:
    before = path.stat()
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
    after = path.stat()
    return digest.hexdigest(), total, (before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns)


def _match(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(Path(path).name, p) for p in patterns)


def _iter_entries(root: Path) -> Iterable[Path]:
    # os.walk does not follow directory symlinks when followlinks=False.
    if root.is_symlink():
        yield root
        return
    if root.is_file():
        yield root
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        base = Path(directory)
        for name in dirs:
            p = base / name
            if p.is_symlink():
                yield p
        for name in files:
            yield base / name


def _record(path: Path, root: Path, mode: str, excludes: list[str]) -> dict:
    rel = str(path.relative_to(root)) if path != root else "."
    common = {"root": str(root), "relative_path": rel, "path": str(path), "mode": mode}
    if _match(rel, excludes) or _match(str(path), excludes):
        return {**common, "kind": "excluded", "status": "excluded"}
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            return {**common, "kind": "symlink", "status": "metadata_only", "bytes": 0, "mtime_ns": info.st_mtime_ns, "target": os.readlink(path)}
        if not stat.S_ISREG(info.st_mode):
            return {**common, "kind": "special", "status": "metadata_only", "bytes": 0, "mtime_ns": info.st_mtime_ns}
        if mode == "metadata":
            return {**common, "kind": "file", "status": "metadata_only", "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        digest, read_bytes, stable = _sha256(path)
        return {**common, "kind": "file", "status": "hashed" if stable else "changed_during_hash", "bytes": read_bytes, "mtime_ns": info.st_mtime_ns, "sha256": digest}
    except FileNotFoundError:
        return {**common, "kind": "missing", "status": "missing", "bytes": 0}
    except OSError as exc:
        return {**common, "kind": "error", "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _roots(values: list[str], mode: str) -> list[tuple[Path, str]]:
    out = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f"input root does not exist: {value}")
        out.append((path, mode))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-root", action="append", default=[], help="root whose regular files are SHA-256 hashed")
    parser.add_argument("--metadata-root", action="append", default=[], help="root inventoried without reading contents")
    parser.add_argument("--exclude", action="append", default=[], help="fnmatch pattern against relative path or basename")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    roots = _roots(args.hash_root, "hash") + _roots(args.metadata_root, "metadata")
    if not roots:
        parser.error("at least one --hash-root or --metadata-root is required")
    output = args.output.expanduser().resolve()
    summary_path = (args.summary or output.with_suffix(".summary.json")).expanduser().resolve()
    if (output.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite {output} or {summary_path}; choose fresh paths or --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter(); bytes_by_mode = Counter(); roots_summary = defaultdict(lambda: Counter())
    temp = output.with_name(output.name + ".part")
    if temp.exists():
        temp.unlink()
    try:
        with temp.open("x", encoding="utf-8") as stream:
            for root, mode in roots:
                # The root itself is useful metadata, but file roots should be
                # emitted as one record and directories should enumerate below.
                entries = [root] if root.is_file() or root.is_symlink() else _iter_entries(root)
                for path in entries:
                    row = _record(path, root, mode, args.exclude)
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    counts[row["status"]] += 1
                    bytes_by_mode[mode] += int(row.get("bytes", 0) or 0)
                    roots_summary[str(root)][row["status"]] += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    summary = {
        "schema": "aic.asset_manifest.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "output": str(output),
        "hash_algorithm": "sha256",
        "hash_chunk_bytes": CHUNK,
        "roots": [{"path": str(p), "mode": m} for p, m in roots],
        "exclude_patterns": args.exclude,
        "records": sum(counts.values()),
        "status_counts": dict(sorted(counts.items())),
        "bytes_by_mode": dict(bytes_by_mode),
        "roots_summary": {k: dict(v) for k, v in sorted(roots_summary.items())},
        "warning": "metadata_only records are inventory evidence, not content integrity proofs; changed/error records must be rechecked.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": str(summary_path), "records": summary["records"], "status_counts": summary["status_counts"], "bytes_by_mode": summary["bytes_by_mode"]}, ensure_ascii=False))
    return 0 if not any(k in counts for k in ("error", "changed_during_hash", "missing")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
