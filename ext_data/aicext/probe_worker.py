"""Parallel, resumable media probing: ffprobe + full-decode timeline per video.

Results are cached in ``processed/probe_cache.jsonl`` keyed by (path, size,
mtime) so reruns only probe new or changed files.  Timelines are written to
``processed/timelines/<uid>.npz``.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .common import append_jsonl, read_jsonl
from .media import probe, save_timeline, timeline


def _key(path: str) -> str:
    st = os.stat(path)
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def _probe_one(uid: str, path: str, tl_path: str, decode: bool) -> dict:
    out = {"uid": uid, "path": path, "key": _key(path)}
    try:
        meta = probe(path)
        out.update(meta)
        tl = timeline(path, decode=decode)
        save_timeline(tl, tl_path)
        out.update({k: tl[k] for k in ("frame_count", "missing_pts", "duration_s", "fps_measured",
                                       "non_increasing", "max_gap_s", "method", "time_base")})
        out["timeline_path"] = tl_path
        out["ok"] = True
    except Exception as exc:
        out["ok"] = False
        out["error"] = repr(exc)[:500]
    return out


def probe_all(items: list[tuple[str, str]], dataset_root: Path, workers: int = 4,
              decode: bool = True) -> dict[str, dict]:
    """items: (uid, media_path). Returns uid -> probe result."""
    cache_path = dataset_root / "processed" / "probe_cache.jsonl"
    cache = {}
    for row in read_jsonl(cache_path):
        cache[row["uid"]] = row
    results, todo = {}, []
    for uid, path in items:
        if not path or not os.path.exists(path):
            results[uid] = {"uid": uid, "ok": False, "error": "missing file"}
            continue
        c = cache.get(uid)
        if c and c.get("ok") and c.get("key") == _key(path) and os.path.exists(c.get("timeline_path", "")):
            results[uid] = c
        else:
            todo.append((uid, path))
    tl_dir = dataset_root / "processed" / "timelines"
    tl_dir.mkdir(parents=True, exist_ok=True)
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_probe_one, uid, path, str(tl_dir / f"{uid.replace(':', '_').replace('/', '_')}.npz"),
                              decode) for uid, path in todo]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                results[r["uid"]] = r
                append_jsonl(cache_path, r)
                if i % 50 == 0:
                    print(f"probed {i}/{len(todo)}", flush=True)
    return results
