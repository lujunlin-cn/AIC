"""YouTube acquisition queue with resume, bounded retries and a failure list.

Queue file (JSONL, append-only): one row per attempt
``{"yt_id", "dataset", "status", "attempt", "error", "path", "time"}``.
The latest row per id wins.  ``status`` in {queued, done, failed, gave_up}.
Permanent errors (private / removed / unavailable) stop retries at once.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .common import append_jsonl, now, read_jsonl

# "This video is unavailable" is checked against YouTube oEmbed (404 for all sampled ids, 2026-09-28)
PERMANENT = ("Video unavailable", "This video is unavailable", "Private video", "This video is not available",
             "has been removed",
             "account associated with this video has been terminated", "copyright", "members-only",
             "confirm your age", "Sign in to confirm your age", "does not exist")


def latest(queue: Path) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(queue):
        out[r["yt_id"]] = r
    return out


def reclassify(queue: Path) -> int:
    """Mark ``failed`` rows whose recorded error is permanent as ``gave_up`` (no new attempt)."""
    n = 0
    for r in latest(queue).values():
        err = (r.get("error") or "").lower()
        if r["status"] == "failed" and any(s.lower() in err for s in PERMANENT):
            append_jsonl(queue, {**r, "status": "gave_up", "permanent": True, "time": now(),
                                 "reclassified": True})
            n += 1
    return n


TRANSIENT = ("Network is unreachable", "timed out", "Connection reset", "Temporary failure in name resolution")


def reset_transient(queue: Path) -> int:
    """Re-queue ``failed`` ids whose last error was a local network problem (attempt counter reset)."""
    n = 0
    for r in latest(queue).values():
        err = r.get("error") or ""
        if r["status"] == "failed" and any(s in err for s in TRANSIENT):
            append_jsonl(queue, {**r, "status": "queued", "attempt": 0, "time": now(), "reset_reason": "transient"})
            n += 1
    return n


def enqueue(queue: Path, dataset: str, ids: Iterable[str], out_dir: Path) -> int:
    cur = latest(queue)
    n = 0
    for y in ids:
        if y not in cur:
            append_jsonl(queue, {"yt_id": y, "dataset": dataset, "status": "queued", "attempt": 0,
                                 "path": None, "out_dir": str(out_dir), "time": now()})
            n += 1
    return n


def run(queue: Path, max_attempts: int = 3, limit: int | None = None, proxy: str | None = None,
        fmt: str = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b",
        sleep_s: float = 2.0) -> dict:
    """Download queued ids sequentially (polite, single stream)."""
    cur = latest(queue)
    todo = [r for r in cur.values() if r["status"] in ("queued", "failed") and r["attempt"] < max_attempts]
    if limit:
        todo = todo[:limit]
    stats = {"done": 0, "failed": 0, "gave_up": 0}
    ytdlp = os.environ.get("AIC_YTDLP", "yt-dlp")
    for r in todo:
        out_dir = Path(r["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob(f"{r['yt_id']}.*"))
        existing = [p for p in existing if p.suffix in (".mp4", ".mkv", ".webm") and not p.name.endswith(".part")]
        if existing:
            append_jsonl(queue, {**r, "status": "done", "path": str(existing[0]), "time": now()})
            stats["done"] += 1
            continue
        cmd = [ytdlp, "--no-progress", "--no-playlist", "--continue", "--retries", "3",
               "--fragment-retries", "3", "-f", fmt, "--merge-output-format", "mp4",
               "--write-info-json", "-o", str(out_dir / "%(id)s.%(ext)s"),
               f"https://www.youtube.com/watch?v={r['yt_id']}"]
        if proxy:
            cmd[1:1] = ["--proxy", proxy]
        attempt = r["attempt"] + 1
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            err = (p.stderr or "")[-800:]
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            ok, err = False, "timeout 1800s"
        files = [q for q in out_dir.glob(f"{r['yt_id']}.*") if q.suffix in (".mp4", ".mkv", ".webm")]
        if ok and files:
            append_jsonl(queue, {**r, "status": "done", "attempt": attempt, "path": str(files[0]),
                                 "error": None, "time": now()})
            stats["done"] += 1
        else:
            permanent = any(s.lower() in err.lower() for s in PERMANENT)
            status = "gave_up" if permanent or attempt >= max_attempts else "failed"
            append_jsonl(queue, {**r, "status": status, "attempt": attempt, "error": err.strip()[-400:],
                                 "permanent": permanent, "time": now()})
            stats[status] += 1
        time.sleep(sleep_s)
    return stats
