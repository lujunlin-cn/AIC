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


TRANSIENT = ("Network is unreachable", "timed out", "Connection reset", "Temporary failure in name resolution",
             "proxy", "getaddrinfo")


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


def proxy_alive(proxy: str, timeout: float = 6.0) -> bool:
    """True when the reverse-tunnel proxy can actually reach YouTube right now."""
    try:
        import requests
        r = requests.head("https://www.youtube.com/generate_204",
                          proxies={"http": proxy, "https": proxy}, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def run(queue: Path, max_attempts: int = 3, limit: int | None = None, proxy: str | None = None,
        fmt: str = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b",
        sleep_s: float = 2.0, order: list[str] | None = None, budget_bytes: int | None = None,
        progress_every: int = 25) -> dict:
    """Download queued ids sequentially (polite, single stream).

    ``order`` restricts and reorders the work list (a curated subset; ids already
    done or permanently given up are skipped regardless).  ``budget_bytes`` stops
    the run once the media bytes on disk reach it — counted across sessions, so a
    stopped run resumes where it left off.
    """
    cur = latest(queue)
    todo = [r for r in cur.values() if r["status"] in ("queued", "failed") and r["attempt"] < max_attempts]
    if order is not None:
        by_id = {r["yt_id"]: r for r in todo}
        todo = [by_id[i] for i in order if i in by_id]
    if limit:
        todo = todo[:limit]
    stats = {"done": 0, "failed": 0, "gave_up": 0, "budget_stop": False}
    spent = 0.0
    for r in cur.values():   # media already on disk from earlier sessions counts toward the budget
        if r["status"] == "done":
            old = [q for q in Path(r["out_dir"]).glob(f"{r['yt_id']}.*")
                   if q.suffix in (".mp4", ".mkv", ".webm")]
            if old:
                spent += old[0].stat().st_size
    ytdlp = os.environ.get("AIC_YTDLP", "yt-dlp")
    waited = 0.0
    last_failed = False
    for n, r in enumerate(todo):
        if budget_bytes is not None and spent >= budget_bytes:
            stats["budget_stop"] = True
            print(f"[budget] reached {spent/1e9:.1f} GB, stopping", flush=True)
            break
        if progress_every and n and n % progress_every == 0:
            print(f"[progress] {n}/{len(todo)} done={stats['done']} failed={stats['failed']} "
                  f"gave_up={stats['gave_up']} spent={spent/1e9:.1f} GB", flush=True)
        if proxy and (n % 25 == 0 or last_failed):
            waited = 0.0
            while not proxy_alive(proxy):
                if waited >= 3600:
                    print("[proxy] still down after 60 min, stopping (re-run to resume)", flush=True)
                    return stats
                print(f"[proxy] down, waiting 60s ({waited:.0f}s so far)", flush=True)
                time.sleep(60)
                waited += 60
        if last_failed:
            last_failed = False
        out_dir = Path(r["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob(f"{r['yt_id']}.*"))
        existing = [p for p in existing if p.suffix in (".mp4", ".mkv", ".webm") and not p.name.endswith(".part")]
        if existing:
            spent += existing[0].stat().st_size
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
            size = files[0].stat().st_size
            spent += size
            append_jsonl(queue, {**r, "status": "done", "attempt": attempt, "path": str(files[0]),
                                 "bytes": size, "error": None, "time": now()})
            stats["done"] += 1
        else:
            permanent = any(s.lower() in err.lower() for s in PERMANENT)
            status = "gave_up" if permanent or attempt >= max_attempts else "failed"
            append_jsonl(queue, {**r, "status": status, "attempt": attempt, "error": err.strip()[-400:],
                                 "permanent": permanent, "time": now()})
            stats[status] += 1
            last_failed = True
        time.sleep(sleep_s)
    return stats
