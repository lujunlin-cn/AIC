"""Resumable HTTP downloads with bounded retries and a download ledger.

* Partial data is kept in ``<dest>.part`` and resumed with HTTP Range.
* A finished file is only renamed into place after its size (and optional
  SHA-256) match; the ledger records URL, size, SHA-256 and attempts.
* Google Drive virus-scan interstitials are handled by re-requesting via
  ``drive.usercontent.google.com`` with ``confirm=t``.
* ``AIC_EXT_PROXY`` (e.g. ``http://127.0.0.1:18890``) is applied only to hosts
  that cannot be reached directly from the server.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from .common import EXT_ROOT, append_jsonl, ensure_space, log, now, sha256_file

PROXY = os.environ.get("AIC_EXT_PROXY", "")
# Hosts known to be unreachable from the server without the proxy.
PROXY_HOSTS = ("google.com", "googleusercontent.com", "box.com", "boxcloud.com",
               "youtube.com", "googlevideo.com", "huggingface.co", "hf.co",
               "xethub.hf.co", "dropbox.com", "dropboxusercontent.com",
               "onedrive.live.com", "1drv.ms", "sharepoint.com", "fbaipublicfiles.com",
               "facebook.com", "meta.com", "cloudfront.net", "amazonaws.com")
LEDGER = EXT_ROOT / "_registry" / "downloads.jsonl"
UA = "Mozilla/5.0 (X11; Linux x86_64) AIC-dataset-fetch/1.0"
CHUNK = 4 << 20
RANGE_SPAN = 64 << 20  # request size in bounded-Range mode
QUOTA_SLEEP = (15, 600)  # backoff between refused chunk requests: start, cap (s)
QUOTA_PATIENCE_S = float(os.environ.get("AIC_EXT_QUOTA_PATIENCE_H", "12")) * 3600  # give up (resumable later)


def proxies_for(url: str) -> dict:
    host = urlparse(url).hostname or ""
    if PROXY and any(host == h or host.endswith("." + h) for h in PROXY_HOSTS):
        return {"http": PROXY, "https": PROXY}
    return {}


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.trust_env = False  # never inherit global proxies silently
    return s


def gdrive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def head_size(url: str, sess: Optional[requests.Session] = None, timeout: int = 40) -> Optional[int]:
    """Content length via a 1-byte Range request (works where HEAD is refused)."""
    sess = sess or session()
    r = sess.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout,
                 proxies=proxies_for(url), allow_redirects=True)
    try:
        cr = r.headers.get("Content-Range", "")
        if r.status_code == 206 and "/" in cr and cr.split("/")[-1].isdigit():
            return int(cr.split("/")[-1])
        cl = r.headers.get("Content-Length")
        if r.status_code == 200 and cl and cl.isdigit() and "text/html" not in r.headers.get("Content-Type", ""):
            return int(cl)
        return None
    finally:
        r.close()


def fetch(url: str, dest: str | Path, *, dataset: str, expected_size: Optional[int] = None,
          expected_sha256: Optional[str] = None, retries: int = 6, timeout: int = 60,
          min_bytes: int = 1, headers: Optional[dict] = None, check_space: bool = True,
          sess: Optional[requests.Session] = None, quiet: bool = False) -> dict:
    """Download ``url`` to ``dest`` resumably. Returns the ledger row."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sess = sess or session()
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        row = {"time": now(), "dataset": dataset, "url": url, "path": str(dest),
               "size": dest.stat().st_size, "status": "exists"}
        if expected_sha256:
            row["sha256"] = sha256_file(dest)
            if row["sha256"] != expected_sha256:
                raise ValueError(f"sha256 mismatch for existing {dest}")
        return row
    if check_space and expected_size:
        have = part.stat().st_size if part.exists() else 0
        ensure_space(expected_size - have)
    last_err = None
    total = expected_size
    # Drive "Quota exceeded": full GETs are refused but bounded Range requests are served
    # intermittently.  Switch to fixed-size chunks at a slow, polite pace; give up only after
    # QUOTA_PATIENCE consecutive refusals without progress.
    bounded = False
    quota_misses, quota_since = 0, None
    attempt = 0
    while attempt < retries:
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have == total:
            break
        hdr = {**(headers or {})}
        want = None
        if bounded:
            want = min(have + RANGE_SPAN, total) - have
            hdr["Range"] = f"bytes={have}-{have + want - 1}"
        elif have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with sess.get(url, headers=hdr, stream=True, timeout=timeout,
                          proxies=proxies_for(url), allow_redirects=True) as r:
                ctype = r.headers.get("Content-Type", "")
                if r.status_code == 416 and total is not None and have >= total:
                    break
                if r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                if "text/html" in ctype and (expected_size or 0) > 1 << 20:
                    body = r.raw.read(4096, decode_content=True)
                    if total and b"Quota exceeded" in body:
                        if not bounded:
                            bounded = True
                            log(dataset, "fetch_bounded_range", url=url, span=RANGE_SPAN, have=have)
                            continue
                        quota_misses += 1
                        quota_since = quota_since or time.time()
                        if time.time() - quota_since >= QUOTA_PATIENCE_S:
                            raise IOError(f"quota exceeded {quota_misses}x without progress")
                        wait = min(QUOTA_SLEEP[1], QUOTA_SLEEP[0] * 2 ** min(quota_misses - 1, 10))
                        if quota_misses in (1, 5) or quota_misses % 20 == 0:
                            log(dataset, "fetch_quota_wait", url=url, misses=quota_misses, have=have, wait_s=wait)
                        time.sleep(wait)
                        continue
                    raise IOError(f"HTML instead of file: {body[:200]!r}")
                if bounded and r.status_code != 206:
                    raise IOError(f"bounded Range not honoured (HTTP {r.status_code})")
                if have and r.status_code == 200:
                    have = 0  # server ignored Range; restart cleanly
                mode = "ab" if have else "wb"
                cr = r.headers.get("Content-Range", "")
                if cr and "/" in cr and cr.split("/")[-1].isdigit():
                    total = int(cr.split("/")[-1])
                elif r.status_code == 200 and r.headers.get("Content-Length", "").isdigit():
                    total = int(r.headers["Content-Length"])
                if check_space and total and attempt == 1:
                    ensure_space(total - have)
                t0, got = time.time(), 0
                with part.open(mode) as f:
                    for block in r.iter_content(CHUNK):
                        if block:
                            f.write(block)
                            got += len(block)
                if not quiet:
                    log(dataset, "fetch_chunk", url=url, got=got, seconds=round(time.time() - t0, 1),
                        size=part.stat().st_size, total=total)
            if total is None or part.stat().st_size >= total:
                break
            if bounded and got:  # progress in chunk mode resets the failure budget
                attempt, quota_misses, quota_since = 0, 0, None
                if not quiet or part.stat().st_size // (1 << 30) != (part.stat().st_size - got) // (1 << 30):
                    log(dataset, "fetch_progress", url=url, size=part.stat().st_size, total=total)
            else:
                attempt += 1
        except Exception as exc:  # network errors are retried with backoff
            attempt += 1
            last_err = repr(exc)
            log(dataset, "fetch_retry", url=url, attempt=attempt, error=last_err[:300])
            time.sleep(min(60, 5 * attempt))
    size = part.stat().st_size if part.exists() else 0
    ok = size >= min_bytes and (total is None or size == total) and \
        (expected_size is None or size == expected_size)
    row = {"time": now(), "dataset": dataset, "url": url, "path": str(dest), "size": size,
           "expected_size": expected_size, "server_size": total}
    if not ok:
        row.update(status="failed", error=last_err or "size mismatch")
        append_jsonl(LEDGER, row)
        raise IOError(f"download failed {url}: {row['error']}")
    digest = sha256_file(part)
    if expected_sha256 and digest != expected_sha256:
        row.update(status="failed", error="sha256 mismatch", sha256=digest)
        append_jsonl(LEDGER, row)
        raise IOError(f"sha256 mismatch {url}")
    part.replace(dest)
    row.update(status="downloaded", sha256=digest)
    append_jsonl(LEDGER, row)
    return row


def gdrive_fetch(file_id: str, dest: str | Path, **kw) -> dict:
    """Google Drive large-file download without cookies (confirm=t)."""
    return fetch(gdrive_url(file_id), dest, **kw)


def github_raw(repo: str, path: str, ref: str = "HEAD") -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def github_head_commit(repo: str, sess: Optional[requests.Session] = None) -> Optional[str]:
    sess = sess or session()
    try:
        r = sess.get(f"https://api.github.com/repos/{repo}/commits?per_page=1", timeout=30)
        if r.ok and r.json():
            return r.json()[0]["sha"]
    except Exception:
        pass
    try:  # API rate limits: fall back to the git smart-http ref advertisement
        r = sess.get(f"https://github.com/{repo}.git/info/refs?service=git-upload-pack", timeout=30)
        m = re.search(rb"([0-9a-f]{40}) HEAD", r.content)
        return m.group(1).decode() if m else None
    except Exception:
        return None
