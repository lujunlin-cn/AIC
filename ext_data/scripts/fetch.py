#!/usr/bin/env python3
"""Resumable single-file fetch CLI (wraps aicext.download.fetch).

Example:
  python -m scripts.fetch --dataset DHF1K --gdrive 1UEFQmRdDbtVT-ePjMZVrv9oVV0ra631s \
      --dest /data/aic/external_datasets/DHF1K/raw/archives/video.rar --expected-size 4032563381
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import log  # noqa: E402
from aicext.download import fetch, gdrive_url  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url")
    g.add_argument("--gdrive")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--expected-size", type=int)
    ap.add_argument("--sha256")
    ap.add_argument("--retries", type=int, default=20)
    a = ap.parse_args()
    url = a.url or gdrive_url(a.gdrive)
    log(a.dataset, "fetch_start", url=url, dest=a.dest)
    row = fetch(url, a.dest, dataset=a.dataset, expected_size=a.expected_size,
                expected_sha256=a.sha256, retries=a.retries)
    log(a.dataset, "fetch_done", **{k: row.get(k) for k in ("path", "size", "sha256", "status")})


if __name__ == "__main__":
    main()
