"""Canonical source-video IDs and leakage groups.

A *group* is the unit that must never be split across train/val: every clip,
annotator, crop ratio and derived version of one source video shares it.

* DHF1K and RetargetVid share media, so both map DHF1K video ``NNN`` to
  ``dhf1k:NNN`` (group ``dhf1k:NNN``).
* YouTube-backed datasets map to ``yt:<11-char id>`` so the same YouTube video
  collides across TVSum, QVHighlights, YouTube Highlights, PHD2, Mr.HiSum ...
* Datasets with native media only use ``<dataset>:<id>``.
"""
from __future__ import annotations

import hashlib
import re

YT_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def dhf1k_id(n: int | str) -> str:
    return f"dhf1k:{int(n):03d}"


def youtube_id(yid: str) -> str:
    yid = yid.strip()
    if not YT_RE.match(yid):
        raise ValueError(f"not a YouTube id: {yid!r}")
    return f"yt:{yid}"


def qvh_youtube(vid: str) -> str:
    """QVHighlights ``vid`` is ``<ytid>_<start>_<end>``; the ytid may contain ``_``."""
    m = re.match(r"^(.{11})_\d+(\.\d+)?_\d+(\.\d+)?$", vid)
    if not m:
        raise ValueError(vid)
    return youtube_id(m.group(1))


def native_id(dataset: str, item: str) -> str:
    return f"{dataset.lower()}:{item}"


def split_bucket(group_id: str, salt: str = "aic-ext-v1") -> float:
    """Deterministic [0,1) hash used when a dataset has no official val split."""
    h = hashlib.sha256(f"{salt}:{group_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)
