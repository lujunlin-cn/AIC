"""Read a remote ZIP's central directory with HTTP Range requests (no full download).

Used to verify repacked mirrors member-by-member (name, size, CRC-32) against an
official archive when the whole-file checksums cannot match.
"""
from __future__ import annotations

import struct
from typing import Optional

import requests

from .download import proxies_for, session


def _get(sess: requests.Session, url: str, start: int, end: int) -> bytes:
    r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120, proxies=proxies_for(url))
    if r.status_code != 206:
        raise IOError(f"range not honoured: HTTP {r.status_code}")
    return r.content


def central_directory(url: str, sess: Optional[requests.Session] = None) -> dict[str, tuple[int, int]]:
    """Return {name: (uncompressed_size, crc32)} for every member."""
    sess = sess or session()
    head = sess.get(url, headers={"Range": "bytes=0-0"}, timeout=60, proxies=proxies_for(url))
    total = int(head.headers["Content-Range"].split("/")[-1])
    tail = _get(sess, url, max(0, total - 70000), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise IOError("EOCD not found")
    _, _, _, _, n_total, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if cd_off == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        _, _, _, _, _, _, _, n_total, cd_size, cd_off = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
    cd = _get(sess, url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (crc, csize, usize, nlen, xlen, clen) = struct.unpack("<IIIHHH", cd[p + 16:p + 34])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if usize == 0xFFFFFFFF:  # zip64 extra field carries the real size
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    usize = struct.unpack("<Q", extra[q + 4:q + 12])[0]
                    break
                q += 4 + hlen
        out[name] = (usize, crc)
        p += 46 + nlen + xlen + clen
    return out
