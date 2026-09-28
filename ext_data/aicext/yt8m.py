"""Stream YouTube-8M frame-level TFRecords and keep only selected examples.

No TensorFlow: TFRecord framing (u64 length, u32 crc, data, u32 crc) and the
``tf.train.SequenceExample`` protobuf are decoded by hand.  Only examples whose
context ``id`` is requested are materialised; the shard itself is never
written to disk, so peak disk use is the extracted features only.

Output per shard: ``<out>/<shard>.npz``-free HDF5 ``<out>/<shard>.h5`` with
``<video_id>/rgb`` uint8[T,1024], ``<video_id>/audio`` uint8[T,128],
``<video_id>/labels`` int64[], attrs ``yt8m_id``.  Values are the quantized
bytes as published; dequantize with ``x * 4/255 - 2`` (YouTube-8M / Mr.HiSum).
"""
from __future__ import annotations

import hashlib
import itertools
import struct
from typing import Iterator

VOCAB = [chr(c) for c in range(ord("a"), ord("z") + 1)] + \
        [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)]
FILE_IDS = ["".join(p) for p in itertools.product(VOCAB, repeat=2)]


def shard_remote_name(local: str) -> str:
    """``train0026`` (Mr.HiSum metadata) -> ``trainaA`` (mirror file stem)."""
    prefix, num = local[:-4], int(local[-4:])
    return prefix + FILE_IDS[num]


def _varint(buf: memoryview, i: int) -> tuple[int, int]:
    shift = res = 0
    while True:
        b = buf[i]
        i += 1
        res |= (b & 0x7F) << shift
        if not b & 0x80:
            return res, i
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, int, object]]:
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
            yield fno, wt, v
        elif wt == 2:
            ln, i = _varint(buf, i)
            yield fno, wt, buf[i:i + ln]
            i += ln
        elif wt == 5:
            yield fno, wt, buf[i:i + 4]
            i += 4
        elif wt == 1:
            yield fno, wt, buf[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wt}")


def _map_entries(buf: memoryview) -> Iterator[tuple[str, memoryview]]:
    """Features / FeatureLists: repeated field 1 = MapEntry{1: key, 2: value}."""
    for fno, _, entry in _fields(buf):
        if fno != 1:
            continue
        key, val = None, None
        for f2, _, v in _fields(entry):
            if f2 == 1:
                key = bytes(v).decode()
            elif f2 == 2:
                val = v
        yield key, val


def _bytes_list(feature: memoryview) -> list[bytes]:
    out = []
    for fno, _, v in _fields(feature):
        if fno == 1:  # bytes_list
            for f2, _, b in _fields(v):
                if f2 == 1:
                    out.append(bytes(b))
    return out


def _int64_list(feature: memoryview) -> list[int]:
    out = []
    for fno, _, v in _fields(feature):
        if fno == 3:  # int64_list
            for f2, wt, x in _fields(v):
                if f2 != 1:
                    continue
                if wt == 0:
                    out.append(x)
                else:  # packed
                    j = 0
                    while j < len(x):
                        val, j = _varint(x, j)
                        out.append(val)
    return out


def context_id(rec: memoryview) -> tuple[str | None, list[int]]:
    vid, labels = None, []
    for fno, _, v in _fields(rec):
        if fno == 1:
            for k, feat in _map_entries(v):
                if k == "id":
                    b = _bytes_list(feat)
                    vid = b[0].decode() if b else None
                elif k == "labels":
                    labels = _int64_list(feat)
            break
    return vid, labels


def feature_lists(rec: memoryview) -> dict[str, list[bytes]]:
    out = {}
    for fno, _, v in _fields(rec):
        if fno == 2:
            for k, flist in _map_entries(v):
                frames = []
                for f2, _, feat in _fields(flist):
                    if f2 == 1:
                        b = _bytes_list(feat)
                        frames.append(b[0] if b else b"")
                out[k] = frames
    return out


def iter_records(stream, md5=None) -> Iterator[memoryview]:
    """TFRecord framing over a file-like stream (CRCs not verified; md5 of the whole shard is)."""
    def read(n):
        chunks, got = [], 0
        while got < n:
            c = stream.read(n - got)
            if not c:
                break
            chunks.append(c)
            got += len(c)
        data = b"".join(chunks)
        if md5 is not None:
            md5.update(data)
        return data

    while True:
        head = read(12)
        if not head:
            return
        if len(head) < 12:
            raise IOError("truncated TFRecord header")
        (ln,) = struct.unpack("<Q", head[:8])
        body = read(ln + 4)
        if len(body) < ln + 4:
            raise IOError("truncated TFRecord body")
        yield memoryview(body)[:ln]


def extract_shard(url: str, wanted: dict[str, str], out_path, expected_md5: str | None,
                  sess, timeout: int = 120) -> dict:
    """Stream one shard; ``wanted`` maps yt8m random_id -> our video key."""
    import h5py
    import numpy as np

    md5 = hashlib.md5()
    found = {}
    tmp = str(out_path) + ".tmp"
    with sess.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with h5py.File(tmp, "w") as h:
            n_rec = 0
            for rec in iter_records(r.raw, md5):
                n_rec += 1
                rid, labels = context_id(rec)
                if rid not in wanted:
                    continue
                fl = feature_lists(rec)
                rgb = np.frombuffer(b"".join(fl.get("rgb", [])), np.uint8).reshape(-1, 1024)
                aud = np.frombuffer(b"".join(fl.get("audio", [])), np.uint8).reshape(-1, 128)
                g = h.create_group(wanted[rid])
                g.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=1)
                g.create_dataset("audio", data=aud, compression="gzip", compression_opts=1)
                g.create_dataset("labels", data=np.asarray(labels, np.int64))
                g.attrs["yt8m_id"] = rid
                found[wanted[rid]] = int(rgb.shape[0])
    digest = md5.hexdigest()
    if expected_md5 and digest != expected_md5:
        raise IOError(f"md5 mismatch {url}: {digest} != {expected_md5}")
    import os
    os.replace(tmp, out_path)
    return {"records": n_rec, "found": found, "md5": digest}
