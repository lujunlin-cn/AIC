#!/usr/bin/env python3
"""P0-a: match official semifinal 426 videos against in-stock PHD2 sources.

Stage 1: brute-force pHash match (official keyframe hashes vs PHD2 probe bank).
Stage 2: per-candidate verification by re-decoding source frames at the
estimated temporal offset; pass = median hamming <= 14 and inlier >= 0.6.
Stage 3: join PHD2 selections (train/test GIF intervals) on matched vids;
report interval-window overlap (intervals shifted into official timeline).

Outputs under /data/aic/experiments_910a/LFM_V11/r12_p0a/:
  candidates.jsonl, matched.jsonl, summary.json
"""
import json
import os

import numpy as np
import av
from PIL import Image

BASE = "/data/aic/experiments_910a/LFM_V11/r12_p0a"
BANK_DIR = os.path.join(BASE, "bank")
OFFICIAL_KFS = os.path.join(BASE, "official_kfs.jsonl")
INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
SELECTIONS = [
    "/data/aic/external_datasets/PHD2/annotations/selections/train.json",
    "/data/aic/external_datasets/PHD2/annotations/selections/test.json",
]
RAW_DIR = "/data/aic/external_datasets/PHD2/raw/youtube"

HAM_CAND = 12      # candidate hit threshold (of 63 bits)
HAM_PASS_MED = 14  # verification median threshold
HAM_INlier = 18    # verification inlier threshold
INLIER_MIN = 0.6
CHUNK_Q = 64

_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """q:(m,) uint64, b:(n,) uint64 -> (m,n) popcount of xor, chunked over q."""
    out = np.empty((q.shape[0], b.shape[0]), dtype=np.uint16)
    for i in range(0, q.shape[0], CHUNK_Q):
        x = q[i : i + CHUNK_Q, None] ^ b[None, :]
        out[i : i + CHUNK_Q] = _LUT[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(-1)
    return out


def h2u64(hexlist):
    return np.array([int(h, 16) for h in hexlist], dtype=np.uint64)


def seek_frame(path: str, ts: float):
    """Decode one frame at/after ts. Returns (frame_ts, 32x32 gray) or None."""
    try:
        with av.open(path) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            tb = st.time_base
            c.seek(int(ts / tb), stream=st, backward=True)
            got = None
            for frame in c.decode(st):
                fts = float(frame.pts * tb) if frame.pts is not None else ts
                got = (fts, frame)
                if fts >= ts - 0.05:
                    break
            if got is None:
                return None
            img = got[1].to_image().convert("L").resize((32, 32), Image.BILINEAR)
            return got[0], np.asarray(img, dtype=np.float64) / 255.0
    except Exception:  # noqa: BLE001
        return None


_D = np.zeros((32, 32), dtype=np.float64)
for k in range(32):
    for n in range(32):
        _D[k, n] = np.cos(np.pi * (2 * n + 1) * k / 64) * (np.sqrt(1 / 32) if k == 0 else np.sqrt(2 / 32))


def phash64(gray32: np.ndarray) -> int:
    coef = _D @ gray32 @ _D.T
    block = coef[:8, :8].flatten()[1:]
    bits = block > np.median(block)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def main():
    # ---- load official
    intake = {}
    for line in open(INTAKE):
        r = json.loads(line)
        intake[r["video_id"]] = r
    off = {}
    for line in open(OFFICIAL_KFS):
        r = json.loads(line)
        off[r["vid"]] = r

    # ---- load bank
    vids, hashes, tss = [], [], []
    for fn in sorted(os.listdir(BANK_DIR)):
        for line in open(os.path.join(BANK_DIR, fn)):
            r = json.loads(line)
            if not r.get("h"):
                continue
            n = len(r["h"])
            vids.extend([r["vid"]] * n)
            tss.extend(r["ts"])
            hashes.extend(r["h"])
    bank_h = h2u64(hashes)
    bank_ts = np.array(tss, dtype=np.float64)
    vid2i = {v: i for i, v in enumerate(sorted(set(vids)))}
    bank_vid = np.array([vid2i[v] for v in vids], dtype=np.int32)
    vid_names = sorted(vid2i, key=vid2i.get)
    print(f"bank: {bank_h.shape[0]} probes over {len(vid_names)} vids", flush=True)

    # ---- stage 1: candidates
    sel = {}
    for p in SELECTIONS:
        sel.update(json.load(open(p)))

    cand_f = open(os.path.join(BASE, "candidates.jsonl"), "w", buffering=1)
    match_f = open(os.path.join(BASE, "matched.jsonl"), "w", buffering=1)
    n_matched = 0
    stats = {"n_official": len(off), "n_bank_vids": len(vid_names), "matched": 0,
             "verified": 0, "with_interval_in_window": 0}
    rows = []
    _done = 0
    for ov, r in sorted(off.items(), key=lambda kv: int(kv[0])):
        _done += 1
        if _done % 50 == 0:
            print(f"[match] {_done}/{len(off)} matched={stats['matched']}", flush=True)
        if not r["h"]:
            continue
        q = h2u64(r["h"])
        qts_frames = [int(f) for f in r["frames"]]
        fps = intake[ov]["fps"]
        qts = np.array([f / fps for f in qts_frames], dtype=np.float64)
        hm = hamming_matrix(q, bank_h)
        # best hit per query
        best_idx = hm.argmin(1)
        best_val = hm[np.arange(len(q)), best_idx]
        cand_counts = {}
        for qi, bi, val in zip(range(len(q)), best_idx, best_val):
            if val <= HAM_CAND:
                bv = vid_names[bank_vid[bi]]
                cand_counts.setdefault(bv, []).append((float(qts[qi]), float(bank_ts[bi]), int(val)))
        if not cand_counts:
            cand_f.write(json.dumps({"vid": ov, "hits": 0}) + "\n")
            continue
        # rank candidates by hit count then min hamming
        ranked = sorted(cand_counts.items(), key=lambda kv: (-len(kv[1]), min(x[2] for x in kv[1])))
        for bv, hits in ranked[:1]:
            cand_f.write(json.dumps({"vid": ov, "cand": bv, "n_hits": len(hits), "hits": hits[:20]}) + "\n")
        # ---- stage 2: verify top candidate
        bv, hits = ranked[0]
        offs = np.array([t[1] - t[0] for t in hits], dtype=np.float64)
        offset = float(np.median(offs))
        src_path = os.path.join(RAW_DIR, bv + ".mp4")
        med_val, inlier, n_try = 63, 0.0, 0
        vals = []
        for t_off_q in qts:
            src_ts = t_off_q + offset
            fr = seek_frame(src_path, src_ts)
            if fr is None:
                continue
            n_try += 1
            v = int(bin(q[int(np.where(qts == t_off_q)[0][0])] ^ phash64(fr[1])).count("1"))
            vals.append(v)
        if n_try:
            med_val = int(np.median(vals))
            inlier = float(np.mean([v <= HAM_INlier for v in vals]))
        ok = n_try >= 5 and med_val <= HAM_PASS_MED and inlier >= INLIER_MIN
        odur = intake[ov]["duration"]
        sdur = None
        # source duration from bank rows (any row of this vid)
        sdur = float(bank_ts[bank_vid == vid2i[bv]].max()) if np.any(bank_vid == vid2i[bv]) else None
        row = {"vid": ov, "phd2_vid": bv, "offset": round(offset, 3),
               "n_hits": len(hits), "n_verify": n_try, "med_ham": med_val,
               "inlier": round(inlier, 3), "verified": bool(ok),
               "official_dur": odur, "source_dur_probe_max": sdur}
        # ---- stage 3: GIF interval overlap (informational)
        if ok:
            stats["verified"] += 1
            ivs = []
            for user, lst in sel.get(bv, {}).items():
                for x in lst:
                    a, b = x["t0"] - offset, x["t1"] - offset
                    if b > 0 and a < odur:
                        ivs.append({"user": user, "in_off": [round(max(a, 0.0), 2), round(min(b, odur), 2)],
                                    "is_last": x.get("is_last")})
            row["intervals_in_window"] = ivs
            if ivs:
                stats["with_interval_in_window"] += 1
        if ok:
            stats["matched"] += 1
            match_f.write(json.dumps(row) + "\n")
        rows.append(row)
    cand_f.close()
    match_f.close()

    stats["offsets"] = sorted(set(round(r["offset"], 1) for r in rows if r.get("verified")))[:50]
    with open(os.path.join(BASE, "summary.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print("P0A_SUMMARY " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
