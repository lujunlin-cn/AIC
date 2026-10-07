#!/usr/bin/env python3
"""P0-a: PHD2 in-stock media pHash bank (32-shard, CPU).

Probe every 2.0s of each in-stock PHD2 source video, compute 64-bit pHash
per probe frame, append one jsonl line per video:
  {"vid": <ytid>, "dur": <sec>, "n": <probes>, "ts": [...], "h": ["<16hex>", ...]}

Resume: videos already present in the shard output are skipped.
Usage: python3 p0a_phd2_phash_bank.py --shard i --nshard 32
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import av

RAW_DIR = "/data/aic/external_datasets/PHD2/raw/youtube"
OUT_DIR = "/data/aic/experiments_910a/LFM_V11/r12_p0a/bank"
PROBE_STEP = 2.0
MAX_PROBES = 400
OMP_LIMIT = "2"

# 32x32 DCT-II basis matrix
_D = np.zeros((32, 32), dtype=np.float64)
for k in range(32):
    for n in range(32):
        _D[k, n] = np.cos(np.pi * (2 * n + 1) * k / 64) * (np.sqrt(1 / 32) if k == 0 else np.sqrt(2 / 32))


def phash64(img_gray_small: np.ndarray) -> int:
    """img: 32x32 float64 gray in [0,1]. Returns 63-bit hash in an int."""
    coef = _D @ img_gray_small @ _D.T
    block = coef[:8, :8].flatten()[1:]  # exclude DC
    med = np.median(block)
    bits = block > med
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def to_gray32(frame) -> np.ndarray:
    img = frame.to_image().convert("L").resize((32, 32), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def probe_video(path: str):
    """Return (dur, [(ts, hash_hex), ...]). Seek-based single-frame probes."""
    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        dur = float(st.duration * st.time_base) if st.duration else None
        if dur is None:
            try:
                dur = float(c.duration * av.time_base)
            except Exception:
                dur = None
        if not dur or dur <= 0:
            return None, []
        t = 0.0
        tb = st.time_base
        while t < dur and len(out) < MAX_PROBES:
            try:
                c.seek(int(t / tb), stream=st, backward=True)
                got = None
                for frame in c.decode(st):
                    fts = float(frame.pts * tb) if frame.pts is not None else t
                    if fts >= t - 0.05:
                        got = frame
                        break
                    got = frame  # keep last decoded as fallback
                if got is not None:
                    out.append((round(fts, 3), "%016x" % phash64(to_gray32(got))))
            except (av.AVError, OSError, ValueError):
                pass
            t += PROBE_STEP
    return dur, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshard", type=int, default=32)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = OMP_LIMIT

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"shard_{args.shard:02d}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["vid"])
                except Exception:
                    pass

    vids = sorted(
        f[: -len(".mp4")]
        for f in os.listdir(RAW_DIR)
        if f.endswith(".mp4")
    )
    mine = [v for i, v in enumerate(vids) if i % args.nshard == args.shard]
    todo = [v for v in mine if v not in done]
    print(f"[shard {args.shard}] mine={len(mine)} done={len(done)} todo={len(todo)}", flush=True)

    n_err = 0
    with open(out_path, "a") as fo:
        for k, vid in enumerate(todo):
            path = os.path.join(RAW_DIR, vid + ".mp4")
            try:
                dur, probes = probe_video(path)
                if dur is None or not probes:
                    n_err += 1
                    fo.write(json.dumps({"vid": vid, "dur": dur, "n": 0, "ts": [], "h": []}) + "\n")
                else:
                    fo.write(json.dumps({"vid": vid, "dur": round(dur, 3), "n": len(probes),
                                         "ts": [p[0] for p in probes], "h": [p[1] for p in probes]}) + "\n")
            except Exception as e:  # noqa: BLE001
                n_err += 1
                fo.write(json.dumps({"vid": vid, "dur": None, "n": 0, "ts": [], "h": [], "err": str(e)[:120]}) + "\n")
            fo.flush()
            if (k + 1) % 50 == 0:
                print(f"[shard {args.shard}] pairs_total={k + 1}/{len(todo)} err={n_err}", flush=True)
    print(f"[shard {args.shard}] DONE total={len(todo)} err={n_err}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
