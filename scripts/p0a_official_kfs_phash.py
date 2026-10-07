#!/usr/bin/env python3
"""P0-a: official semifinal 426 keyframes pHash (CPU, fast).

Reads /data/aic/semifinal_20261001/keyframes/keyframes/<vid>/*.png
(filenames are frame indices), writes one jsonl line per video:
  {"vid": "<id>", "frames": ["<frame_idx>", ...], "h": ["<16hex>", ...]}
"""
import json
import os

import numpy as np
from PIL import Image

KF_DIR = "/data/aic/semifinal_20261001/keyframes/keyframes"
OUT = "/data/aic/experiments_910a/LFM_V11/r12_p0a/official_kfs.jsonl"

_D = np.zeros((32, 32), dtype=np.float64)
for k in range(32):
    for n in range(32):
        _D[k, n] = np.cos(np.pi * (2 * n + 1) * k / 64) * (np.sqrt(1 / 32) if k == 0 else np.sqrt(2 / 32))


def phash64(img_gray_small: np.ndarray) -> int:
    coef = _D @ img_gray_small @ _D.T
    block = coef[:8, :8].flatten()[1:]
    med = np.median(block)
    bits = block > med
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fo:
        for vid in sorted(os.listdir(KF_DIR), key=lambda s: int(s) if s.isdigit() else 0):
            vdir = os.path.join(KF_DIR, vid)
            if not os.path.isdir(vdir):
                continue
            frames, hashes = [], []
            for fn in sorted(os.listdir(vdir), key=lambda s: int(os.path.splitext(s)[0]) if os.path.splitext(s)[0].isdigit() else -1):
                if not fn.endswith(".png"):
                    continue
                img = Image.open(os.path.join(vdir, fn)).convert("L").resize((32, 32), Image.BILINEAR)
                frames.append(os.path.splitext(fn)[0])
                hashes.append("%016x" % phash64(np.asarray(img, dtype=np.float64) / 255.0))
            fo.write(json.dumps({"vid": vid, "frames": frames, "h": hashes}) + "\n")
    print(f"official_kfs DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
