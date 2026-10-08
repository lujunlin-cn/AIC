"""R11 S2b: person presence / box-stability features from S2a detections.

Slot-granular proxy for tracking continuity (2 s slot gaps make frame IoU
tracking unreliable): per-slot presence, confidence, person count, mean box
area, and center-jitter to the previous presence slot (nearest-center L2).
Writes the R10 readout contract (per-frag npz with a (8, 5) feature block and
an avalid mask) so r10_audio_head.py --feature s2b_track consumes it as-is.
PHD2 pool (GT manifest = r9_audit).  CPU only.
"""
import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--det-root', default='/data/aic/experiments_910a/LFM_V11/r11_s2a')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_s2b_feats')
a = p.parse_args()

OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
files = sorted(Path(a.det_root).glob('p*/*.npz'))
print(f'{len(files)} det frags', flush=True)
n = bad = 0
for f in files:
    try:
        with np.load(f) as src:  # NpzFile holds an fd -- close it or die at ulimit
            box, conf, nd, t = src['det_box'], src['det_conf'], src['n_det'], src['t']
        if box.ndim != 2 or nd.ndim != 1 or box.shape[0] != nd.shape[0]:
            bad += 1
            continue
        S = box.shape[0]
        feat = np.zeros((S, 5), np.float32)
        avalid = np.ones(S, np.float32)
        centers = [None] * S
        for s in range(S):
            # det schema: per slot ONE box (best conf) in box[s] (4 coords);
            # n_det[s] = raw detection count that slot (may exceed 1).
            if int(nd[s]) <= 0 or box.ndim != 2 or box.shape[1] != 4:
                continue
            b = box[s].reshape(1, 4)
            cs = np.asarray(conf[s]).reshape(-1)
            if cs.size == 0:
                continue
            w = np.maximum(b[0, 2] - b[0, 0], 0)
            h = np.maximum(b[0, 3] - b[0, 1], 0)
            centers[s] = b[0, :2] + (b[0, 2:] - b[0, :2]) / 2
            feat[s, 0] = 1.0                                   # presence
            feat[s, 1] = float(cs[0])                          # box conf
            feat[s, 2] = float(w * h)                          # box area
            feat[s, 4] = float(nd[s])                          # n detections
        for s in range(1, S):
            if centers[s] is not None and centers[s - 1] is not None:
                feat[s, 3] = float(np.linalg.norm(centers[s] - centers[s - 1]))  # center jitter
        np.savez(OUT / f'{f.stem}.npz', s2b_track=feat, avalid=avalid, t=t)
        n += 1
    except Exception:  # noqa: BLE001
        bad += 1
print(f'wrote {n} frags ({bad} skipped malformed) -> {OUT}', flush=True)
