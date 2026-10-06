"""R9 signal-census CPU probe (V10 Q2 families 4/7 + Q3 pre-check).

Extracts, per frozen fragment, cheap per-slot scalar features with NO
learned encoder, plus the audio-track existence fact for the whole manifest:

  vis (8,5): frame-diff energy, Laplacian sharpness, brightness,
             saturation, adjacent-frame histogram chi2  (slot context
             [t-3..t+4] integer seconds, 112x112 - resolution is fine for
             statistics)
  aud (8,6): per-second RMS mean/std/max/flux, silence ratio, cheer
             contrast (max-median)/(median+eps) over the same window
  has_audio: per SOURCE (broadcast through every frag of that source)

Decode: PyAV, low resolution, 8-way sharded over fragments (CPU only,
nice-able, no NPU).  One npz per fragment in p{shard}/ - same layout as the
NPU feature dirs so the screen script can glob them identically.
No official media; no labels read here.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

p = argparse.ArgumentParser()
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--pool-feats', default='/data/aic/experiments_910a/LFM_V10/pool_feats')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r9_signal_probe')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=8)
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402

man = []
for split in ('train_public', 'eval_public'):
    for l in (Path(a.manifest_dir) / f'{split}.jsonl').read_text().splitlines():
        if l.strip():
            man.append(json.loads(l))
fids = [r['fragment_id'] for r in man]
srcs = {r['fragment_id']: r['source_id'] for r in man}
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
MEDIA = Path(a.media_root)
POOL = Path(a.pool_feats)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} frags', flush=True)


def to_gray_small(im):
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (112, 112))


def vis_feats(grays):
    """(8 slots x up to 8 ctx frames) -> (8,5) slot scalars."""
    out = np.zeros((8, 5), np.float32)
    for si in range(8):
        gs = grays[si]
        dif, lap, bri, sat, hist = [], [], [], [], []
        for k in range(1, len(gs)):
            if gs[k] is None or gs[k - 1] is None:
                continue
            dif.append(float(np.abs(gs[k].astype(np.int16) -
                                    gs[k - 1].astype(np.int16)).mean()))
            lap.append(float(cv2.Laplacian(gs[k], cv2.CV_32F).var()))
            hist.append(float(_hist_chi2(gs[k - 1], gs[k])))
        out[si] = [np.mean(dif) if dif else 0,
                   np.mean(lap) if lap else 0,
                   np.mean([g for g in gs if g is not None]) if any(g is not None for g in gs) else 0,
                   0.0,   # saturation filled by caller (needs colour)
                   np.mean(hist) if hist else 0]
    return out


def _hist_chi2(ga, gb):
    ha = cv2.calcHist([ga], [0], None, [16], [0, 256]).ravel()
    hb = cv2.calcHist([gb], [0], None, [16], [0, 256]).ravel()
    s = ha + hb
    m = s > 0
    return float(((ha[m] - hb[m]) ** 2 / np.maximum(s[m], 1)).sum())


def rms_per_second(container, st):
    """Decode the whole audio track; return (per-second RMS array, ok)."""
    if st is None:
        return None, False
    buf = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32) / 32768.0 if arr.dtype.kind == 'i' else arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        buf.append(arr)
    if not buf:
        return None, False
    x = np.concatenate(buf)
    n = len(x) // 16000
    if n <= 0:
        return None, False
    rms = np.array([float(np.sqrt((x[i * 16000:(i + 1) * 16000] ** 2).mean() + 1e-12))
                    for i in range(n)], np.float32)
    return rms, True


def aud_feats(rms, t8):
    out = np.zeros((8, 6), np.float32)
    med = float(np.median(rms[rms > 0])) if (rms > 0).any() else 0.0
    for si, t in enumerate(t8.tolist()):
        lo, hi = int(max(t - 3, 0)), int(min(t + 4, len(rms) - 1))
        w = rms[lo:hi + 1]
        if len(w) == 0:
            continue
        flux = float(np.abs(np.diff(w)).mean()) if len(w) > 1 else 0.0
        sil = float((w < max(np.quantile(rms, 0.10), 1e-4)).mean())
        cheer = float((w.max() - med) / (med + 1e-4)) if med > 0 else 0.0
        out[si] = [float(w.mean()), float(w.std()), float(w.max()), flux, sil, cheer]
    return out


t0 = time.time(); done = 0; errs = 0; n_audio = 0
seen_src = {}
for w in mine:
    outp = OUTP / f'{w}.npz'
    if outp.exists():
        done += 1
        continue
    try:
        t8 = np.load(POOL / f'{w}.npz')['t'].astype(np.float64)
        src = srcs[w]
        vp = MEDIA / f'{src}.mp4'
        need = {}
        c = av.open(str(vp))
        st_v = c.streams.video[0]
        fps = float(st_v.average_rate)
        st_a = c.streams.audio[0] if c.streams.audio else None
        for si, t in enumerate(t8.tolist()):
            for oi in range(-3, 5):
                fr = max(int(round((t + oi) * fps)), 0)
                need.setdefault(fr, []).append((si, oi))
        minfr, maxfr = min(need), max(need)
        c.seek(max(int((minfr - 2) / fps * av.time_base), 0), backward=True)
        frames = {}
        for fr in c.decode(video=0):
            idx = int(round(fr.pts * fr.time_base * fps))
            if idx in need:
                frames[idx] = fr.to_ndarray(format='bgr24')
            if idx >= maxfr:
                break
        rms = seen_src.get(src)
        if rms is None and st_a is not None:
            rms, ok = rms_per_second(c, st_a)
            if ok:
                n_audio += 1
            seen_src[src] = rms if ok else False
        elif st_a is None:
            seen_src[src] = False
        c.close()
        grays = [[None] * 8 for _ in range(8)]
        for si, t in enumerate(t8.tolist()):
            for k, oi in enumerate(range(-3, 5)):
                fr = max(int(round((t + oi) * fps)), 0)
                if fr in frames:
                    grays[si][k] = to_gray_small(frames[fr])
        v = vis_feats(grays)
        sr = seen_src.get(src, False)
        au = aud_feats(sr, t8) if isinstance(sr, np.ndarray) else np.zeros((8, 6), np.float32)
        np.savez(outp, vis=v, aud=au,
                 has_audio=np.uint8(1 if isinstance(sr, np.ndarray) else 0),
                 t=t8.astype(np.float32), fps=np.float32(fps))
        done += 1
        if done % 300 == 0:
            el = time.time() - t0
            print(f'{done}/{len(mine)} {done/el:.2f}/s eta {(len(mine)-done)/(done/el)/60:.0f}min',
                  flush=True)
    except Exception as e:
        errs += 1
        print(f'ERR {w}: {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} errs {errs} audio_sources {n_audio} '
      f'in {(time.time()-t0)/60:.1f}min', flush=True)
