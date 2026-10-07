"""R10 family-T coverage probe (R10 4.1/4.2): OCR on the frozen slot anchors.

Protocol: decode the 8 anchor frames per fragment (same seek discipline as
R9/R10 video extraction), run PaddleOCR (CPU) per frame, and store per-slot
text statistics.  First pass is a COVERAGE audit on the pilot sources (R10:
test on non-official images before any fps forecast): how many anchors carry
machine-readable text at all, how much of it is digits (scoreboards), and
whether text CHANGES between adjacent slots (the state-change signal of
family T).

Sharded, one json per fragment (resume-safe).  CPU only.  No official media.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import av
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
ap.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
ap.add_argument('--pool-feats', default='/data/aic/experiments_910a/LFM_V10/pool_feats')
ap.add_argument('--sources', default='/data/aic/experiments_910a/LFM_V11/r10_audio/pilot_sources_64.txt')
ap.add_argument('--limit', type=int, default=13, help='first N pilot frags (13*8=104 anchors)')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r10_ocr')
a = ap.parse_args()

# Engine note: R10 4.2 prefers PaddleOCR-mobile/EasyOCR, but paddle SIGSEGVs
# in this Ascend container (its NPU probe collides with CANN 9.0.0) and
# EasyOCR weights live on GitHub releases (unreachable here).  Tesseract 4.1
# is the fallback; deviation recorded in reports/r10/evidence_ledger.md.
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
print('tesseract ready', flush=True)

man = []
for split in ('train_public', 'eval_public'):
    for l in (Path(a.manifest_dir) / f'{split}.jsonl').read_text().splitlines():
        if l.strip():
            man.append(json.loads(l))
srcs = {r['fragment_id']: r['source_id'] for r in man}
keep = set(Path(a.sources).read_text().split())
man = [r for r in man if r['source_id'] in keep]
fids = [r['fragment_id'] for r in man]
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard][:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} frags', flush=True)
MEDIA = Path(a.media_root)
POOL = Path(a.pool_feats)


def read_anchor_frames(vid_path, t8):
    """8 anchor frames at slot centres (integer-second t, R9 index rule)."""
    c = av.open(str(vid_path))
    try:
        st = c.streams.video[0]
        fps = float(st.average_rate)
        need = {max(int(round(t * fps)), 0): si for si, t in enumerate(t8.tolist())}
        c.seek(max(int((float(t8.min()) - 2.0) * av.time_base), 0), backward=True)
        imgs = {}
        for fr in c.decode(video=0):
            idx = int(round(fr.pts * fr.time_base * fps))
            if idx in need:
                imgs[need[idx]] = cv2.resize(fr.to_ndarray(format='rgb24'),
                                             (640, 360), interpolation=cv2.INTER_AREA)
            if idx >= max(need):
                break
    finally:
        c.close()
    return [imgs.get(si) for si in range(8)]


t0 = time.time(); done = 0; errs = 0
for w in mine:
    outp = OUTP / f'{w}.json'
    if outp.exists():
        done += 1
        continue
    try:
        t8 = np.load(POOL / f'{w}.npz')['t'].astype(np.float64)
        frames = read_anchor_frames(MEDIA / f'{srcs[w]}.mp4', t8)
        slots = []
        for img in frames:
            if img is None:
                slots.append({'n_text': 0, 'n_chars': 0, 'n_digits': 0, 'texts': []})
                continue
            d = pytesseract.image_to_data(img, output_type=Output.DICT,
                                          config='--psm 11')
            texts = [t for t, c in zip(d['text'], d['conf'])
                     if t.strip() and int(c) > 40]
            joined = ' '.join(texts)
            slots.append({'n_text': len(texts), 'n_chars': len(joined),
                          'n_digits': sum(ch.isdigit() for ch in joined),
                          'texts': texts[:10]})
        outp.write_text(json.dumps({'fragment_id': w, 'slots': slots}))
        done += 1
        if done % 4 == 0:
            print(f'{done}/{len(mine)} {(time.time()-t0)/60:.1f}min', flush=True)
    except Exception as e:
        errs += 1
        print(f'ERR {w}: {type(e).__name__} {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} errs {errs} in {(time.time()-t0)/60:.1f}min', flush=True)
