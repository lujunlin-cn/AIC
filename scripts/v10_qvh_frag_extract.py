"""Extract the 8-frame 1 s grid for QV fragments (PyAV, resumable).

Each fragment is a [t0, t0+L) window inside a 150 s QV clip.  Mirrors
v9_phd2_extract.py: one backward seek to just before t0, sequential decode
through the <=14 s window, match frames to the 1 s grid on pts.  Source lookup
uses the lowercase-first-char bucket the HF mirror actually uses.
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path('/data/aic/external_datasets/QVHighlights')


def _find_video(vid):
    for ch in (vid[0], vid[0].lower()):
        p = ROOT / 'raw' / 'videos' / ch / f'{vid}.mp4'
        if p.exists():
            return p
    return None


def _one(job):
    r, out, n_frames, tol = job
    fd = out / 'frames' / r['video_id']
    if fd.exists() and len(list(fd.glob('*.jpg'))) >= n_frames:
        return 'skip'
    fd.mkdir(parents=True, exist_ok=True)
    src = _find_video(r['src'])
    if src is None:
        return 'missing'
    times = [r['t0'] + (r['L'] * j) / n_frames for j in range(n_frames)]
    got, nxt = {}, 0
    try:
        import av
        c = av.open(str(src))
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        tb = float(s.time_base) if s.time_base else 1.0
        c.seek(int(max(times[0] - 0.5, 0) / tb), stream=s, backward=True)
        t_end = times[-1] + tol
        budget = int((r['L'] + 4.0) * 25 * 2) + 64
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base) if fr.pts is not None else -1.0
            if t > t_end or nxt >= n_frames or budget <= 0:
                break
            budget -= 1
            if t + tol < times[nxt]:
                continue
            got[nxt] = fr.to_ndarray(format='bgr24')
            nxt += 1
        c.close()
    except Exception as e:
        return f'err:{type(e).__name__}'
    import cv2
    for i, t in enumerate(times):
        img = got.get(i)
        if img is None:
            continue
        # filename is the ABSOLUTE time in the source clip so the saliency
        # label slice can be re-derived; v10 frag feats read it back.
        cv2.imwrite(str(fd / f'{t:.3f}.jpg'), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return 'ok' if len(got) == n_frames else f'partial{len(got)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--tol', type=float, default=0.15)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.index.read_text().splitlines() if l.strip()]
    rows = rows[a.shard::a.nshards]
    print(f'QV_FRAG_EXTRACT shard {a.shard}/{a.nshards}: {len(rows)}', flush=True)
    jobs = [(r, a.out, a.frames, a.tol) for r in rows]
    t0, done = time.time(), {}
    with Pool(a.workers) as p:
        for i, st in enumerate(p.imap_unordered(_one, jobs, chunksize=4)):
            done[st] = done.get(st, 0) + 1
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f'PROGRESS {i+1}/{len(jobs)} {el:.0f}s eta={el/(i+1)*(len(jobs)-i-1):.0f}s {done}', flush=True)
    print(f'DONE {done} wall={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
