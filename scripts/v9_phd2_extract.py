"""Frame extraction for the PHD2 fragment pool (PyAV, multiprocess, resumable).

Decoder: PyAV, not OpenCV.  35% of the PHD2 media is AV1 (4,667/13,339) and the
OpenCV/FFmpeg build in this image has no libdav1d, so cv2.VideoCapture returns
zero frames on those files.  PyAV 18 links libdav1d and decodes all three
codecs present (h264 68%, av1 35%, vp9 1%).

Decode strategy: one backward seek to just before the fragment start, then
sequential decode through the <=14 s window (a mid-file ms-seek makes libav
decode from the nearest keyframe, so 8 scattered seeks cost far more than 1).
Frames are matched on presentation timestamp against the 1 s keyframe grid.

Rotated pseudo-portrait copies (`*_r`, 90 deg CW, W/H swapped) come out of the
same decode pass, so that channel is free.
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

D = Path('/data/aic/external_datasets/PHD2')


def _one(job):
    idx, out, n_frames, tol = job
    r = idx
    fd = out / 'frames' / r['video_id']
    if len(list(fd.glob('*.jpg'))) >= n_frames:
        return 'skip'
    fd.mkdir(parents=True, exist_ok=True)
    times = [r['t0'] + (r['L'] * j) / n_frames for j in range(n_frames)]
    rot = bool(r.get('rotated'))
    got, nxt = {}, 0
    try:
        import av
        c = av.open(str(D / 'raw' / 'youtube' / f"{r['src']}.mp4"))
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        # container.seek() with an explicit stream takes the offset in that
        # stream's time_base TICKS, i.e. offset = seconds / time_base.  With
        # time_base 1/12800 the two directions differ by 1.6e8: multiplying
        # instead of dividing truncates every offset to 0 and the decode then
        # restarts at t=0, so the fragment window comes back empty.
        tb = float(s.time_base) if s.time_base else av.time_base
        c.seek(int(max(times[0] - 0.5, 0) / tb), stream=s, backward=True)
        t_end = times[-1] + tol
        budget = int((r['L'] + 4.0) * (r.get('fps') or 25) * 2) + 64
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base) if fr.pts is not None else -1.0
            if t > t_end or nxt >= n_frames or budget <= 0:
                break
            budget -= 1
            if t + tol < times[nxt]:
                continue
            img = fr.to_ndarray(format='bgr24')
            if rot:
                img = np.ascontiguousarray(np.rot90(img, k=-1))
            got[nxt] = img
            nxt += 1
        c.close()
    except Exception as e:  # noqa: BLE001 - a bad file must not kill the pool
        return f'err:{type(e).__name__}'
    import cv2
    for i, t in enumerate(times):
        img = got.get(i)
        if img is None:
            continue
        cv2.imwrite(str(fd / f'{t:.3f}.jpg'), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return 'ok' if len(got) == n_frames else f'partial{len(got)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--tol', type=float, default=0.12)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (a.pool / 'index.jsonl').read_text().splitlines() if l.strip()]
    rows = rows[a.shard::a.nshards]
    print(f'EXTRACT start rows={len(rows)} workers={a.workers}', flush=True)
    jobs = [(r, a.pool, a.frames, a.tol) for r in rows]
    t0 = time.time()
    done = {}
    with Pool(a.workers) as p:
        for i, st in enumerate(p.imap_unordered(_one, jobs, chunksize=4)):
            done[st] = done.get(st, 0) + 1
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f'PROGRESS {i + 1}/{len(jobs)} {el:.0f}s '
                      f'eta={el / (i + 1) * (len(jobs) - i - 1):.0f}s {done}', flush=True)
    n_jpg = len(list((a.pool / 'frames').rglob('*.jpg')))
    print(f'EXTRACT_DONE jpg={n_jpg} wall={time.time() - t0:.0f}s {done}', flush=True)


if __name__ == '__main__':
    main()