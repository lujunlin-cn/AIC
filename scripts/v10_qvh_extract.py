"""Frame extraction for QVHighlights on the 1 Hz grid (PyAV, resumable).

QVHighlights videos are pre-cut 150 s clips stored under
raw/videos/<bucket>/<ytid>_<start>_<end>.mp4.  Unlike the PHD2 fragments we
cannot decode a short window - the temporal head needs features across the
WHOLE clip on a 1 s grid, so this decodes the file once and keeps the frame
nearest each integer second.

A clip of ~150 s at ~25 fps is ~3750 decode steps but only ~150 writes; the
decode is the cost, not the I/O.  Frames land under <out>/frames/<vid>/ as
<t>.jpg so v9_phd2_lfm_feats.py can pool them unchanged (it reads
index.jsonl for video_id + the frames/ tree, both of which we emit).

Resumable: a video counts as done when its frame count reaches the expected
1 Hz grid length, so re-running fills only the gaps.
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path('/data/aic/external_datasets/QVHighlights')


def _find_video(vid):
    # The HF mirror buckets each clip under the LOWERCASED first character of
    # its filename (videos/<char>/<ytid>_<start>_<end>.mp4) while the filename
    # itself keeps its original case - so 'NUs...mp4' lives under 'n/'.
    # Try both casings to be safe.
    for ch in (vid[0], vid[0].lower()):
        p = ROOT / 'raw' / 'videos' / ch / f'{vid}.mp4'
        if p.exists():
            return p
    return None


def _one(job):
    r, out = job
    vid = r['vid']
    fd = out / 'frames' / vid
    dur = float(r['duration'])
    n_frames = int(round(dur))          # one frame per second
    if fd.exists() and len(list(fd.glob('*.jpg'))) >= n_frames - 1:
        return 'skip'
    fd.mkdir(parents=True, exist_ok=True)
    src = _find_video(vid)
    if src is None:
        return 'missing'
    # target times: one per integer second across the clip
    times = np.arange(0.0, dur, 1.0)
    got, nxt = {}, 0
    try:
        import av
        c = av.open(str(src))
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        # 1 Hz grid needs the whole clip: sequential decode, keep the frame
        # nearest each integer second.  No seek - we start at t=0 anyway.
        last_t, last_img = -1e9, None
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base) if fr.pts is not None else -1.0
            if nxt >= len(times):
                break
            # keep the frame whose pts is closest to the grid point
            while nxt < len(times) and t >= times[nxt] - 0.5:
                if last_img is None or abs(t - times[nxt]) <= abs(last_t - times[nxt]):
                    got[nxt] = fr.to_ndarray(format='bgr24')
                else:
                    got[nxt] = last_img
                nxt += 1
            last_t, last_img = t, fr.to_ndarray(format='bgr24')
        # flush trailing grid points
        while nxt < len(times) and last_img is not None:
            got[nxt] = last_img
            nxt += 1
        c.close()
    except Exception as e:  # noqa: BLE001
        return f'err:{type(e).__name__}'
    import cv2
    for i, t in enumerate(times):
        img = got.get(i)
        if img is None:
            continue
        cv2.imwrite(str(fd / f'{t:.3f}.jpg'), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return 'ok' if len(got) >= n_frames - 1 else f'partial{len(got)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    rows = {}
    for line in a.annotations.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['vid']] = {'vid': r['vid'], 'duration': r['duration']}
    recs = sorted(rows.values(), key=lambda x: x['vid'])
    recs = recs[a.shard::a.nshards]
    if a.limit:
        recs = recs[:a.limit]
    (a.out / 'frames').mkdir(parents=True, exist_ok=True)
    # emit the index the feature extractor reads
    with (a.out / 'index.jsonl').open('a') as fh:
        for r in recs:
            fh.write(json.dumps({'video_id': r['vid'], 'duration': r['duration']}) + '\n')
    print(f'QV_EXTRACT shard {a.shard}/{a.nshards}: {len(recs)} videos', flush=True)
    jobs = [(r, a.out) for r in recs]
    t0, done = time.time(), {}
    with Pool(a.workers) as p:
        for i, st in enumerate(p.imap_unordered(_one, jobs, chunksize=2)):
            done[st] = done.get(st, 0) + 1
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f'PROGRESS {i+1}/{len(jobs)} {el:.0f}s eta={el/(i+1)*(len(jobs)-i-1):.0f}s {done}', flush=True)
    print(f'QV_EXTRACT_DONE {done} wall={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
