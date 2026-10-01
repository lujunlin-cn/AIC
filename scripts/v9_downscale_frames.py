"""Downscale fragment frames to the deployment resolution (longest side 640).

The teacher run on raw 1280x720 PHD2 frames delivered 0.36 qps versus 0.73 qps
on the semifinal keyframes.  The difference is not the teacher: the semifinal
keyframes are stored at longest side 640 (640x360 / 360x640), while PHD2 media
is stored at native 1280x720.  Vision tokens scale with area, so 1280x720 costs
~4x the tokens of 640x360.

Resizing to 640 therefore does two things at once:
  * restores the production throughput (roughly 2x measured, bounded by the
    ViT token count rather than by anything else);
  * removes a resolution domain gap - the student head would otherwise be
    trained on 720p frames and deployed on 640-longest-side frames.

In-place, resumable, multiprocess.  Files already at the target size are skipped.
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

cv2.setNumThreads(1)


def _one(job):
    f, max_side, q = job
    im = cv2.imread(str(f))
    if im is None:
        return 'unreadable'
    h, w = im.shape[:2]
    if max(h, w) <= max_side:
        return 'already'
    s = max_side / max(h, w)
    out = cv2.resize(im, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(f), out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return 'resized'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--max-side', type=int, default=640)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--quality', type=int, default=90)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    files = sorted((a.pool / 'frames').rglob('*.jpg'))
    if a.limit:
        files = files[:a.limit]
    print(f'DOWNSCALE start n={len(files)} max_side={a.max_side}', flush=True)
    t0 = time.time()
    done = {}
    with Pool(a.workers) as p:
        for i, st in enumerate(p.imap_unordered(
                _one, [(f, a.max_side, a.quality) for f in files], chunksize=64)):
            done[st] = done.get(st, 0) + 1
            if (i + 1) % 5000 == 0:
                el = time.time() - t0
                print(f'PROGRESS {i + 1}/{len(files)} {el:.0f}s {done}', flush=True)
    print(f'DOWNSCALE_DONE wall={time.time() - t0:.0f}s {done}', flush=True)


if __name__ == '__main__':
    main()