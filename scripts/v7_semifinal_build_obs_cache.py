"""Semifinal-ready obs-cache builder for the 910A: CPU YuNet path, FasterRCNN optional.

Same npz schema as scripts.build_obs_cache (aic.obs_cache.save_cache), but the
FasterRCNN detection column is skipped entirely (policy QWEN_POINT / the V7 student
head chain never reads it), so the whole step runs on CPU (cv2 YuNet + PyAV) on the
910A where CUDA is unavailable.
"""
import argparse, json, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--index', required=True)
ap.add_argument('--part', type=int, default=0)
ap.add_argument('--parts', type=int, default=1)
ap.add_argument('--face', required=True)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()

import cv2  # noqa: E402
cv2.setNumThreads(4)
import av  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.obs_cache import ObservedFacePath, save_cache  # noqa: E402
from aic.video import _decoded  # noqa: E402
from aic.inference import _record_path  # noqa: E402

records = [json.loads(l) for l in open(a.index)]
jobs = [(r['video_id'], _record_path(r, None), {'t': r['targetRatioWH']},
         (r['width'], r['height'], r['frame_count'])) for r in records]
jobs = jobs[a.part::a.parts]
a.output.mkdir(parents=True, exist_ok=True)
start, done = time.time(), []
for vid, path, ratios, meta in jobs:
    t0 = time.time()
    if all((a.output / f'{vid}_{k}.npz').exists() for k in ratios):
        continue
    paths = {k: ObservedFacePath(r, a.face) for k, r in ratios.items()}
    crops = {k: [] for k in ratios}
    obs = {k: [] for k in ratios}
    dets = []
    n, buf = 0, []

    def flush(buf):
        for f in buf:
            dets.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)))
            for k, p in paths.items():
                c, o = p.step(f)
                crops[k].append(c)
                obs[k].append(o)

    for stamp, frame in _decoded(path):
        if stamp.index != n:
            raise ValueError('non-contiguous frame index')
        buf.append(frame.to_ndarray(format='rgb24'))
        n += 1
        if len(buf) == a.batch:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        W, H = s.codec_context.width, s.codec_context.height
    if meta and (W, H, n) != meta:
        raise ValueError(f'{vid}: metadata mismatch {(W, H, n)} vs {meta}')
    for k, r in ratios.items():
        save_cache(a.output / f'{vid}_{k}.npz', W, H, r, crops[k], obs[k], dets)
    done.append({'video_id': vid, 'frames': n, 'seconds': round(time.time() - t0, 1)})
    print(json.dumps({'part': a.part, 'done': len(done), 'total': len(jobs), 'vid': vid,
                      'wall': round(time.time() - start, 1)}), flush=True)
(a.output / f'part{a.part}_of{a.parts}.json').write_text(json.dumps(
    {'part': a.part, 'parts': a.parts, 'videos': done, 'wall_seconds': round(time.time() - start, 1),
     'mode': 'cpu_yunet_only_qwen_point_schema'}, indent=1) + '\n')
print('DONE', a.part, len(done), flush=True)
