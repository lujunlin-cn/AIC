"""Semifinal keyframe extraction on the 910A: 1 s grid + B0 shot starts (lfm_venv PyAV).

Grid = every round(fps)-th frame plus every obs-cache reset frame (the MAX_WINDOW
keyframe grid).  PNGs at long side 640 (the spec the head inference path was probed
on).  Also writes points/{vid}.json skeletons with the keyframe list, which
v7_s_official_points.py consumes unchanged.
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--index', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402
from PIL import Image  # noqa: E402

kdir = a.output / 'keyframes'
pdir = a.output / 'points'
kdir.mkdir(parents=True, exist_ok=True)
pdir.mkdir(parents=True, exist_ok=True)
for line in open(a.index):
    r = json.loads(line)
    vid = r['video_id']
    z = np.load(Path(a.cache) / f'{vid}_t.npz')
    reset = z['reset'].astype(bool)
    n = int(r['frame_count'])
    fps = float(r['fps'])
    step = max(int(round(fps)), 1)
    kfs = sorted(set(list(range(0, n, step)) + list(np.where(reset)[0].tolist())))
    kd = kdir / vid
    kd.mkdir(exist_ok=True)
    if not (pdir / f'{vid}.json').exists():
        with av.open(r['video_path']) as c:
            s = c.streams.video[0]
            W0, H0 = s.codec_context.width, s.codec_context.height
        scale = 640.0 / max(W0, H0)
        tw, th = int(round(W0 * scale)), int(round(H0 * scale))
        (pdir / f'{vid}.json').write_text(json.dumps(
            {'video_id': vid, 'W': int(tw), 'H': int(th), 'fps': fps, 'step': step,
             'keyframes': kfs}) + '\n')
    if all((kd / f'{k}.png').exists() for k in kfs):
        continue
    want = set(kfs)
    got, i = 0, 0
    with av.open(r['video_path']) as c:
        s = c.streams.video[0]
        scale = 640.0 / max(s.codec_context.width, s.codec_context.height)
        tw, th = int(round(s.codec_context.width * scale)), int(round(s.codec_context.height * scale))
        for fr in c.decode(video=0):
            if i in want:
                im = cv2.resize(fr.to_ndarray(format='rgb24'), (tw, th))
                Image.fromarray(im).save(kd / f'{i}.png')
                got += 1
            i += 1
    if got != len(kfs) or i != n:
        raise ValueError(f'{vid}: decoded {i} frames (expect {n}), extracted {got}/{len(kfs)}')
    print(vid, len(kfs), flush=True)
print('KFS_DONE', flush=True)
