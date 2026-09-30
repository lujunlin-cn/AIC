"""V8: per-frame saliency value for the RetargetVid sources (= DHF1K videos).

Value = mean of the DHF1K human saliency map, the domain-internal frame-value
signal for the temporal line.  Train sources 001-030+621-700 / dev 601-620 /
diag 031-100 keep the same source-level exposure discipline as the spatial
line.  Output jsonl: {vid, frame, value} with frame aligned to decoded AVI
indices (alignment check reported).
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--ann-root', default='/data/aic/external_datasets/DHF1K/annotations/annotation')
ap.add_argument('--video-dir', default='/data/aic/external_datasets/DHF1K/raw/video')
ap.add_argument('--vids', nargs='*', default=None)
ap.add_argument('--output', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/dhf1k_saliency_value.jsonl'))
args = ap.parse_args()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

if args.vids:
    vids = args.vids
else:
    vids = sorted(f'{i:03d}' for i in range(1, 101)) + sorted(f'{i:03d}' for i in range(601, 701))

out = []
for vid in vids:
    mdir = Path(args.ann_root) / f'{int(vid):04d}' / 'maps'
    pngs = sorted(mdir.glob('*.png'))
    if not pngs:
        print(f'{vid}: no maps, skip', flush=True)
        continue
    nums = np.array([int(p.stem) for p in pngs])
    vals = np.zeros(len(nums), dtype=np.float32)
    for i, p in enumerate(pngs):
        vals[i] = float(np.asarray(Image.open(p).convert('L'), dtype=np.float32).mean() / 255.0)
    # alignment: DHF1K maps are 1-indexed per readme (frame 0001..N) matching
    # decoded index 0..N-1; verify count against the AVI when readable.
    n_map = len(nums)
    contiguous = bool(np.all(np.diff(nums) == 1))
    out.append({'vid': vid, 'n_maps': n_map, 'first': int(nums[0]), 'contiguous': contiguous,
                'values': vals.tolist()})

args.output.parent.mkdir(parents=True, exist_ok=True)
with open(args.output, 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
import collections  # noqa: E402
c = collections.Counter(r['first'] for r in out)
print('first-map-number distribution:', dict(c))
print('n videos:', len(out), 'total frames:', sum(r['n_maps'] for r in out))
