"""Quantify the run-to-run point jitter of the student inference path.

The vision-only path reproduced the full-model path to within 0.005 of a
normalised coordinate, but 910A inference is not bit-reproducible, so the
question that matters is whether that spread is random jitter (harmless, it
averages out over 234k predictions) or a systematic offset in one direction
(an actual behaviour change).  This compares two independent runs of the same
script on different cards: if they differ by a similar magnitude with both
signs, the vision-only difference is inside the noise floor.
"""
import json
from pathlib import Path

import numpy as np

A = Path('/data/aic/semifinal_20261001/points_VISIONONLY_test/points')
B = Path('/data/aic/semifinal_20261001/points_VISIONONLY_test2/points')
R = Path('/data/aic/semifinal_20261001/points_V9_A1raw/points')

rows = []
for f in sorted(A.glob('*.json')):
    g = B / f.name
    r = R / f.name
    if not g.exists() or not r.exists():
        continue
    pa = json.loads(f.read_text())['ratios']['t']['points']
    pb = json.loads(g.read_text())['ratios']['t']['points']
    pr = json.loads(r.read_text())['ratios']['t']['points']
    if not (len(pa) == len(pb) == len(pr)):
        continue
    d_run = np.array([[x[0] - y[0], x[1] - y[1]] for x, y in zip(pa, pb)])
    d_ref = np.array([[x[0] - y[0], x[1] - y[1]] for x, y in zip(pa, pr)])
    rows.append((d_run, d_ref))

if not rows:
    raise SystemExit('no overlapping videos')
run = np.concatenate([d for d, _ in rows])
ref = np.concatenate([d for _, d in rows])
print('videos compared: %d, points: %d' % (len(rows), len(run)))
for name, d in (('visiononly_runA - runB (same script, card0 vs card1)', run),
                ('visiononly_runA - fullmodel_reference', ref)):
    mag = np.abs(d).max(1)
    print(f'{name}:')
    print(f'   mean|dx|={np.abs(d[:,0]).mean():.5f}  mean|dy|={np.abs(d[:,1]).mean():.5f}  '
          f'mean max-axis={mag.mean():.5f}  p95={np.percentile(mag,95):.5f}  '
          f'max={mag.max():.5f}')
    print(f'   signed mean dx={d[:,0].mean():+.6f} dy={d[:,1].mean():+.6f}  '
          f'(near zero => jitter, not a shift)')
    print(f'   exactly equal: {int((mag == 0).sum())}/{len(mag)}')