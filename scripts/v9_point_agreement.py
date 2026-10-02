"""Sanity-check the V9 student inference against the V8 head and the teacher.

A retrained head must move predictions, but not wildly: if the new points sit in
a different part of the frame than both the previous head and the teacher, the
likely cause is an inference-path bug (feature order, candidate geometry, ratio
handling) rather than a real gain.  This compares the three point sets on the
same 426 videos, per geometry class, and reports how far the new head moved and
whether it moved toward or away from the teacher.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, default=Path('/data/aic/semifinal_20261001/intake/index.enriched.jsonl'))
ap.add_argument('--new', type=Path, required=True, help='points dir of the new head')
ap.add_argument('--old', type=Path, required=True, help='points dir of the previous head')
ap.add_argument('--teacher', type=Path, required=True)
ap.add_argument('--out', type=Path, default=None)
a = ap.parse_args()

meta = {json.loads(l)['video_id']: json.loads(l)
        for l in a.index.read_text().splitlines() if l.strip()}


def load(d, vid):
    p = Path(d) / f'{vid}.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    t = r['ratios'].get('t')
    if not t:
        return None
    pts = [x for x in t['points'] if not x[2]]
    if not pts:
        return None
    return np.array([[x[0], x[1]] for x in pts])


acc = defaultdict(lambda: {'n': 0, 'new': [], 'old': [], 'tea': [],
                           'd_new_old': [], 'd_new_tea': [], 'd_old_tea': []})
for vid, m in meta.items():
    n, o, t = load(a.new, vid), load(a.old, vid), load(a.teacher, vid)
    if n is None or o is None or t is None:
        continue
    W, H = m['width'], m['height']
    ar = W / H
    g = 'landscape' if ar > 1.5 else ('portrait' if ar < 0.75 else 'square')
    rw, rh = m['targetRatioWH']
    key = f"{g}->{rw}:{rh}"
    k = min(len(n), len(o), len(t))
    n, o, t = n[:k], o[:k], t[:k]
    s = acc[key]
    s['n'] += 1
    s['new'] += n.tolist()
    s['old'] += o.tolist()
    s['tea'] += t.tolist()
    s['d_new_old'] += np.abs(n - o).max(1).tolist()
    s['d_new_tea'] += np.abs(n - t).max(1).tolist()
    s['d_old_tea'] += np.abs(o - t).max(1).tolist()

out = {'classes': {}}
for k in sorted(acc):
    s = acc[k]
    dn, do = np.mean(s['d_new_old']), np.mean(s['d_old_tea'])
    out['classes'][k] = {
        'videos': s['n'],
        'mean_abs_move_vs_V8_head': round(float(dn), 4),
        'mean_abs_dist_new_to_teacher': round(float(np.mean(s['d_new_tea'])), 4),
        'mean_abs_dist_V8_to_teacher': round(float(do), 4),
        'moved_toward_teacher': bool(np.mean(s['d_new_tea']) < do),
        'new_x_std': round(float(np.array(s['new'])[:, 0].std()), 4),
        'new_y_std': round(float(np.array(s['new'])[:, 1].std()), 4),
    }
print(json.dumps(out, indent=1))
if a.out:
    a.out.write_text(json.dumps(out, indent=1))