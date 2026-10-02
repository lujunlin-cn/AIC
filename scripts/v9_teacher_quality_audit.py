"""Audit the PHD2 teacher points before distilling on them.

The student can only inherit what the teacher actually knows, so the KD stream's
ceiling is the teacher's signal quality - not its volume.  Four diagnostics,
all computable from the frozen points plus the cached LFM features:

  1. parse rate and degenerate-centre rate, by geometry class.  A class where the
     teacher falls back to (0.5, 0.5) teaches "sit in the middle", which is the
     one thing the candidate head must not learn.
  2. within-fragment temporal stability: consecutive keyframes of one clip are
     ~1 s apart, so the subject point should move smoothly.  Large jumps mean
     the teacher is unstable there, and those rows deserve down-weighting.
  3. agreement with an independent saliency proxy - the LFM token grid's own
     response energy - measured as the rank percentile of the teacher point
     among grid cells.  A teacher that sits on the strongest-response cell is
     more likely to be pointing at the subject than at the frame centre.
  4. the same statistics split by source rotation state, because rotation is the
     only source of portrait geometry in this pool and it is the geometry the
     official drop leans on hardest (33.1% of videos).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True)
ap.add_argument('--points', type=Path, required=True)
ap.add_argument('--feat-root', type=Path, default=None)
ap.add_argument('--out', type=Path, default=None)
ap.add_argument('--limit', type=int, default=1500)
a = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.max_window_path import geometry  # noqa: E402

rows = [json.loads(l) for l in (a.pool / 'index_v2.jsonl').read_text().splitlines() if l.strip()]


def cls(r):
    W, H = int(r['W']), int(r['H'])
    w, h, axis = geometry(W, H, tuple(r['targetRatioWH']))
    tag = 'rot' if r.get('rotated') else 'nat'
    return f"{tag}_{ {None: 'noslide', 0: 'x', 1: 'y'}[axis] }".replace(' ', '')


acc = defaultdict(lambda: {'n': 0, 'pts': [], 'ok': 0, 'degen': 0, 'jumps': [],
                           'rank': [], 'agree_center': []})
n_seen = 0
for r in rows:
    p = a.points / f"{r['video_id']}.json"
    if not p.exists():
        continue
    n_seen += 1
    if n_seen > a.limit:
        break
    rec = json.loads(p.read_text())
    t = rec['ratios']['t']
    pts, st = t['points'], t['status']
    k = cls(r)
    s = acc[k]
    s['n'] += 1
    s['ok'] += sum(1 for x in st if x == 'ok')
    good = [(pt, kf) for pt, x, kf in zip(pts, st, rec['keyframes']) if x == 'ok' and not pt[2]]
    for pt, _ in good:
        s['pts'].append([pt[0], pt[1]])
        if abs(pt[0] - 0.5) < 1e-9 and abs(pt[1] - 0.5) < 1e-9:
            s['degen'] += 1
    if len(good) > 1:
        a_ = np.array([[g[0][0], g[0][1]] for g in good])
        d = np.abs(np.diff(a_, axis=0)).max(1)
        s['jumps'] += d.tolist()
    if a.feat_root:
        for pt, kf in good:
            f = a.feat_root / r['video_id'] / f'{kf}.npz'
            if not f.exists():
                continue
            g = np.load(f)['grid'].astype(np.float32)
            e = np.abs(g).mean(-1)                      # (fh, fw) response energy
            fh, fw = e.shape
            ix = min(fw - 1, max(0, int(pt[0] * fw)))
            iy = min(fh - 1, max(0, int(pt[1] * fh)))
            flat = np.sort(e.reshape(-1))
            v = e[iy, ix]
            s['rank'].append(float(np.searchsorted(flat, v) / max(len(flat), 1)))
            s['agree_center'].append(
                float(abs(pt[0] - 0.5) < abs(e[iy].mean() - e.mean()) and
                      abs(pt[1] - 0.5) < abs(e[:, ix].mean() - e.mean())))

out = {'fragments_scanned': n_seen, 'classes': {}}
for k in sorted(acc):
    s = acc[k]
    P = np.array(s['pts']) if s['pts'] else np.zeros((0, 2))
    J = np.array(s['jumps']) if s['jumps'] else np.zeros(0)
    out['classes'][k] = {
        'fragments': s['n'],
        'parse_ok_rate': round(s['ok'] / max(s['n'] * 8, 1), 4),
        'n_points': int(len(P)),
        'degenerate_centre_rate': round(s['degen'] / max(len(P), 1), 4),
        'x_mean': round(float(P[:, 0].mean()), 4) if len(P) else None,
        'y_mean': round(float(P[:, 1].mean()), 4) if len(P) else None,
        'x_std': round(float(P[:, 0].std()), 4) if len(P) else None,
        'y_std': round(float(P[:, 1].std()), 4) if len(P) else None,
        'consecutive_jump_mean': round(float(J.mean()), 4) if len(J) else None,
        'consecutive_jump_p90': round(float(np.percentile(J, 90)), 4) if len(J) else None,
        'saliency_rank_percentile': round(float(np.mean(s['rank'])), 4) if s['rank'] else None,
        'beats_centre_on_both_axes': (round(float(np.mean(s['agree_center'])), 4)
                                      if s['agree_center'] else None),
    }
print(json.dumps(out, indent=1))
if a.out:
    a.out.write_text(json.dumps(out, indent=1))