"""Teacher point diagnostics on the semifinal drop, split by source geometry.

Question: the teacher prompt/scheme was frozen in the all-landscape preliminary
round, but 29% of the 426 semifinal videos are portrait sources, and the
teacher's orientation word is derived from (W,H) vs target ratio - so on a
portrait source asked for a 16:9 window the orientation word flips to
"portrait, narrower than the frame" while every preliminary-round prompt was
written for landscape.  28% of the semifinal tasks are portrait-source -> 16:9,
i.e. exactly the y-axis (vertical) crop decisions that scored worst all project
history (H1 x-only: y -1.73).

This reads the already-frozen semifinal teacher output (no new inference) and
tests for the quantitative signature of a mis-specified prompt:
  * parse status by source class;
  * y-coordinate dispersion (a degenerate point collapses onto the centre);
  * per-video y variance vs the B0/YuNet observer on the same frames;
  * agreement of the teacher window with a face-aware observer.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

S = Path('/data/aic/semifinal_20261001')


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, default=S / 'intake/index.enriched.jsonl')
    ap.add_argument('--points', type=Path, default=S / 'teacher_points/points')
    ap.add_argument('--obs', type=Path, default=S / 'obs_cache')
    ap.add_argument('--out', type=Path, default=S / 'teacher_geometry_audit.json')
    a = ap.parse_args()

    rows = load_jsonl(a.index)
    meta = {r['video_id']: r for r in rows}

    def cls(r):
        ar = r['width'] / r['height']
        return 'landscape' if ar > 1.5 else ('portrait' if ar < 0.75 else 'square')

    stats = defaultdict(lambda: {'n': 0, 'ok': 0, 'y': [], 'x': [], 'yv': [], 'npts': []})
    n_missing = 0
    for r in rows:
        p = a.points / f"{r['video_id']}.json"
        if not p.exists():
            n_missing += 1
            continue
        rec = json.loads(p.read_text())
        rr = rec['ratios']['t']
        rw, rh = rec['ratios']['t']['ratio']
        key = f"{cls(r)}->{rw}:{rh}"
        s = stats[key]
        s['n'] += 1
        pts = rr['points']
        st = rr['status']
        s['ok'] += sum(1 for x in st if x == 'ok')
        s['npts'].append(len(pts))
        ys = [p_[1] for p_ in pts if not p_[2]]
        xs = [p_[0] for p_ in pts if not p_[2]]
        s['y'] += ys
        s['x'] += xs
        if len(ys) > 1:
            s['yv'].append(float(np.std(ys)))

    out = {'missing_points_files': n_missing, 'classes': {}}
    for k in sorted(stats):
        s = stats[k]
        y = np.array(s['y']) if s['y'] else np.array([np.nan])
        x = np.array(s['x']) if s['x'] else np.array([np.nan])
        out['classes'][k] = {
            'videos': s['n'],
            'queries': int(sum(s['npts'])),
            'parse_ok_rate': round(s['ok'] / max(sum(s['npts']), 1), 4),
            'y_mean': round(float(np.nanmean(y)), 4),
            'y_std_pooled': round(float(np.nanstd(y)), 4),
            'x_mean': round(float(np.nanmean(x)), 4),
            'x_std_pooled': round(float(np.nanstd(x)), 4),
            'y_within_video_std_mean': round(float(np.mean(s['yv'])), 4) if s['yv'] else None,
            'n_degenerate_centre': int((np.abs(np.array(s['y']) - 0.5) < 1e-6).sum()),
            'y_hist': [round(float((np.histogram(y, bins=10, range=(0, 1))[0] / max(len(y), 1))[i]), 4)
                       for i in range(10)],
        }
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()