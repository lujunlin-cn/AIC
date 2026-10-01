"""Semifinal drop decomposed by what each task actually asks for.

The per-source x/y split used so far counted source orientation, not the
decision the task requires.  What matters is the sliding axis returned by
aic.max_window_path.geometry:

  axis=None -> the target ratio equals the source ratio, the window IS the whole
               frame, so the crop carries no information at all and the score of
               such a video is decided purely by which frames are kept;
  axis=0    -> x-axis crop (portrait window sliding horizontally);
  axis=1    -> y-axis crop (landscape window sliding vertically).

This also quantifies how much headroom the temporal row still has: the temporal
decision applies to 100% of videos while the spatial decision applies to only
the subset with a free axis.
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.max_window_path import geometry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/semifinal_20261001/intake/index.enriched.jsonl'))
ap.add_argument('--out', type=Path, default=None)
a = ap.parse_args()

rows = [json.loads(l) for l in a.index.read_text().splitlines() if l.strip()]
by = Counter()
span_frac = defaultdict(list)
frames_by = Counter()
for r in rows:
    W, H = int(r['width']), int(r['height'])
    rw, rh = r['targetRatioWH']
    w, h, axis = geometry(W, H, (rw, rh))
    key = {None: 'no_slide(full_frame)', 0: 'x_axis', 1: 'y_axis'}[axis]
    by[key] += 1
    frames_by[key] += int(r['frame_count'])
    if axis is not None:
        span = (W - w) if axis == 0 else (H - h)
        span_frac[key].append(span / (W if axis == 0 else H))

out = {
    'videos': len(rows),
    'by_sliding_axis': dict(by),
    'frames_by_sliding_axis': dict(frames_by),
    'share': {k: round(v / len(rows), 4) for k, v in by.items()},
    'mean_travel_fraction': {k: round(sum(v) / len(v), 4) for k, v in span_frac.items()},
    'reading': {
        'no_slide': ('source ratio equals target ratio; every method emits the full '
                     'frame, so spatial IoU on GT frames is 1 by construction and the '
                     'score of these videos is decided entirely by the temporal row'),
        'x_axis': 'portrait window sliding horizontally',
        'y_axis': 'landscape window sliding vertically',
    },
}
print(json.dumps(out, indent=1, ensure_ascii=False))
if a.out:
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))