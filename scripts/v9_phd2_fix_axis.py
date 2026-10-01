"""Re-target the fragments whose window has no sliding axis.

`geometry` returns axis=None whenever the source aspect equals the target
aspect, which means the crop IS the whole frame: one candidate, zero spatial
decision, and - measured on this pool - 54.2% of the fragments (3,191 of 5,886).
Training a candidate-utility head on those rows teaches it nothing, and paying
the 32B teacher for their points is worse than wasted.

The official drop does contain such videos (37.3% by the sliding-axis audit),
but they are won or lost on the temporal row alone, so mirroring that share in
the spatial training pool is actively harmful.  Those rows are re-targeted to a
ratio whose window can actually slide:

    landscape / square source -> 9:16  (x axis)
    portrait source            -> 16:9  (y axis)

The rotated pseudo-portrait copies already exercise the y axis, so this mostly
converts dead horizontal rows into x-axis rows.  original_target is kept for
provenance.  Points computed under the old target are NOT reusable: the teacher
prompt embeds the ratio, so a 16:9 point is not a 9:16 point.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.max_window_path import geometry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True)
ap.add_argument('--in-name', default='index_clean.jsonl')
ap.add_argument('--out-name', default='index_v2.jsonl')
ap.add_argument('--report', type=Path, default=None)
a = ap.parse_args()

rows = [json.loads(l) for l in (a.pool / a.in_name).read_text().splitlines() if l.strip()]
out, changed = [], 0
before, after = Counter(), Counter()
for r in rows:
    W, H = int(r['W']), int(r['H'])
    rw, rh = r['targetRatioWH']
    _, _, axis = geometry(W, H, (rw, rh))
    before['axis_None' if axis is None else f'axis_{axis}'] += 1
    if axis is None:
        if W > H:
            rw, rh = 9, 16
        else:
            rw, rh = 16, 9
        r = dict(r, original_targetRatioWH=r['targetRatioWH'],
                 targetRatioWH=[rw, rh], retargeted=True)
        changed += 1
    _, _, ax2 = geometry(W, H, (rw, rh))
    after['axis_None' if ax2 is None else f'axis_{ax2}'] += 1
    out.append(r)

(a.pool / a.out_name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
rep = {'rows': len(rows), 'retargeted': changed,
       'axis_before': dict(before), 'axis_after': dict(after),
       'rule': 'axis=None -> landscape/square use 9:16, portrait uses 16:9',
       'note': ('teacher points computed under the previous target are not reusable; '
                'the prompt embeds the ratio')}
print(json.dumps(rep, indent=1))
if a.report:
    a.report.write_text(json.dumps(rep, indent=1))