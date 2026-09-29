"""Post-hoc multi-person tag for the frozen LFM450 dev manifest (descriptive split only).

A source video is tagged multi_person when >= 30% of its frames carry >= 2
cached COCO person detections (label 1, score >= 0.5) in the OBS cache (the
same FasterRCNN detections already stored for every RetargetVid video).  The
manifest itself is unchanged; this only adds an analysis split, computed after
the run and disclosed as post-hoc.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--run-dir', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()

dev = json.loads(a.manifest.read_text())
tags = {}
for v in dev['videos']:
    z = np.load(a.cache / f"{v['vid']}_1-3.npz")
    det, off = z['det'], z['det_off']
    n = len(off) - 1
    cnt = np.array([int(((det[off[t]:off[t + 1], 5] == 1) & (det[off[t]:off[t + 1], 4] >= .5)).sum()) for t in range(n)])
    tags[v['vid']] = {'multi_person_frame_rate': float((cnt >= 2).mean()), 'multi_person': bool((cnt >= 2).mean() >= .3)}

rows = [json.loads(l) for f in sorted(a.run_dir.glob('rows_s*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
iou = {(r['vid'], r['ratio'], r['policy']): r['iou'] for r in rows if 'iou' in r}
rng = np.random.default_rng(0)


def unit(pol, vids):
    return {v: np.mean([iou[(v, r, pol)] for r in ('1-3', '3-1') if (v, r, pol) in iou]) for v in vids}


out = {'rule': 'multi_person if >=30% frames have >=2 person dets (score>=.5); post-hoc descriptive split', 'tags': tags, 'splits': {}}
for name, vids in (('multi_person', [v for v, t in tags.items() if t['multi_person']]),
                   ('single_or_no_person', [v for v, t in tags.items() if not t['multi_person']])):
    s = {'n_src': len(vids), 'mean_iou': {p: float(np.mean(list(unit(p, vids).values()))) for p in ('CENTER', 'B0', 'LFM_GROUND', 'LFM_GROUND+INTERP', 'QWEN_T')}}
    for p, b in (('LFM_GROUND', 'B0'), ('LFM_GROUND', 'CENTER'), ('LFM_GROUND', 'QWEN_T')):
        A, B = unit(p, vids), unit(b, vids)
        d = np.array([A[v] - B[v] for v in vids])
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        s[f'{p} - {b}'] = {'delta': float(d.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                           'wins': int((d > 1e-9).sum()), 'losses': int((d < -1e-9).sum())}
    out['splits'][name] = s
a.output.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out['splits'], indent=1))
