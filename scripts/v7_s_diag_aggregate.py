"""V7 S-line: aggregate per-keyframe head results into the frozen DIAG table.

Paired per-source-video (mean of both ratios) deltas + 10k bootstrap CI for:
head vs CENTER / B0 (recomputed on the same keyframes from the obs cache) /
the raw-LFM zero-shot row / the teacher-point row (from the previous round's
rows_s*.jsonl).  Layer splits from the frozen dev manifest; multi-person tag
post-hoc (same rule as lfm450_multiperson_split.py).
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--per', type=Path, required=True, help='per_diag_031_100.jsonl from v7_s_train_head')
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--ann', type=Path, default=Path('/data/aic/experiments_910a/LFM450_EVAL_V1/annotations'))
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments_910a/LFM450_EVAL_V1/dev_manifest_v1.json'))
ap.add_argument('--prev-rows', type=Path, nargs='*', default=[], help='previous round rows_s*.jsonl (LFM_GROUND / QWEN_T policies)')
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from scripts.max_window_path_eval import RATIOS, boxes, load_gt  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

per = [json.loads(l) for l in args.per.read_text().splitlines() if l.strip()]
dev = json.loads(args.manifest.read_text())
layer = {v['vid']: v['layer'] for v in dev['videos']}

# B0 IoU on the same keyframes
b0_iou = {}
for vid in sorted({p['vid'] for p in per}):
    for r in RATIOS:
        z = np.load(args.cache / f'{vid}_{r}.npz')
        gt = load_gt(args.ann, vid, r)
        c = np.asarray(z['b0'], float)
        bx = boxes(c, RATIOS[r])
        for row in per:
            if row['vid'] == vid and row['ratio'] == r:
                b0_iou[(vid, r, row['kf'])] = float(iou(bx[row['kf']][None], gt[:, row['kf']]).mean())

# previous round per-(vid,ratio) policy IoU (frame-averaged) if provided
prev = {}
for f in args.prev_rows:
    for line in f.read_text().splitlines():
        rr = json.loads(line)
        if 'iou' in rr:
            prev[(rr['vid'], rr['ratio'], rr['policy'])] = rr['iou']

# source-video units
vids = sorted({p['vid'] for p in per})
units = {}
for v in vids:
    pv = [p for p in per if p['vid'] == v]
    units[v] = {
        'head': float(np.mean([p['iou'] for p in pv])),
        'center': float(np.mean([p['center'] for p in pv])),
        'best': float(np.mean([p['best'] for p in pv])),
        'b0': float(np.mean([b0_iou[(v, p['ratio'], p['kf'])] for p in pv])),
        'axis': {p['ratio']: p['iou'] for p in pv},
        'layer': layer.get(v, '?'),
    }
for pol in ('LFM_GROUND', 'QWEN_T'):
    vv = [v for v in vids if all((v, r, pol) in prev for r in RATIOS)]
    for v in vv:
        units[v][pol] = float(np.mean([prev[(v, r, pol)] for r in RATIOS]))

# multi-person tag (same rule as lfm450_multiperson_split)
tags = {}
for v in vids:
    z = np.load(args.cache / f'{v}_1-3.npz')
    det, off = z['det'], z['det_off']
    n = len(off) - 1
    cnt = np.array([int(((det[off[t]:off[t + 1], 5] == 1) & (det[off[t]:off[t + 1], 4] >= .5)).sum()) for t in range(n)])
    tags[v] = bool((cnt >= 2).mean() >= .3)

rng = np.random.default_rng(0)


def paired(a, b, sel=None):
    vs = [v for v in vids if (sel is None or sel(v)) and a in units[v] and b in units[v]]
    if len(vs) < 5:
        return {'n': len(vs), 'delta': None}
    d = np.array([units[v][a] - units[v][b] for v in vs])
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return {'n': len(vs), 'delta': float(d.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            'wins': int((d > 1e-9).sum()), 'losses': int((d < -1e-9).sum()),
            'mean_a': float(d.mean() + np.mean([units[v][b] for v in vs])), 'mean_b': float(np.mean([units[v][b] for v in vs]))}


out = {'macro_iou': {k: float(np.mean([units[v][k] for v in vids if k in units[v]])) for k in ('head', 'center', 'b0', 'best', 'LFM_GROUND', 'QWEN_T')},
       'axis_iou': {r: float(np.mean([units[v]['axis'][r] for v in vids])) for r in RATIOS},
       'paired': {
           'head-center': paired('head', 'center'),
           'head-b0': paired('head', 'b0'),
           'head-LFM_GROUND': paired('head', 'LFM_GROUND'),
           'head-QWEN_T': paired('head', 'QWEN_T'),
       },
       'subgroups': {},
       'oracle_best-center': paired('best', 'center')}
for name, sel in {
    'x_axis_1-3': None, 'y_axis_3-1': None,
    'fast_motion': lambda v: units[v]['layer'] == 'fast_motion',
    'nonperson': lambda v: units[v]['layer'] == 'nonperson',
    'shot_cuts': lambda v: units[v]['layer'] == 'shot_cuts',
    'regular': lambda v: units[v]['layer'] == 'regular',
    'multi_person': lambda v: tags[v], 'single_person': lambda v: not tags[v],
}.items():
    if name == 'x_axis_1-3':
        continue
    out['subgroups'][name] = {'head-b0': paired('head', 'b0', sel)}
# axis split needs per-ratio pairing on the keyframe level
for r in RATIOS:
    sub = [(v, p) for v in vids for p in per if p['vid'] == v and p['ratio'] == r]
    if len(sub) < 10:
        continue
    d = np.array([p['iou'] - b0_iou[(v, p['ratio'], p['kf'])] for v, p in sub])
    bs = d[rng.integers(0, len(d), (10000, len(sub)))].mean(1)
    out['subgroups'][f'axis_{r}_head-b0_kflevel'] = {'n': len(sub), 'delta': float(d.mean()),
                                                     'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({'macro': out['macro_iou'], 'paired': out['paired']}, indent=1))
