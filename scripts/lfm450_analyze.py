"""Aggregate LFM450 spatial rows: paired source-video bootstrap, axis/layer splits,
localisation diagnostics and failure taxonomy.  Pure CPU, reads rows_s*.jsonl +
raw_s*.jsonl written by scripts/lfm450_eval_spatial.py.

Pairing unit = source video: both ratio tasks of a video are averaged into one
unit before resampling (they are not independent samples).
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--boot', type=int, default=10000)
a = ap.parse_args()

rows = [json.loads(l) for f in sorted(a.run_dir.glob('rows_s*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
raw = [json.loads(l) for f in sorted(a.run_dir.glob('raw_s*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
sc = [r for r in rows if 'iou' in r]
iou = {(r['vid'], r['ratio'], r['policy']): r['iou'] for r in sc}
meta = {(r['vid'], r['ratio']): r for r in sc if r['policy'] == 'B0'}
vids = sorted({r['vid'] for r in sc})
policies = sorted({r['policy'] for r in sc})
rng = np.random.default_rng(0)


def unit_vals(pol, keys):
    """Per-source-video mean over the given (vid, ratio) task keys."""
    by = defaultdict(list)
    for v, r in keys:
        if (v, r, pol) in iou:
            by[v].append(iou[(v, r, pol)])
    return {v: float(np.mean(x)) for v, x in by.items()}


def paired(pol, base, keys):
    A, B = unit_vals(pol, keys), unit_vals(base, keys)
    common = sorted(set(A) & set(B))
    if not common:
        return None
    d = np.array([A[v] - B[v] for v in common])
    bs = d[rng.integers(0, len(d), (a.boot, len(d)))].mean(1)
    return {'n_src': len(d), 'delta': float(d.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            'wins': int((d > 1e-9).sum()), 'losses': int((d < -1e-9).sum())}


all_keys = sorted(meta)
subsets = {'all': all_keys,
           'x_axis(1-3 portrait target)': [k for k in all_keys if meta[k]['axis'] == 'x'],
           'y_axis(3-1 landscape target)': [k for k in all_keys if meta[k]['axis'] == 'y']}
for layer in sorted({m['layer'] for m in meta.values()}):
    subsets[f'layer={layer}'] = [k for k in all_keys if meta[k]['layer'] == layer]

out = {'n_source_videos': len(vids), 'n_tasks': len(all_keys), 'policies': policies,
       'all_crops_legal': all(r['legal'] for r in sc), 'tables': {}, 'paired': {}}
for name, keys in subsets.items():
    out['tables'][name] = {p: {'mean_iou': float(np.mean(list(unit_vals(p, keys).values()))) if unit_vals(p, keys) else None,
                               'n_src': len(unit_vals(p, keys))} for p in policies}
comparisons = [('LFM_GROUND', 'CENTER'), ('LFM_GROUND', 'B0'), ('LFM_GROUNDR', 'CENTER'), ('LFM_GROUNDR', 'B0'),
               ('LFM_GROUND', 'LFM_GROUNDR'), ('QWEN_T', 'B0'), ('LFM_GROUND', 'QWEN_T'),
               ('LFM_GROUND+INTERP', 'LFM_GROUND'), ('LFM_GROUND+INTERP', 'QWEN_T+INTERP'), ('B0', 'CENTER')]
for name, keys in subsets.items():
    out['paired'][name] = {f'{p} - {b}': paired(p, b, keys) for p, b in comparisons}

# localisation diagnostics (per keyframe, free axis): |pred centre - mean GT crop centre| and hit rate
loc = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r.get('loc') and r['policy'] in ('LFM_GROUND', 'LFM_GROUNDR', 'QWEN_T', 'B0_RAW') and '+' not in r['policy']:
        ax = meta[(r['vid'], r['ratio'])]['axis']
        for sub in ('all', ax):
            if r['loc']['abs_err'] is not None:
                loc[sub][r['policy']].append((r['loc']['abs_err'], r['loc']['hit'], r['loc']['n']))
out['localisation'] = {sub: {p: {'abs_err': float(np.average([x[0] for x in v], weights=[x[2] for x in v])),
                                 'hit_rate': float(np.average([x[1] for x in v], weights=[x[2] for x in v])),
                                 'n_kf': int(sum(x[2] for x in v))} for p, v in d.items()} for sub, d in loc.items()}

# parse / format taxonomy from raw replies
tax = {}
for pr in ('GROUND', 'GROUNDR'):
    rr = [x for x in raw if x['prompt'] == pr]
    st = Counter(x['status'] for x in rr)
    nb = Counter(min(x['n_boxes'], 3) for x in rr)
    lab = Counter()
    wide, tiny, full = 0, 0, 0
    for x in rr:
        b = x.get('box')
        if b:
            w, h = b[2] - b[0], b[3] - b[1]
            wide += w >= .9
            full += (w >= .9 and h >= .9)
            tiny += w * h < .01
        for g in __import__('re').findall(r'"label"\s*:\s*"([^"]*)"', x['reply']):
            lab[g.lower()[:30]] += 1
    tax[pr] = {'n': len(rr), 'status': dict(st), 'parse_ok_rate': st.get('ok', 0) / max(len(rr), 1),
               'n_boxes_hist(3=3+)': dict(nb), 'box_width>=0.9': wide, 'box_fullframe': full, 'box_area<1%': tiny,
               'top_labels': lab.most_common(8), 'avg_s': float(np.mean([x['s'] for x in rr])) if rr else None}
out['format'] = tax

# per-video failure cases for overlays (largest losses/gains vs CENTER and B0)
dv = []
for v in vids:
    for r in ('1-3', '3-1'):
        if (v, r, 'LFM_GROUND') in iou:
            dv.append({'vid': v, 'ratio': r, 'axis': meta[(v, r)]['axis'], 'layer': meta[(v, r)]['layer'],
                       'lfm': iou[(v, r, 'LFM_GROUND')], 'center': iou[(v, r, 'CENTER')], 'b0': iou[(v, r, 'B0')],
                       'qwen_t': iou[(v, r, 'QWEN_T')], 'd_vs_b0': iou[(v, r, 'LFM_GROUND')] - iou[(v, r, 'B0')]})
dv.sort(key=lambda x: x['d_vs_b0'])
out['worst_vs_b0'] = dv[:6]
out['best_vs_b0'] = dv[-6:][::-1]
a.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')

print('source videos', len(vids), 'tasks', len(all_keys), 'legal', out['all_crops_legal'])
for name in subsets:
    t = out['tables'][name]
    print(f'{name:32s}', '  '.join(f'{p}={t[p]["mean_iou"]:.4f}' for p in ('CENTER', 'B0', 'LFM_GROUND', 'LFM_GROUNDR', 'QWEN_T', 'LFM_GROUND+INTERP') if t[p]['mean_iou'] is not None), f'(n={t["B0"]["n_src"]})')
for name in ('all', 'x_axis(1-3 portrait target)', 'y_axis(3-1 landscape target)'):
    for k, v in out['paired'][name].items():
        if v:
            print(f'  [{name[:6]}] {k:36s} {v["delta"]:+.4f} [{v["ci95"][0]:+.4f},{v["ci95"][1]:+.4f}] {v["wins"]}/{v["losses"]}')
print('LOC', json.dumps(out['localisation'], indent=None)[:900])
print('FMT', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'top_labels'} for k, v in tax.items()}))
print('LABELS', json.dumps({k: v['top_labels'] for k, v in tax.items()}, ensure_ascii=False))
