"""Multi-seed replication of the PHD2 KD-stream effect.

The single-seed result was confirm +0.0178 [+0.0076,+0.0282] on 227 sources.
That is one draw.  This pools the per-row IoU of every (arm, seed) pair and
estimates the effect two ways:

  per-seed   each seed's paired delta on the confirmation pool, then the mean
             and spread across seeds - if the delta is positive in every seed it
             is not a seed artefact.
  pooled     all seeds' rows concatenated, source-level paired bootstrap, which
             is the tighter test because it resamples sources rather than seeds.

Seeds are compared inside a seed (A1_raw_s vs A0_ctrl_s) so that any seed-level
difference in optimisation difficulty cancels out.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--arms', type=Path, default=Path('/data/aic/experiments_910a/LFM_V9'))
ap.add_argument('--pair', nargs=2, default=['A0_ctrl', 'A1_raw'])
ap.add_argument('--pool', default='confirm')
ap.add_argument('--boot', type=int, default=4000)
ap.add_argument('--out', type=Path, default=None)
a = ap.parse_args()

# seed 0 lives in arms/, seeds 1-2 in seeds/<arm>_s<seed>/
runs = {}
for arm in a.pair:
    for seed, d in [(0, a.arms / 'arms' / arm)] + \
                   [(s, a.arms / 'seeds' / f'{arm}_s{s}') for s in (1, 2)]:
        f = d / f'per_{a.pool}.jsonl'
        if f.exists():
            runs[(arm, seed)] = f


def cells(f):
    out = defaultdict(list)
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r['src'], r['ratio'])].append(r['iou'])
    return {k: float(np.mean(v)) for k, v in out.items()}


data = {k: cells(v) for k, v in runs.items()}
ctrl, treat = a.pair
rows = []
for seed in sorted({s for (_, s) in data}):
    c, t = data.get((ctrl, seed)), data.get((treat, seed))
    if not c or not t:
        continue
    common = sorted(set(c) & set(t))
    if not common:
        continue
    by_src = defaultdict(list)
    for k in common:
        by_src[k[0]].append(t[k] - c[k])
    src_means = np.array([np.mean(by_src[s]) for s in sorted(by_src)])
    rows.append({'seed': seed, 'n_cells': len(common), 'n_sources': len(src_means),
                 'delta': round(float(src_means.mean()), 4),
                 'ctrl_mean_iou': round(float(np.mean([c[k] for k in common])), 4),
                 'treat_mean_iou': round(float(np.mean([t[k] for k in common])), 4)})

out = {'pool': a.pool, 'control': ctrl, 'treatment': treat, 'per_seed': rows}
if rows:
    ds = [r['delta'] for r in rows]
    out['across_seed'] = {
        'mean_delta': round(float(np.mean(ds)), 4),
        'min_delta': round(float(np.min(ds)), 4),
        'max_delta': round(float(np.max(ds)), 4),
        'std_delta': round(float(np.std(ds)), 4),
        'seeds_positive': int(sum(1 for d in ds if d > 0)),
        'seeds_total': len(ds),
    }
    # pooled: concatenate rows across seeds, bootstrap sources
    src_all = defaultdict(list)
    for seed in [r['seed'] for r in rows]:
        c, t = data[(ctrl, seed)], data[(treat, seed)]
        for k in set(c) & set(t):
            src_all[k[0]].append(t[k] - c[k])
    srcs = sorted(src_all)
    m = np.array([np.mean(src_all[s]) for s in srcs])
    rng = np.random.default_rng(20261002)
    boot = m[rng.integers(0, len(srcs), size=(a.boot, len(srcs)))].mean(1)
    out['pooled'] = {
        'n_sources': len(srcs),
        'delta': round(float(m.mean()), 4),
        'ci95': [round(float(np.percentile(boot, 2.5)), 4),
                 round(float(np.percentile(boot, 97.5)), 4)],
        'significant': bool(np.percentile(boot, 2.5) > 0),
    }
print(json.dumps(out, indent=1))
if a.out:
    a.out.write_text(json.dumps(out, indent=1))