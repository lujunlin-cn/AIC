"""V8 direction-1 arm comparison: source-level paired bootstrap on held-out pools.

Arms (identical head, 1,511,425 params): B1 rv_native (V7 replica), B2 +rv_rot,
B3 rv+rot+live_train, B4 live_train only; KD0.1/KD0.3 teacher-preference
distillation on rv_native.  Two seeds each.

Selection (rv_dev / live_dev) is OPTIMISTIC for arms that used it; the
decision pools are rv_diag and live_confirm - fresh for every V8 head.
Paired unit = (vid, ratio): per-source mean(iou(argmax) - iou(center)).
Controls are paired within seed, then averaged across seeds.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--arms', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/arms'))
ap.add_argument('--pools', nargs='*', default=['rv_diag', 'live_confirm', 'rv_dev', 'live_dev'])
ap.add_argument('--n-boot', type=int, default=10000)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402

ARMS = {'B1': ('rv_native',), 'B2': ('rv_native', 'rv_rot'), 'B3': ('rv_native', 'rv_rot', 'live_train'),
        'B4': ('live_train',), 'KD0.1': ('rv_native',), 'KD0.3': ('rv_native',)}
CONTROLS = [('B3', 'B1'), ('B4', 'B1'), ('B2', 'B1'), ('KD0.1', 'B1'), ('KD0.3', 'B1')]
SEEDS = [0, 1]


def load_pool(arm, seed, pool):
    p = args.arms / f'{arm}_s{seed}' / f'per_{pool}.jsonl'
    if not p.exists():
        return None
    per = [json.loads(l) for l in open(p)]
    src = {}
    for r in per:
        src.setdefault((r['vid'], r['ratio']), []).append(r['iou'] - r['center'])
    return {k: float(np.mean(v)) for k, v in src.items()}


def paired(a, b):
    ks = sorted(set(a) & set(b))
    return ks, np.array([a[k] - b[k] for k in ks])


rng = np.random.default_rng(0)
out = {'arms': {}, 'controls': {}, 'note': 'd_center per (vid,ratio); bootstrap CI 95%; '
        'rv_diag/live_confirm are fresh confirmation pools for all V8 heads'}

for arm in ARMS:
    entry = {}
    for seed in SEEDS:
        s = args.arms / f'{arm}_s{seed}' / 'summary.json'
        if not s.exists():
            entry['missing'] = True
            break
        entry['missing'] = False
        summ = json.loads(s.read_text())
        entry.setdefault('best_score', []).append(round(summ['best_score'], 4))
        entry.setdefault('wall_s', []).append(summ['wall_s'])
        for pool in args.pools:
            d = load_pool(arm, seed, pool)
            if d is not None:
                entry.setdefault(pool, []).append(round(float(np.mean(list(d.values()))), 4))
    out['arms'][arm] = entry

for hi, lo in CONTROLS:
    ctrl = {}
    for pool in args.pools:
        per_seed = []
        for seed in SEEDS:
            a, b = load_pool(hi, seed, pool), load_pool(lo, seed, pool)
            if a is None or b is None:
                continue
            _, d = paired(a, b)
            per_seed.append(d)
        if not per_seed:
            continue
        n = min(len(d) for d in per_seed)  # align length across seeds (sources identical in practice)
        D = np.stack([d[:n] for d in per_seed]).mean(0)  # mean over seeds, per source
        boots = [float(rng.choice(D, len(D), replace=True).mean()) for _ in range(args.n_boot)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        ctrl[pool] = {'mean': round(float(D.mean()), 4), 'ci': [round(c, 4) for c in ci],
                      'n_src': int(len(D)), 'win': int((D > 0).sum()), 'loss': int((D < 0).sum())}
    out['controls'][f'{hi}-{lo}'] = ctrl

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=1) + '\n')

# console summary
print(f'{"arm":6s}', *[f'{p:>22s}' for p in args.pools])
for arm, e in out['arms'].items():
    if e.get('missing'):
        print(f'{arm:6s} MISSING')
        continue
    row = f'{arm:6s}'
    for pool in args.pools:
        v = e.get(pool, [])
        row += f'  {np.mean(v):+.4f}(n{len(v)})' if v else f'  {"--":>14s}'
    print(row, 'best_dev=', e['best_score'])
print()
for name, ctrl in out['controls'].items():
    for pool in ('rv_diag', 'live_confirm'):
        if pool in ctrl:
            c = ctrl[pool]
            print(f'{name:8s} {pool:14s} {c["mean"]:+.4f} [{c["ci"][0]:+.4f},{c["ci"][1]:+.4f}] '
                  f'n={c["n_src"]} W/L={c["win"]}/{c["loss"]}')
print('WROTE', args.output)
