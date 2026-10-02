"""Paired source-level bootstrap over V9 student-head arms.

Same protocol the V8 arm comparison used, because the numbers have to be
comparable with it: one value per (source video, ratio) cell, arms differ only
in their KD configuration, and the bootstrap resamples SOURCE VIDEOS (not rows),
since rows inside a source are not independent.

Reports, per pool and per pair:
  mean IoU per arm
  paired delta (hi - lo) with a 95% percentile CI
  wins / losses at the cell level
  a two-sided sign test p-value, as a sanity check on the CI

A pool counts as a win only if the CI excludes zero.  Arm ranking on a single
mean is explicitly not enough - V6 and V8 both produced arms whose local mean
improved and whose platform score fell.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--arms', type=Path, required=True, help='dir with <arm>/per_<pool>.jsonl')
ap.add_argument('--names', nargs='*', default=None)
ap.add_argument('--pools', nargs='*',
                default=['confirm', 'phd2_val', 'rv_dev', 'rv_rot_dev', 'live_dev'])
ap.add_argument('--boot', type=int, default=4000)
ap.add_argument('--seed', type=int, default=20261002)
ap.add_argument('--out', type=Path, default=None)
a = ap.parse_args()

names = a.names or sorted(p.name for p in a.arms.iterdir()
                          if p.is_dir() and list(p.glob('per_*.jsonl')))


def load(arm, pool):
    f = a.arms / arm / f'per_{pool}.jsonl'
    if not f.exists():
        return None
    out = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r['src'], r['ratio'])
        out.setdefault(key, []).append(r['iou'])
    return {k: float(np.mean(v)) for k, v in out.items()}


def sign_p(w, n):
    """Two-sided exact-ish sign test via normal approximation."""
    if n == 0:
        return float('nan')
    z = (abs(w - n / 2) - 0.5) / max(np.sqrt(n / 4), 1e-9)
    from math import erfc
    return erfc(z / np.sqrt(2))


res = {'arms': names, 'pools': {}}
for pool in a.pools:
    data = {nm: load(nm, pool) for nm in names}
    data = {k: v for k, v in data.items() if v}
    if not data:
        continue
    common = set.intersection(*[set(v) for v in data.values()])
    if not common:
        continue
    keys = sorted(common)
    srcs = sorted({k[0] for k in keys})
    sidx = {s: i for i, s in enumerate(srcs)}
    per_arm = {nm: np.array([data[nm][k] for k in keys]) for nm in data}
    # source-level aggregation so the bootstrap resamples the independent unit
    by_src = {nm: defaultdict(list) for nm in data}
    for k, kk in enumerate(keys):
        for nm in data:
            by_src[nm][kk[0]].append(per_arm[nm][k])
    src_means = {nm: np.array([np.mean(by_src[nm][s]) for s in srcs]) for nm in data}
    rng = np.random.default_rng(a.seed)
    boot_idx = rng.integers(0, len(srcs), size=(a.boot, len(srcs)))
    entry = {'n_cells': len(keys), 'n_sources': len(srcs),
             'mean_iou': {nm: round(float(per_arm[nm].mean()), 4) for nm in data},
             'pairs': {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            hi, lo = names[i], names[j]
            if hi not in src_means or lo not in src_means:
                continue
            d = src_means[hi] - src_means[lo]
            bd = d[boot_idx].mean(1)
            cell_d = per_arm[hi] - per_arm[lo]
            w = int((cell_d > 0).sum())
            l = int((cell_d < 0).sum())
            entry['pairs'][f'{hi}_vs_{lo}'] = {
                'delta': round(float(d.mean()), 4),
                'ci95': [round(float(np.percentile(bd, 2.5)), 4),
                         round(float(np.percentile(bd, 97.5)), 4)],
                'wins': w, 'losses': l, 'sign_p': round(sign_p(w, w + l), 4),
                'significant': bool(np.percentile(bd, 2.5) > 0 or
                                   np.percentile(bd, 97.5) < 0),
            }
    res['pools'][pool] = entry

print(json.dumps(res, indent=1))
for pool, e in res['pools'].items():
    print(f'\n== {pool}  (cells={e["n_cells"]}, sources={e["n_sources"]})')
    for nm, v in sorted(e['mean_iou'].items(), key=lambda kv: -kv[1]):
        print(f'  {nm:16s} {v:.4f}')
    for k, v in e['pairs'].items():
        flag = 'SIG' if v['significant'] else '   '
        print(f'  {flag} {k:34s} {v["delta"]:+.4f} [{v["ci95"][0]:+.4f},{v["ci95"][1]:+.4f}] '
              f'W/L={v["wins"]}/{v["losses"]} p={v["sign_p"]:.3f}')
if a.out:
    a.out.write_text(json.dumps(res, indent=1))