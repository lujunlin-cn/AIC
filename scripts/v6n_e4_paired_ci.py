"""V6-NEXT E4: paired per-video deltas + source-clustered bootstrap CI (confirm).

Prereg promotion gate: paired gain mean > 0 AND source-clustered 95% CI
lower bound > 0 vs the locked T0 baseline, on the one-shot confirmation split.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

E4 = Path('/data/aic/experiments/V6N_E4')
RELEASE = Path('/data/aic/external_datasets/_releases/mrhisum_feat_subset_v1')


def load_rows(csv_path):
    with open(csv_path) as f:
        return {r['video_id']: float(r['spearman']) for r in csv.DictReader(f)}


def group_map():
    gm = {}
    for name in ('dev', 'confirmation'):
        for l in (RELEASE / 'manifests' / f'{name}.jsonl').read_text().splitlines():
            r = json.loads(l)
            gm[r['video_id']] = r['group_id']
    return gm


def cluster_bootstrap_ci(deltas_by_group, n=10000, seed=20260930):
    rng = np.random.default_rng(seed)
    groups = list(deltas_by_group.keys())
    arrs = [np.asarray(deltas_by_group[g]) for g in groups]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(float(np.mean(np.concatenate([arrs[i] for i in pick]))))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    gm = group_map()
    base = load_rows(E4 / 'confirm_weights_t0_s1.csv')
    cands = ('t1_s1', 't1_s2', 't1_s3', 't2_s1', 't2_s2', 't2_s3')
    out = []
    for cand in cands:
        rows = load_rows(E4 / f'confirm_weights_{cand}.csv')
        vids = sorted(set(base) & set(rows))
        by_g = defaultdict(list)
        for v in vids:
            by_g[gm[v]].append(rows[v] - base[v])
        deltas = np.concatenate([np.asarray(x) for x in by_g.values()])
        lo, hi = cluster_bootstrap_ci(by_g)
        rec = {'candidate': cand, 'pairs': len(vids), 'source_groups': len(by_g),
               'paired_delta_mean': float(deltas.mean()),
               'delta_ci95_lo': lo, 'delta_ci95_hi': hi,
               'win_rate': float((deltas > 0).mean()),
               'gate_pass': bool(deltas.mean() > 0 and lo > 0)}
        out.append(rec)
        print(json.dumps(rec))
    (E4 / 'confirm_paired_gate.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
