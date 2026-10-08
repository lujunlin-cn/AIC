"""R11 S4 pre-audit: official-domain (live) vs diagnostic-domain (rv) structure.

S4 proper is a packing gate (B3-vs-candidate frame divergence on the official
set, F12-stratified).  With no pending candidate this pre-audit quantifies the
structural gap between the pools the official score lives on (live_dev,
live_confirmation) and the diagnostic pools (rv_dev, rv_diag), so every future
candidate's S4 divergence budget has a baseline.  CPU only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--table', default='/data/aic/experiments_910a/LFM_V11/r11/s1_score_table.npz')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r11_s4_preaudit.json')
a = p.parse_args()

d = np.load(a.table, allow_pickle=True)
pred, pool, vid = d['pred'], d['pool'].astype(str), d['vid'].astype(str)
dens = (pred > 0.5).mean(1)
top = pred.max(1)

report = {}
for pl in sorted(set(pool)):
    m = pool == pl
    srcs = np.unique(vid[m])
    src_var = np.array([pred[m][vid[m] == v].var() for v in srcs])
    report[pl] = {
        'n_frags': int(m.sum()),
        'n_sources': int(len(srcs)),
        'keep_density_mean': round(float(dens[m].mean()), 4),
        'keep_density_p50': round(float(np.percentile(dens[m], 50)), 4),
        'keep_density_p90': round(float(np.percentile(dens[m], 90)), 4),
        'top_score_mean': round(float(top[m].mean()), 4),
        'src_scorevar_mean': round(float(src_var.mean()), 5),
    }

live = {k: v for k, v in report.items() if k.startswith('live')}
rv = {k: v for k, v in report.items() if k.startswith('rv')}
for k in report['live_dev']:
    if k in ('n_frags', 'n_sources'):
        continue
    lv = np.mean([v[k] for v in live.values()])
    rvv = np.mean([v[k] for v in rv.values()])
    report[f'gap_live_minus_rv::{k}'] = round(float(lv - rvv), 4)

report['note'] = ('pre-audit only: S4 proper needs a pending candidate '
                  '(frame-divergence vs B3 on the official set, F12 stratified).')
Path(a.out).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
