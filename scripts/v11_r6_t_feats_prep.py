"""R6 T-CONTEXT step 2: build the feature-extraction index from the plan.

One row per variant: video_id = '<src>/<variant_dir>' (nested under
frames_root), src, split, event, window name, t0/t1 (absolute seconds),
frame stamps.  Also creates the output subdirectories that the
round-4 CPU extractor expects (out-root/<src>/).
Output: t_context_index.jsonl
"""
import argparse, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--plan', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_t_context_plan.json'))
ap.add_argument('--out-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/t_context_feats'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/t_context_index.jsonl'))
args = ap.parse_args()

plan = json.loads(args.plan.read_text())
n = 0
args.out_root.mkdir(parents=True, exist_ok=True)
with args.out.open('w') as f:
    for rec in plan['sources']:
        (args.out_root / rec['src']).mkdir(parents=True, exist_ok=True)
        for e in rec['events']:
            vd = f'{e["a"]:.2f}_{e["b"]:.2f}'
            for wname, v in e['variants'].items():
                f.write(json.dumps({
                    'video_id': f'{rec["src"]}/{vd}_{wname}',
                    'src': rec['src'], 'split': rec['split'],
                    'event': [e['a'], e['b']],
                    'window': wname, 't0': v['start'], 't1': v['end'],
                    'stamps': v['frames']}) + '\n')
                n += 1
print('index rows:', n)
print('WROTE', args.out)
