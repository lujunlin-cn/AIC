"""Metadata bundle for the LoRA probe (V11 stage-1 requirements).

Produces, from disk only:
  1. sampler-pool sizes -> per-fragment exposure counts for the 900-step run
  2. the exact evaluation-fragment manifest (mid-run 270 and full pass),
     with SHA-256 of the manifest file
  3. a prevalence audit of the probe pool: measured keep-all binary temporal
     F1 and mean per-fragment positive rate, resolving the p=0.33 vs
     keep-all 0.6237 total mismatch flagged by the V11 review (sec 2.2.1)

CPU only.  Reads index, selections, frame dirs - no NPU, no features.
Label construction is byte-identical to v10_lora_tower_probe.fragment_labels.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames'))
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--selections', type=Path, default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--fragments-per-step', type=int, default=8)
ap.add_argument('--eval-limit', type=int, default=400)
ap.add_argument('--out-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10'))
args = ap.parse_args()

sel = json.loads(args.selections.read_text())
eval_set = set(json.loads(args.eval_sources.read_text()))
rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]


def fragment(r):
    """Byte-identical label logic to the probe script."""
    t0 = float(r['t0'])
    ivs = []
    for recs in sel.get(r['src'], {}).values():
        for rec in recs:
            a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
            if b_ > a_:
                ivs.append((a_, b_))
    kdir = args.frames_root / r['video_id']
    jpgs = sorted(kdir.glob('*.jpg'), key=lambda p: float(p.stem))
    if len(jpgs) < 4 or not ivs:
        return None
    stems = [float(p.stem) for p in jpgs]
    times = np.array([s - t0 for s in stems], np.float32)
    half = ((stems[-1] - stems[0]) / max(len(stems) - 1, 1)) * 0.5
    y = np.zeros(len(jpgs), np.float32)
    for a_, b_ in ivs:
        y[(times + half >= a_) & (times - half < b_)] = 1.0
    n = len(jpgs)
    return {'video_id': r['video_id'], 'src': r['src'], 't0': t0, 'L': float(r['L']),
            'n_frames': n, 'n_pos': int(y.sum()), 'n_intervals': len(ivs),
            'mixed': bool(0 < y.sum() < n)}


frags = []
skipped = 0
for r in rows:
    f = fragment(r)
    if f is None:
        skipped += 1
        continue
    frags.append(f)

tr = [f for f in frags if f['src'] not in eval_set]
ev = [f for f in frags if f['src'] in eval_set]
tr_mix = [f for f in tr if f['mixed']]
ev_mix = [f for f in ev if f['mixed']]
ev_mid = [f for f in ev[:args.eval_limit] if f['mixed']]

# ---- 1. exposure ----
n_pool = len(tr_mix)
instances = args.steps * args.fragments_per_step
lam = instances / n_pool
exposure = {
    'sampler_pool_mixed_fragments': n_pool,
    'sampler_pool_all_rows': len(tr),
    'note': ('the probe samples WITH replacement: rng.choice over train rows, '
             'mixed-filtered, 8 per step'),
    'effective_fragment_instances': instances,
    'mean_exposures_per_fragment': round(lam, 2),
    'poisson_never_sampled_prob': round(float(np.exp(-lam)), 4),
    'poisson_expected_count_at_max_seen': None,
}

# ---- 2. eval manifest ----
def entry(f):
    return {'video_id': f['video_id'], 'src': f['src'], 't0': f['t0'], 'L': f['L'],
            'n_frames': f['n_frames'], 'n_pos': f['n_pos']}

manifest = {
    'midrun_first_eval_limit_rows': args.eval_limit,
    'midrun_fragments': [entry(f) for f in ev_mid],
    'full_fragments': [entry(f) for f in ev_mix],
}
man_path = args.out_dir / 'eval_fragment_manifest.json'
man_path.write_text(json.dumps(manifest, indent=1))
man_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()

# ---- 3. prevalence audit (probe pool, mixed-only, OR-union labels) ----
def keep_all_f(pool):
    """Official binary temporal arithmetic with every frame kept:
    F_v = 2*sum(y)/(n + sum(y)); macro over the pool."""
    f1s, ps = [], []
    for f in pool:
        s, n = f['n_pos'], f['n_frames']
        f1s.append(2 * s / (n + s))
        ps.append(s / n)
    return float(np.mean(f1s)), float(np.mean(ps)), ps

ka_all, pbar_all, ps_all = keep_all_f(frags)          # train+eval mixed pool
ka_ev, pbar_ev, _ = keep_all_f(ev_mix)                # eval-only
audit = {
    'pool_definition': 'index_clean fragments with >=4 frames and >=1 GIF interval, mixed-only, OR-union labels at real frame timestamps with half-frame tolerance',
    'n_fragments_train_eval': len(frags),
    'n_sources': len(set(f['src'] for f in frags)),
    'keep_all_binary_temporal_F1_macro': round(ka_all, 4),
    'mean_per_fragment_positive_rate_pbar': round(pbar_all, 4),
    'g_of_pbar = 2p/(1+p)': round(2 * pbar_all / (1 + pbar_all), 4),
    'jensen_note': 'keep-all macro F equals mean of g(p_v); by concavity it sits BELOW g(mean p) - the gap is the reported inconsistency',
    'eval_only': {'n_fragments': len(ev_mix), 'keep_all_F1': round(ka_ev, 4), 'pbar': round(pbar_ev, 4)},
    'per_fragment_p_deciles': [round(float(x), 4) for x in np.quantile(ps_all, [0, .1, .25, .5, .75, .9, 1])],
    'earlier_values': {'p_0.39': '65% highlight-anchored pool (biased, retired)',
                       'p_0.33': 'estimate quoted in the V10 report; provenance = platform-score consistency, not a pool measurement',
                       'keep_all_0.6237': 'oracle-decomposition pool; this audit recomputes the probe-pool value for comparison'},
}

out = {
    'rows_total': len(rows), 'rows_without_frames_or_intervals': skipped,
    'train': {'rows': len(tr), 'mixed': len(tr_mix)},
    'eval': {'rows': len(ev), 'mixed': len(ev_mix),
             'midrun_mixed_in_first_400_rows': len(ev_mid)},
    'exposure': exposure,
    'eval_manifest_sha256': man_sha,
    'prevalence_audit': audit,
}
(args.out_dir / 'lora_meta.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('rows_total', 'train', 'eval',
                                      'eval_manifest_sha256')}, indent=1))
print(json.dumps(audit, indent=1))
