"""Round-5 T-gate prep: native features + labels for the FRESH confirm set.

GATED SCRIPT - run ONLY if the L-matrix O-arm beats P (preregistered).
Feature construction reads NO model scores and NO promotion decision; the
one-pass model read stays gated separately.  Every access appends to
fresh_confirm_access_log.jsonl.

Builds, for the frozen 500 fresh-confirm sources:
  1. native tubelet features over the FULL source duration (contract
     ac3b657f..., 4-way card sharding, same builder as the dev build);
  2. original-ACTION labels projected from selections/train.json
     (action = any tubelet center inside a GT interval) - stored per source;
  3. an access-log entry.

NOT built here (separate gated builds if needed): anchor-sliced LFM
features for the VTREPLAY-head comparison arm.

Usage: python3 v11_round5_tgate_prep.py --cards 2,3,4,5
Output: /data/aic/experiments_910a/LFM_V11/R5_NATIVE_FEATS_FRESH/
        <src>.npz + labels.json + build summary
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/fresh_confirm_manifest.json'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--out-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/R5_NATIVE_FEATS_FRESH'))
ap.add_argument('--exp', type=Path, default=Path('/data/aic/experiments_910a'))
ap.add_argument('--cards', default='2,3,4,5')
ap.add_argument('--workers-per-shard', type=int, default=6)
args = ap.parse_args()

LOG = args.exp / 'LFM_V11/fresh_confirm_access_log.jsonl'


def log_access(event, extra=None):
    with LOG.open('a') as f:
        f.write(json.dumps({'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                            'event': event, **(extra or {})}) + '\n')


def main():
    man = json.loads(args.manifest.read_text())
    srcs = man['sources']
    print(f'fresh confirm sources: {len(srcs)} (sha {man["sources_sha256"][:16]})',
          flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) split sources across cards, write per-shard source lists
    cards = [c.strip() for c in args.cards.split(',')]
    shard_files = []
    for i, card in enumerate(cards):
        part = {'meta': f'fresh-confirm native shard {i}/{len(cards)}',
                'train_sources': srcs[i::len(cards)], 'eval_sources': []}
        p = args.out_dir / f'shard_sources_s{i}.json'
        p.write_text(json.dumps(part))
        shard_files.append((card, p))

    # 2) launch one MP builder per card (same contract as the dev build)
    procs = []
    for card, sfile in shard_files:
        env = dict(os.environ,
                   ASCEND_RT_VISIBLE_DEVICES=card,
                   TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                   OMP_NUM_THREADS='4')
        cmd = ['/usr/local/python3.11.15/bin/python3',
               'scripts/v11_round5_native_feats_mp.py',
               '--workers', str(args.workers_per_shard),
               '--batch-windows', '32',
               '--sources-file', str(sfile),
               '--out-dir', str(args.out_dir)]
        log = (args.out_dir / f'build_s{card}.log').open('w')
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                      cwd='/root/AIC'))
        print(f'shard card={card} pid={procs[-1].pid}', flush=True)
    rcs = [p.wait() for p in procs]
    print('shard return codes:', rcs, flush=True)

    # 3) labels (action-level, same rule as the dev matrix)
    sel = json.loads(args.selections.read_text())
    labels = {}
    import numpy as np
    n_act_total = 0
    for s in srcs:
        p = args.out_dir / f'{s}.npz'
        if not p.exists():
            continue
        d = np.load(p)
        meta = json.loads(str(d['meta']))
        centers = np.array([m['tubelet_centers_s'] for m in meta], np.float32)
        y = np.zeros(len(centers), np.float32)
        ivs = []
        for recs in sel.get(s, {}).values():
            for r in recs:
                if float(r['t1']) > float(r['t0']):
                    ivs.append((float(r['t0']), float(r['t1'])))
        for a in range(len(centers)):
            for lo, hi in ivs:
                if any(lo <= c <= hi for c in centers[a]):
                    y[a] = 1.0
                    break
        labels[s] = {'y': y.tolist(),
                     'centers': centers.tolist(),
                     'pos_frac': round(float(y.mean()), 4)}
        n_act_total += len(y)
    (args.out_dir / 'labels.json').write_text(json.dumps(labels))
    n_pos = sum(1 for v in labels.values() if 0 < sum(v['y']) < len(v['y']))
    print(f'labelled sources: {len(labels)}, mixed-action sources: {n_pos}, '
          f'total actions: {n_act_total}', flush=True)

    log_access('native_feature_build', {
        'sources_requested': len(srcs),
        'sources_built': len(labels),
        'contract': 'ac3b657f42264c9a-family',
        'note': 'feature construction + label projection only; no model score read'})

    summary = {'sources_requested': len(srcs), 'sources_built': len(labels),
               'mixed_action_sources': n_pos, 'total_actions': n_act_total,
               'shard_rcs': rcs}
    (args.out_dir / 'prep_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
