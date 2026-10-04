"""Round-4 clean confirm-set: freeze 500 NEW PHD2 training sources.

Implements steps 1-3 of the GPT-6-PRO Q6.5 protocol.  The 3-seed pooled
expected-F evidence lives on sources that are dev-exposed (every V10/V11
experiment trained or selected on them), so no local number so far is a clean
confirm.  This script freezes a source-disjoint confirmation pool BEFORE any
label is read and BEFORE any comparison is run, and writes an exclusion audit.

Exclusion order (a source must survive ALL of):
  official   testing.csv ids, tier0 (official-test is_last hosts) and tier2
             (testing spares) from the download manifest - the family the
             official 426-video drop is drawn from
  dev        every src in PHD2_FRAG_V1/index*.jsonl (all fragment experiments),
             every id in LFM_V10/eval_sources_50.json, and any other
             *sources*.json found under the experiments root (mechanical sweep)
  labeled    must carry >= 1 valid GT interval in selections/train.json
  media      the source mp4 must be on disk
  length     video_duration >= 12 s so at least one 8-14 s fragment fits

Sampling: sources sorted by id, numpy RandomState(20261004).choice without
replacement - reproducible and fixed before any feature or score exists.

Known limitation, recorded in the manifest: exclusion is ID-level.  Perceptual
near-duplicates across different YouTube ids are NOT audited here; that audit
is listed as a follow-up before this set is used for any headline claim.

Output: /data/aic/experiments_910a/LFM_V11/confirm_sources_500.json
"""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np

D = Path('/data/aic/external_datasets/PHD2')
EXP = Path('/data/aic/experiments_910a')

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--seed', type=int, default=20261004)
ap.add_argument('--out', default=EXP / 'LFM_V11/confirm_sources_500.json')
args = ap.parse_args()


def csv_ids(path):
    ids, dur = set(), {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip('\n').split(',')
            if len(parts) < 5:
                continue
            vid = parts[0]
            ids.add(vid)
            try:
                d = float(parts[4])
                dur[vid] = max(dur.get(vid, 0.0), d)
            except ValueError:
                pass
    return ids, dur


def main():
    train_ids, dur = csv_ids(D / 'annotations/upstream_repo/training.csv')
    test_ids, _ = csv_ids(D / 'annotations/upstream_repo/testing.csv')
    man = json.loads((D / 'splits/subset_download_v1.json').read_text())
    tier0 = set(man['tiers'].get('tier0', []))
    tier2 = set(man['tiers'].get('tier2', []))
    official = test_ids | tier0 | tier2

    dev = set()
    for p in (EXP / 'PHD2_FRAG_V1').glob('index*.jsonl'):
        for line in p.read_text().splitlines():
            if line.strip():
                dev.add(json.loads(line)['src'])
    ev50 = EXP / 'LFM_V10/eval_sources_50.json'
    if ev50.exists():
        dev |= set(json.loads(ev50.read_text()))
    swept = []
    for p in EXP.glob('**/*sources*.json'):
        try:
            v = json.loads(p.read_text())
            if isinstance(v, list):
                ids = {x if isinstance(x, str) else x.get('src') for x in v}
                ids.discard(None)
                if ids:
                    dev |= ids
                    swept.append(str(p))
        except Exception:                                    # noqa: BLE001
            pass

    sel = json.loads((D / 'annotations/selections/train.json').read_text())
    labeled = set()
    for src, users in sel.items():
        ok = any(float(r['t1']) > float(r['t0'])
                 for recs in users.values() for r in recs)
        if ok:
            labeled.add(src)

    media = {p.stem for p in (D / 'raw/youtube').glob('*.mp4')}

    base = train_ids - official - dev
    audit = {
        'training_csv_ids': len(train_ids),
        'minus_official_family': len(train_ids & official),
        'minus_dev_exposed': len((train_ids - official) & dev),
        'minus_unlabeled': len(base - labeled),
        'minus_media_missing': len((base & labeled) - media),
    }
    eligible = sorted((base & labeled & media))
    audit['eligible_before_length'] = len(eligible)
    eligible = [s for s in eligible if dur.get(s, 0.0) >= 12.0]
    audit['minus_too_short'] = audit['eligible_before_length'] - len(eligible)
    audit['eligible_pool'] = len(eligible)

    assert len(eligible) >= args.n, f'pool too small: {len(eligible)}'
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(eligible), args.n, replace=False)
    chosen = sorted(eligible[i] for i in pick)
    body = json.dumps(chosen, sort_keys=True).encode()
    sha = hashlib.sha256(body).hexdigest()

    out = {
        'purpose': 'clean confirm set for the pooled expected-F vs MSE '
                   'comparison; frozen before any label/score of its members '
                   'was read',
        'seed': args.seed, 'n_sources': len(chosen),
        'sources': chosen, 'sources_sha256': sha,
        'exclusion_audit': audit,
        'dev_sweep_files': swept,
        'known_limits': [
            'ID-level exclusion only; perceptual near-duplicate audit across '
            'different YouTube ids is a required follow-up before headline use',
            'labels come from selections/train.json (training users); the '
            'official-domain transfer of ANY local number remains unmeasured',
        ],
        'next_steps_frozen_protocol': [
            'cut 8-14 s fragments with the v9_phd2_fragments protocol',
            'extract the SAME pooled 8-slot features as the dev experiments',
            'evaluate the FROZEN dev checkpoints (expected_f_arm_{mse,exactdp}'
            '_s2026100{6,7,8}.pt) once, all seeds, no re-tuning',
            'report per-seed paired deltas and a source-cluster bootstrap',
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'eligible_pool': len(eligible), 'chosen': len(chosen),
                      'sha16': sha[:16], 'audit': audit}, indent=1))


if __name__ == '__main__':
    main()
