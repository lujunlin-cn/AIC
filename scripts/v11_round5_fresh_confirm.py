"""Round-5 fresh confirm set: freeze 500 NEW sources (R5 section 7.4).

Same eligible-pool construction as round-4 (v11_round4_confirm_set.py:
training.csv minus official family minus dev-exposed, labeled, media on
disk, >=12 s), PLUS every source the round-4 cycle already touched:
  - the round-4 confirm 500 (confirm_sources_500.json; demoted to
    dev-audit set - its results are known and it steered round-5 design)
  - the round-4 audit sources (round4_native_input_audit.json)
Per R5 7.4 the OLD confirm set is no longer unseen; a new read needs a
new freeze.  Slicing (anchor + neutral), full original-frame labels, and
the all-positive/all-negative handling are declared frozen HERE, before
any model score of a member source exists.  Any later access to these
sources appends to fresh_confirm_access_log.jsonl.

This script reads labels ONLY through the eligibility check (labeled /
durations), never to select or weight sources.
Output: /data/aic/experiments_910a/LFM_V11/fresh_confirm_manifest.json
        /data/aic/experiments_910a/LFM_V11/fresh_confirm_access_log.jsonl
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data-root', type=Path,
                default=Path('/data/aic/external_datasets/PHD2'))
ap.add_argument('--exp', type=Path,
                default=Path('/data/aic/experiments_910a'))
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--seed', type=int, default=20261005)
ap.add_argument('--exclude-files', nargs='*', type=Path, default=[
    None,  # replaced below with the two round-4 artifacts
])
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/fresh_confirm_manifest.json'))
args = ap.parse_args()

D = args.data_root
EXP = args.exp
args.exclude_files = [p for p in args.exclude_files if p] or [
    EXP / 'LFM_V11/confirm_sources_500.json',
    EXP / 'LFM_V11/round4_native_input_audit.json']


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
                dur[vid] = max(dur.get(vid, 0.0), float(parts[4]))
            except ValueError:
                pass
    return ids, dur


def collect_strings(obj, acc):
    if isinstance(obj, str):
        acc.add(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_strings(k, acc)
            collect_strings(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_strings(v, acc)
    return acc


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

    # round-4 touched sources (confirm 500 + audit sources)
    touched = set()
    touched_files = []
    for p in args.exclude_files:
        if not p.exists():
            print(f'WARN exclude file missing: {p}')
            continue
        s = collect_strings(json.loads(p.read_text()), set())
        keep = {x for x in s if x in train_ids}
        touched |= keep
        touched_files.append({'file': str(p), 'n_ids_matched': len(keep)})

    base = train_ids - official - dev
    eligible = sorted(base & labeled & media)
    audit = {
        'training_csv_ids': len(train_ids),
        'minus_official_family': len(train_ids & official),
        'minus_dev_exposed': len((train_ids - official) & dev),
        'minus_unlabeled': len(base - labeled),
        'minus_media_missing': len((base & labeled) - media),
        'eligible_before_length': len(eligible),
        'minus_too_short': len([s for s in eligible if dur.get(s, 0.0) < 12.0]),
    }
    eligible = [s for s in eligible if dur.get(s, 0.0) >= 12.0]
    audit['minus_round4_touched'] = len([s for s in eligible if s in touched])
    eligible = [s for s in eligible if s not in touched]
    audit['eligible_pool'] = len(eligible)

    assert len(eligible) >= args.n, f'pool too small: {len(eligible)}'
    rng = np.random.RandomState(args.seed)
    pick = rng.choice(len(eligible), args.n, replace=False)
    chosen = sorted(eligible[i] for i in pick)
    body = json.dumps(chosen, sort_keys=True).encode()
    sha = hashlib.sha256(body).hexdigest()

    out = {
        'purpose': 'round-5 fresh confirm set (R5 7.4): sources never touched '
                   'by round-4 training/dev/confirm/audit; slicing protocol, '
                   'label rules and read discipline frozen BEFORE any model '
                   'score of a member exists',
        'seed': args.seed, 'n_sources': len(chosen),
        'sources': chosen, 'sources_sha256': sha,
        'exclusion_audit': audit,
        'round4_touched_files': touched_files,
        'dev_sweep_files': swept,
        'frozen_protocol': {
            'anchor_slicing': 'v9_phd2_fragments.py UNCHANGED (65 percent '
                              'GIF-anchored / 35 percent uniform, 8-14 s, 8x1 s '
                              'keyframes, rotate 0.30, n_per_video 2)',
            'neutral_slicing': 'v9_phd2_fragments.py --neutral-starts: 100 '
                               'percent label-agnostic uniform starts; SAME '
                               'length distribution, same per-source count, '
                               'same rotate/ratio mix; NO all-pos/all-neg '
                               'filtering in either arm',
            'labels': 'original-frame binary labels projected from '
                      'selections/train.json onto the slice timeline '
                      '(v11_round4_confirm_eval.load_fragments rule)',
            'metrics': 'keep-0.80 original-frame macro F primary; AP and '
                       'positive/empty-GT strata reported separately; '
                       'source-cluster bootstrap',
            'read_discipline': 'one preregistered read per frozen candidate '
                               'group, after checkpoints/thresholds/statistics '
                               'are frozen; every access to these sources '
                               'appends to fresh_confirm_access_log.jsonl',
        },
        'known_limits': [
            'ID-level exclusion only; perceptual near-duplicate audit across '
            'different YouTube ids remains a follow-up',
            'labels are PHD2 training users; official-domain transfer of any '
            'local number stays unmeasured',
        ],
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    log = args.out.parent / 'fresh_confirm_access_log.jsonl'
    if not log.exists():
        with log.open('w') as f:
            f.write(json.dumps({
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'event': 'freeze',
                'n_sources': len(chosen), 'sha256': sha,
                'seed': args.seed}) + '\n')
    print(json.dumps({'n': len(chosen), 'sha16': sha[:16],
                      'audit': audit, 'touched': touched_files}, indent=1))
    print('WROTE', args.out, log)


if __name__ == '__main__':
    main()
