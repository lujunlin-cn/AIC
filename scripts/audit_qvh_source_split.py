"""Audit QVH clip/source leakage; optionally emit a train-purged protocol.

Keep validation fixed, remove train clips whose original YouTube source occurs
in validation. Never overwrite the input manifests or claim a new pristine test.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def original_source(source_id):
    return re.sub(r'_\d+(?:\.\d+)?_\d+(?:\.\d+)?$', '', str(source_id))


def audit(rows):
    groups = {s: {original_source(r['source_id']) for r in rows if r['split'] == s}
              for s in ('train', 'val')}
    overlap = groups['train'] & groups['val']
    purged = [r for r in rows if r['split'] == 'train' and original_source(r['source_id']) in overlap]
    return {'protocol': 'QVH_SOURCE_PURGE_V1', 'original_counts': {
        s: sum(r['split'] == s for r in rows) for s in groups},
        'source_counts': {s: len(g) for s, g in groups.items()},
        'shared_sources': sorted(overlap), 'removed_train_video_ids': [r['video_id'] for r in purged],
        'purged_train_count': sum(r['split'] == 'train' for r in rows) - len(purged),
        'policy': 'validation unchanged; exclude all train clips from shared original source; no pristine test claim'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    result = audit(rows); result['input_sha256'] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    a.output.mkdir(parents=True, exist_ok=False)
    excluded = set(result['removed_train_video_ids'])
    for split in ('train', 'val'):
        selected = [{**r, 'source_group': original_source(r['source_id'])} for r in rows
                    if r['split'] == split and r['video_id'] not in excluded]
        (a.output / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in selected))
    (a.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
