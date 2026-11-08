"""Reuse a hash-bound original P1 cache without any new model calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.run_final_p1_cache import read_rows, sha, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--old-cache', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    old = json.loads(a.old_cache.read_text())
    m = json.loads((a.root/'p1_cache_manifest.json').read_text())
    protocols = {json.loads(f.read_text())['position_protocol_sha256'] for f in a.root.glob('worker_*_load.json')}
    if len(protocols) != 1 or old['position_protocol_sha256'] not in protocols:
        write(a.root/'reuse_audit.json', {'status': 'not_reused',
            'reason': 'old projector protocol identity differs from the current exact protocol',
            'old_protocol_sha256': old['position_protocol_sha256'],
            'current_protocol_sha256': sorted(protocols), 'old_cache_sha256': sha(a.old_cache),
            'new_GPU_seconds': 0, 'new_model_calls': 0, 'outputs_imported': 0})
        raise ValueError('old cache is not the current exact original P1 protocol')
    labels = read_rows(a.root/'inputs/cohort.jsonl')
    key = lambda r: (r['source_id'], r['image_id'], tuple(r['target_ratio']))
    lookup = {key(r): r for r in labels}
    selected = []
    for r in old['records']:
        if key(r) not in lookup:
            continue
        label = lookup[key(r)]
        for field in ('split', 'image_sha256', 'image_width', 'image_height', 'frame_id', 'source_video_sha256'):
            if r.get(field) != label.get(field):
                raise ValueError('old cache input mismatch: ' + field)
        if (r['parent_identity_sha256'] != m['parent_identity_sha256']
                or r['position_kind'] != 'P1_final_deployed' or r['qp_floor'] != .15
                or r['spatial_entrypoint_sha256'] != m['spatial_entrypoint_sha256']):
            raise ValueError('old P1 output identity mismatch')
        selected.append({**r, 'p1_base_sha256': m['p1_base_sha256'],
            'p1_adapter_manifest_sha256': m['p1_adapter_manifest_sha256'],
            'p1_adapter_sha256': m['p1_adapter_sha256_by_route'][
                'portrait' if r['target_ratio'] == [9, 16] else 'vertical'],
            'tokenizer_sha256': m['tokenizer_sha256'],
            'postprocessing_config_sha256': m['postprocessing_config_sha256'],
            'reused_prior_cache_sha256': sha(a.old_cache), 'reused_prior_cache_path': str(a.old_cache)})
    if not selected or old.get('label_pixel_roundtrips', 0) < len(selected)//2:
        raise ValueError('old source lacks sampled-frame pixel verification')
    sources = {r['source_id'] for r in selected}
    expected = [r for r in labels if r['source_id'] in sources]
    if {key(r) for r in expected} != {key(r) for r in selected}:
        raise ValueError('old cache does not cover the entire declared sampled source')
    for source in sources:
        path = a.root/'records'/(source+'.json')
        if path.exists():
            raise FileExistsError('source already completed; do not overwrite')
        write(path, [r for r in selected if r['source_id'] == source])
    write(a.root/'reuse_audit.json', {'cache_sha256': sha(a.old_cache), 'sources': sorted(sources),
        'physical_frames': len(selected)//2, 'view_frames': len(selected),
        'current_protocol_match': True, 'prior_original_pixel_roundtrips': old['label_pixel_roundtrips'],
        'new_GPU_seconds': 0, 'new_model_calls': 0})


if __name__ == '__main__':
    main()
