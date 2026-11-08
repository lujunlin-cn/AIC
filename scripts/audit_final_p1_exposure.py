"""Audit known P1 fine-tuning exposure from actual adapter/Train files on CPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

from scripts.run_final_p1_cache import read_rows, sha, write


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    labels = read_rows(a.labels)
    val = [r for r in labels if r['dataset_source'] == 'LIVE_YT_VC' and r['split'] == 'val']
    branches = {}
    for arm, filename, adapter_path, bank in [
        ('A', 'train_a.json', 'runs/arm_a/checkpoint_1400/adapter/adapter_model.safetensors', 'portrait'),
        ('B', 'train_b.json', 'runs/arm_b/checkpoint_1400/other_geometry/adapter_model.safetensors', 'vertical')]:
        path = a.training_root / 'data' / filename
        rows = json.loads(path.read_text())
        train_sources = set()
        train_hashes = set()
        for r in rows:
            ids = re.findall(r'video\d+', str(r['group']))
            if len(ids) != 1 or r['split'] != 'train':
                raise ValueError('ambiguous training source/split')
            train_sources.add(ids[0]); train_hashes.add(r['image_sha256'])
        checkpoint = a.training_root / adapter_path
        checkpoint_sha = sha(checkpoint)
        matching = checkpoint_sha == c['runtime']['parent_adapter_hashes'][bank]
        exposed_source = sorted({r['source_id'] for r in val if r['source_id'] in train_sources})
        byte_alias_sources = sorted({r['source_id'] for r in val if r['image_sha256'] in train_hashes})
        val_ids = sorted({r['source_id'] for r in val})
        excluded = set(exposed_source + byte_alias_sources)
        build = json.loads((a.training_root / 'data/build_manifest.json').read_text())
        legacy_in_train = build.get('legacy_videos_in_train', [])
        legacy_in_dev = build.get('legacy_videos_in_dev', None)
        branches[arm] = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
            'deployed_adapter_identity_matches': matching,
            'training_manifest': str(path), 'training_manifest_sha256': sha(path),
            'training_rows': len(rows), 'training_sources': len(train_sources),
            'validation_sources': len(val_ids), 'validation_source_id_overlap': exposed_source,
            'validation_sources_with_any_training_image_bytes': byte_alias_sources,
            'independent_validation_sources': [s for s in val_ids if s not in excluded] if matching else None,
            'excluded_validation_sources': sorted(excluded),
            'legacy_video_sources_all_in_train': set(legacy_in_train).issubset(train_sources),
            'legacy_in_dev_record': legacy_in_dev,
            'scope': 'known LIVE P1 fine-tuning manifest; not a proof about unknown foundation-model pretraining'}
    write(a.output, {'branches': branches, 'CUDA_initialized': False,
        'original_training_manifest_not_rewritten': True,
        'historical_validation_was_used_for_model_selection': True,
        'fresh_never_seen_validation': False,
        'policy': 'exclude any Val source with any known P1 training-image alias from independent evaluation'})


if __name__ == '__main__':
    main()
