"""Seal completed diagnostic cache metadata after the shared GPU task exits."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from aic.scale_final_preflight import validate_join
from scripts.run_final_p1_cache import read_rows, sha, write


def recover_completed_sources(root, m, labels):
    """Recover a fully written video receipt interrupted before row export."""
    protocols = {json.loads(p.read_text())['position_protocol_sha256'] for p in root.glob('worker_*_load.json')}
    recovered = []
    for path in sorted(root.glob('workers/*/videos/*.json')):
        value = json.loads(path.read_text())
        source = value['records'][0]['source_id']
        target = root/'records'/(source+'.json')
        if target.exists():
            continue
        if len(protocols) != 1 or value['position_protocol_sha256'] not in protocols:
            raise ValueError('completed source protocol differs')
        expected = [r for r in labels if r['source_id'] == source]
        if value['label_pixel_roundtrips'] != len(expected)//2:
            raise ValueError('completed source pixel roundtrips incomplete')
        rows = []
        for r in value['records']:
            rows.append({**r, 'p1_base_sha256': m['p1_base_sha256'],
                'p1_adapter_sha256': m['p1_adapter_sha256_by_route'][
                    'portrait' if r['target_ratio'] == [9, 16] else 'vertical'],
                'p1_adapter_manifest_sha256': m['p1_adapter_manifest_sha256'],
                'tokenizer_sha256': m['tokenizer_sha256'],
                'postprocessing_config_sha256': m['postprocessing_config_sha256'],
                'recovered_completed_source_after_stop': True, 'source_cache_sha256': sha(path)})
        validate_join(expected, rows, m)
        write(target, rows)
        recovered.append(source)
    audit_path = root/'recovery_audit.json'
    previous = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    write(audit_path, {'recovered_sources': sorted(set(previous.get('recovered_sources', [])+recovered)),
        'new_model_calls': 0, 'new_GPU_seconds': 0,
        'scope': 'only complete atomic real-P1 video receipts; no interpolation or GT substitution'})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--recover-only', action='store_true')
    a = p.parse_args(); root = a.root
    ledger = json.loads((root/'gpu_budget_ledger.json').read_text())
    if any(r['ended_at'] is None for r in ledger['runs']):
        raise RuntimeError('GPU task has not exited; do not finalize running work')
    if not ledger.get('within_budget') or ledger['cumulative_GPU_seconds'] > 3600:
        raise RuntimeError('GPU budget violation cannot be declared ready')
    m = json.loads((root/'p1_cache_manifest.json').read_text())
    labels = read_rows(root/'inputs/cohort.jsonl')
    recover_completed_sources(root, m, labels)
    if a.recover_only:
        return
    raw = []
    for path in sorted((root/'records').glob('*.json')):
        raw.extend(json.loads(path.read_text()))
    validate_join(labels, raw, m)
    exposure_path = root/'parent_exposure_audit.json'
    exposure = json.loads(exposure_path.read_text())
    final = []
    for r in raw:
        branch = 'A' if r['target_ratio'] == [9, 16] else 'B'
        proof = exposure['branches'][branch]
        if not proof['deployed_adapter_identity_matches']:
            raise ValueError('P1 training provenance is not this deployed adapter')
        independent = (r['source_id'] in proof['independent_validation_sources']) if r['split'] == 'val' else None
        final.append({**r, 'p1_validation_source_disjoint_verified': independent,
            'p1_training_manifest_sha256': proof['training_manifest_sha256'],
            'p1_training_exposure_audit_sha256': sha(exposure_path)})
    final.sort(key=lambda r: (r['dataset_source'], r['source_id'], tuple(r['target_ratio']), r['frame_id']))
    target = root/'positions.jsonl'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in final)); tmp.replace(target)
    replay_path = root/'geometry_replay.json'
    replay = json.loads(replay_path.read_text())
    if replay['status'] != 'passed' or replay['checked_view_frames'] != len(final):
        raise ValueError('CPU replay does not cover all retained final P1 records')
    workers = {}
    for path in sorted(root.glob('worker_*_final.json')):
        value = json.loads(path.read_text())
        if not value['parent_immutable'] or value['optimizer_steps'] != 0:
            raise ValueError('P1 frozen/no-optimizer condition failed')
        workers[path.name] = {'sha256': sha(path), 'result': value}
    expected_workers = max(r['workers'] for r in ledger['runs'])
    if len(workers) != expected_workers:
        raise ValueError('missing final frozen-P1 worker receipt')
    features_path = root/'visual_features.json'
    if features_path.exists():
        features = json.loads(features_path.read_text())['features_by_image_sha256']
        if not {r['image_sha256'] for r in final}.issubset(features):
            raise ValueError('visual features do not cover final cache')
    m.update(finalized_at=datetime.now(timezone.utc).isoformat(),
        cache_status='partial_budget_bounded_diagnostic', positions_sha256=sha(target),
        completed_sources=len({r['source_id'] for r in final}), completed_view_frames=len(final),
        completed_physical_frames=len({(r['source_id'], r['frame_id']) for r in final}),
        all_requested_sources_completed=len({r['source_id'] for r in final}) == 1795,
        GPU_seconds=ledger['cumulative_GPU_seconds'], GPU_budget_ledger_sha256=sha(root/'gpu_budget_ledger.json'),
        parent_exposure_audit_sha256=sha(exposure_path), geometry_replay_sha256=sha(replay_path),
        frozen_parent_worker_receipts=workers,
        visual_features_sha256=sha(features_path) if features_path.exists() else None,
        raw_per_source_records_unchanged=True, formal_training_started=False)
    assets = root.parent/'audit/remote_assets.json'
    if assets.exists():
        m['p1_weight_file_identity_evidence'] = {'path': str(assets), 'sha256': sha(assets)}
    m['p1_base_sha256_semantics'] = 'SHA256 of canonical filename-to-shard-SHA256 manifest; individual actual weight hashes in remote_assets.json'
    write(root/'p1_cache_manifest.json', m)
    print(json.dumps({'GPU_seconds': m['GPU_seconds'], 'sources': m['completed_sources'],
        'physical_frames': m['completed_physical_frames'], 'view_frames': len(final)}))


if __name__ == '__main__':
    main()
