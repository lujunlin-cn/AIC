"""Read-only CPU cross-checks of the sealed, budget-bounded diagnostic evidence.

This does not load a model, change any input, or start a GPU/optimizer task.
Its numerical checks independently reconstruct the source means from boxes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from aic.scale_geometry import box_iou_xywh, checked_scale_action, deployed_center


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_budget(ledger):
    runs = ledger['runs']
    if not runs or any(r.get('ended_at') is None for r in runs):
        raise ValueError('GPU ledger is not closed')
    total = sum(r['elapsed_seconds'] for r in runs)
    if (not ledger['within_budget'] or total > 3600 or ledger['cap_seconds'] != 3600
            or not math.isclose(total, ledger['cumulative_GPU_seconds'], abs_tol=1e-6)):
        raise ValueError('GPU budget mismatch or overrun')
    for r in runs:
        elapsed = (datetime.fromisoformat(r['ended_at']) -
                   datetime.fromisoformat(r['started_at'])).total_seconds()
        if abs(elapsed-r['elapsed_seconds']) > 1 or r['optimizer_steps'] != 0:
            raise ValueError('GPU timestamps/optimizer evidence inconsistent')
        if not r['includes_model_load_warmup_and_inference'] or r['formal_training']:
            raise ValueError('GPU accounting or no-training condition failed')
    return total


def validate(root):
    def read(name):
        return json.loads((root/name).read_text())
    m = read('p1_cache_manifest.json')
    seconds = validate_budget(read('gpu_budget_ledger.json'))
    rows = [json.loads(line) for line in (root/'positions.jsonl').read_text().splitlines()]
    if sha(root/'positions.jsonl') != m['positions_sha256']:
        raise ValueError('sealed positions hash differs')
    if len(rows) != m['completed_view_frames'] or seconds != m['GPU_seconds']:
        raise ValueError('manifest counts/budget differ')
    replay = read('geometry_replay.json')
    if replay['status'] != 'passed' or replay['checked_view_frames'] != len(rows) or replay['CUDA_initialized']:
        raise ValueError('final-P1 CPU geometry replay incomplete')
    for receipt in m['frozen_parent_worker_receipts'].values():
        if not receipt['result']['parent_immutable'] or receipt['result']['optimizer_steps']:
            raise ValueError('frozen parent condition failed')
    for evidence in m['smoke_evidence'].values():
        if not evidence['reused'] or sha(evidence['path']) != evidence['sha256']:
            raise ValueError('reused actual model smoke evidence changed')
    revision = read('code_revision.json')
    for path, expected in revision['files'].items():
        if sha(path) != expected:
            raise ValueError('execution code changed: '+path)
    labels = { (r['dataset_source'], r['source_id'], r['image_id'], tuple(r['target_ratio'])): r
               for r in (json.loads(x) for x in (root/'inputs/cohort.jsonl').read_text().splitlines()) }
    exposure = read('parent_exposure_audit.json')
    constants, oracles = read('constant_scale_results.json'), read('scale_oracle_results.json')
    bins = json.loads((root.parent/'config/scale_bins.json').read_text())['values']
    checks = []
    for group, result in constants['by_dataset_and_route'].items():
        dataset, route = group.split('/')
        ratio = [9, 16] if route == 'A' else [16, 9]
        rr = [r for r in rows if r['dataset_source'] == dataset and r['target_ratio'] == ratio]
        if not rr:
            if result.get('result', 'not-null') is not None:
                raise ValueError('unmeasured dataset/route is not null')
            checks.append({'group': group, 'measured': False, 'reason': result['reason']})
            continue
        allowed = set(exposure['branches'][route]['independent_validation_sources'])
        rr = [r for r in rr if r['split'] == 'train' or r['source_id'] in allowed]
        candidates = [x['scale'] for x in result['all_constants']]
        scales = sorted(set(candidates+bins))
        values = defaultdict(list)
        maximum_drift = 0.
        for r in rr:
            key = r['dataset_source'], r['source_id'], r['image_id'], tuple(r['target_ratio'])
            label = labels[key]
            if r['image_sha256'] != label['image_sha256'] or r['frame_id'] != label['frame_id']:
                raise ValueError('image/frame contract mismatch')
            center = deployed_center(r['p1_aic_bbox_xyw'], ratio)
            quality = []
            for s in scales:
                a = checked_scale_action(r['image_width'], r['image_height'], ratio, center, s)
                if not a['valid']:
                    raise ValueError('accepted evaluation includes an illegal fixed-center action')
                maximum_drift = max(maximum_drift, a['position_drift_px'])
                quality.append(box_iou_xywh(a['crop_xywh'], label['gt_crop_xywh']))
            values[r['split'], r['source_id']].append(quality)
        means = {key: [sum(v[j] for v in vv)/len(vv) for j in range(len(scales))]
                 for key, vv in values.items()}
        def mean(split, scale):
            v = [x[scales.index(scale)] for (sp, _), x in means.items() if sp == split]
            return sum(v)/len(v)
        selected = max(candidates, key=lambda s: (mean('train', s), s))
        if selected != result['best_constant_scale']:
            raise ValueError('best constant differs from independent Train-only selection')
        if not math.isclose(mean('val', selected), result['val_source_macro_iou']['mean'], abs_tol=1e-12):
            raise ValueError('Val source-macro constant mismatch')
        oracle = oracles['by_dataset_and_route'][group]
        if not oracle['GT_visible_upper_bound'] or oracle['learned_model_greedy'] is not None:
            raise ValueError('Oracle/model semantics changed')
        independent_order = sorted(bins, key=lambda s: (-mean('train', s), -s))
        if independent_order != oracle['candidate_order']:
            raise ValueError('Oracle prefixes were not selected on Train')
        last = -math.inf
        for k in (1, 2, 4, 8):
            by_source = []
            for (split, _), vv in values.items():
                if split == 'val':
                    by_source.append(sum(max(v[scales.index(s)] for s in independent_order[:k])
                                         for v in vv)/len(vv))
            value = sum(by_source)/len(by_source)
            actual = oracle['oracle_at_k'][str(k)]['val_source_macro_iou']
            if not math.isclose(value, actual['mean'], abs_tol=1e-12) or value < last-1e-12:
                raise ValueError('independent GT-visible source-macro Oracle mismatch')
            if actual['source_count'] != len(by_source):
                raise ValueError('frames were counted as independent sources')
            last = value
        checks.append({'group': group, 'measured': True, 'train_only_best_constant': selected,
                       'oracle_1_2_4_8_independently_recomputed': True,
                       'maximum_accepted_position_drift_px': maximum_drift,
                       'independent_sources': {s: sum(sp == s for sp, _ in values) for s in ('train','val')}})
    readiness = read('scale_learning_readiness.json')
    for key in ('Scale_SFT_started', 'DPO_started', 'official_hidden_data_used', 'smoke_reexecuted'):
        if readiness[key]:
            raise ValueError('execution boundary violated: '+key)
    return {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'validation_script_sha256': sha(__file__), 'CUDA_model_or_optimizer_called': False,
            'GPU_seconds': seconds, 'completed_view_frames': len(rows), 'groups': checks,
            'scope': 'independent source-mean/Oracle reconstruction plus sealed evidence contracts; no new model inference'}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    result = validate(args.root)
    (args.root/'validation_semantic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
