"""CPU audits; unknown parent positions stay unknown, never replaced by GT centers."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from aic.scale_geometry import (box_iou_xywh, checked_scale_action, map_box_to_scale_id,
                                maximum_window)
from aic.scale_preferences import fixed_center_sft_target, gaicd_scale_preferences


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {'count': len(values), 'min': values[0], 'median': statistics.median(values),
            'mean': statistics.mean(values), 'p05': values[int(.05 * (len(values)-1))],
            'p95': values[int(.95 * (len(values)-1))], 'max': values[-1]}


def repair_sacd_splits(root: Path) -> dict:
    """Preserve held-out membership: test > val > train. No annotation-row shuffle."""
    entries = defaultdict(dict)
    original = {}
    for split in ('train', 'val', 'test'):
        p = root / f'{split}.json'
        original[split] = sha256(p)
        for rel in json.loads(p.read_text())['jsons']:
            stem = Path(rel).stem
            image_id, subject_id = stem.rsplit('_', 1)
            entries[image_id][stem] = {'annotation': rel, 'subject_id': subject_id}
    membership = defaultdict(set)
    for split in ('train', 'val', 'test'):
        for rel in json.loads((root / f'{split}.json').read_text())['jsons']:
            membership[Path(rel).stem.rsplit('_', 1)[0]].add(split)
    groups = []
    for image_id in sorted(entries):
        split = next(s for s in ('test', 'val', 'train') if s in membership[image_id])
        groups.append({'image_id': image_id, 'split': split,
                       'subjects': list(entries[image_id].values())})
    image_paths = {p.stem:p for p in (root / 'images').glob('*') if p.is_file()}
    by_hash = defaultdict(list)
    missing_images = []
    for group in groups:
        path = image_paths.get(group['image_id'])
        if path is None:
            missing_images.append(group['image_id'])
        else:
            digest = sha256(path)
            group['image_sha256'] = digest
            by_hash[digest].append(group)
    priority = {'train':0, 'val':1, 'test':2}
    for identical in by_hash.values():
        split = max((g['split'] for g in identical), key=priority.get)
        canonical = min(g['image_id'] for g in identical)
        for group in identical:
            group['split'] = split
            group['source_group_id'] = canonical
    payload = {'image_hash_audit_complete': not missing_images, 'missing_images': missing_images,
               'identical_image_groups': sum(len(v)>1 for v in by_hash.values()),
               'policy': 'preserve_heldout_priority_test_then_val_then_train_no_randomization',
               'original_split_sha256': original, 'groups': groups,
               'source_counts': dict(Counter(g['split'] for g in groups)),
               'subject_counts': {s: sum(len(g['subjects']) for g in groups if g['split'] == s)
                                  for s in ('train', 'val', 'test')},
               'source_disjoint': True, 'test_labels_used': False}
    payload['manifest_content_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def audit_media(rows: list[dict], root: Path, verify_hashes: bool = True) -> dict:
    missing, mismatch = [], []
    for row in rows:
        p = (root / row['input_image']).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError('media path escapes dataset root')
        if not p.is_file():
            missing.append(row['input_image'])
        elif verify_hashes and sha256(p) != row['image_sha256']:
            mismatch.append(row['input_image'])
    return {'expected': len(rows), 'missing_count': len(missing), 'hash_mismatch_count': len(mismatch),
            'missing_examples': missing[:10], 'mismatch_examples': mismatch[:10],
            'complete': not missing and not mismatch}


def position_conditioned_coverage(rows: list[dict], bins: list[float], threshold: float) -> dict:
    """Train-only coverage. Invalid clamp actions never compete for the SFT label."""
    groups = defaultdict(list)
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('position-conditioned label audit is train-only')
        groups[(row['dataset_source'], tuple(row['target_ratio']))].append(row)
    result = {}
    for (dataset, ratio), items in groups.items():
        statuses, scale_counts, sources = Counter(), Counter(), set()
        qmax, gain, invalid_actions, per_bin = [], [], 0, defaultdict(list)
        categories = Counter()
        for row in items:
            target = fixed_center_sft_target(row, bins, threshold)
            statuses[target['label_status']] += 1
            invalid_actions += target['invalid_action_count']
            if target['label_status'] == 'ok':
                scale_counts[target['scale_id']] += 1
                sources.add(row['source_id'])
            valid_scores = []
            for i, scale in enumerate(bins):
                action = checked_scale_action(row['image_width'], row['image_height'], ratio, row['center_xy'], scale)
                if action['valid']:
                    value = box_iou_xywh(action['crop_xywh'], row['gt_crop_xywh'])
                    per_bin[i].append(value)
                    valid_scores.append(value)
            if not valid_scores:
                categories['position_shift_invalid'] += 1
                continue
            best = max(valid_scores)
            qmax.append(best)
            full = checked_scale_action(row['image_width'], row['image_height'], ratio, row['center_xy'], 1.)
            base = box_iou_xywh(full['crop_xywh'], row['gt_crop_xywh']) if full['valid'] else None
            if base is not None:
                gain.append(best-base)
            if best < threshold:
                # A necessary diagnostic, not a claim that the learned Position is wrong:
                # compare to the same bin grid at the human-box center.
                gt = row['gt_crop_xywh']
                gt_center = [gt[0]+gt[2]/2, gt[1]+gt[3]/2]
                ideal = [a for scale in bins if (a := checked_scale_action(row['image_width'], row['image_height'], ratio, gt_center, scale))['valid']]
                oracle = max((box_iou_xywh(a['crop_xywh'], gt) for a in ideal), default=0.)
                categories['position_limited' if oracle >= threshold else 'position_and_scale_grid_or_ratio_mismatch'] += 1
            elif base is not None and best > base + 1e-9:
                categories['fixed_position_scale_can_improve'] += 1
            else:
                categories['fixed_position_no_grid_gain'] += 1
        count = len(items)
        result[f'{dataset}:{ratio[0]}:{ratio[1]}'] = {
            'original_samples': count, 'p1_center_valid_rate': 1-statuses['position_shift_invalid']/count,
            'position_shift_invalid_samples': statuses['position_shift_invalid'],
            'invalid_projected_actions': invalid_actions,
            'position_conditioned_max_iou': summary(qmax), 'oracle_over_scale1_gain': summary(gain),
            'position_mismatch_rate': statuses['position_mismatch']/count,
            'valid_scale_sft_labels': statuses['ok'], 'valid_label_sources': len(sources),
            'gt_bin_distribution': dict(scale_counts), 'iou_by_bin': {str(i):summary(v) for i,v in per_bin.items()},
            'diagnostic_categories': dict(categories), 'threshold': threshold,
            'threshold_fit_split': 'train', 'gt_center_used_for_training_or_label': False}
    return {'status': 'audited', 'by_dataset_ratio': result, 'validation_labels_used': False}


def gaicd_retention(rows: list[dict], bins: list[float], minimum_mos_delta: float,
                     tolerance_px: float = 1.0) -> dict:
    counts = Counter(); reasons = Counter(); mos_values = []; mos_bins = defaultdict(list)
    image_counts = {}; source_pairs = Counter(); unknown_centers = 0
    for row in rows:
        if row.get('split') != 'train' or row.get('release') != 'journal':
            raise ValueError('retention audit requires GAICD Journal train only')
        center = row.get('center_xy'); kept = 0
        if center is None: unknown_centers += 1
        W,H = row['image_width'], row['image_height'];rw,rh = row['target_ratio']
        for candidate in row['candidates']:
            counts['total_candidates'] += 1
            mos = candidate.get('mos')
            if not isinstance(mos,(float,int)) or not math.isfinite(mos) or mos in (-2,-1):
                reasons['unrated_or_invalid_mos'] += 1;continue
            counts['rated_candidates'] += 1
            if not 1 <= mos <= 5:
                reasons['mos_out_of_1_to_5_range'] += 1;continue
            counts['valid_mos_candidates'] += 1; mos_values.append(mos)
            box = candidate.get('crop_xywh');x,y,w,h=box
            if not candidate.get('valid',True) or min(w,h)<=0 or x<0 or y<0 or x+w>W+1e-6 or y+h>H+1e-6:
                reasons['invalid_geometry']+=1;continue
            counts['valid_geometry_candidates']+=1
            if abs(h-w*rh/rw)>tolerance_px:
                reasons['target_ratio_mismatch']+=1;continue
            counts['target_ratio_matched']+=1
            if center is None:continue
            if max(abs(x+w/2-center[0]),abs(y+h/2-center[1]))>tolerance_px:
                reasons['position_center_mismatch']+=1;continue
            counts['position_center_matched']+=1
            scale_id=map_box_to_scale_id(box,W,H,tuple(row['target_ratio']),tuple(center),bins,tolerance_px)
            if scale_id is None:reasons['scale_bin_or_position_shift_mismatch']+=1;continue
            counts['scale_bin_matched']+=1;kept+=1;mos_bins[scale_id].append(mos)
        image_counts[row['image_id']] = kept if center is not None else None
        if center is not None:
            source_pairs[row['source_id']] += len(gaicd_scale_preferences(row,bins,minimum_mos_delta,100000))
    for key in ['total_candidates','rated_candidates','valid_geometry_candidates','target_ratio_matched','position_center_matched','scale_bin_matched']:
        counts.setdefault(key,0)
    out = dict(counts)
    out.update(status='blocked_missing_P1_centers' if unknown_centers else 'audited',
               images=len(rows), images_without_p1_center=unknown_centers,
               valid_preference_pairs=sum(source_pairs.values()) if not unknown_centers else None,
               mos_distribution=summary(mos_values),mos_by_scale_bin={str(k):summary(v) for k,v in mos_bins.items()},
               effective_candidates_by_image=image_counts,preference_pairs_by_source=dict(source_pairs),
               filter_reasons={k:{'count':v,'fraction_of_total':v/max(1,counts['total_candidates'])} for k,v in reasons.items()},
               target_ratio_tolerance_px=tolerance_px, test_or_validation_labels_used=False)
    if unknown_centers:
        out['position_center_matched']=None;out['scale_bin_matched']=None
    return out
