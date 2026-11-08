"""CPU diagnostics conditional on immutable, final-deployed P1 positions.

Train selects constants and fixed Oracle prefixes. Val evaluates them once.
Oracle uses visible public GT and is never a learned-model result. All intervals
resample independent source summaries, not frames.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np

from aic.scale_geometry import box_iou_xywh, checked_scale_action, deployed_center

SEED = 20261108
CONSTANTS = [.87, .90, .93, .95, .97, 1.]


def source_id(r):
    return (r['dataset_source'], str(r.get('independent_source_id', r['source_id'])))


def macro_array(rows, values):
    grouped = defaultdict(list)
    for r, value in zip(rows, values):
        grouped[source_id(r)].append(value)
    ids = sorted(grouped)
    return ids, np.asarray([np.mean(grouped[s], axis=0) for s in ids])


def summary(values, repeats=2000):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        return {'source_count': len(a), 'mean': None, 'median': None, 'p05': None,
                'p95': None, 'bootstrap_ci95': None, 'reason': 'no complete finite source observations'}
    ci = None
    if len(a) > 1:
        rng = np.random.default_rng(SEED)
        boot = np.concatenate([a[rng.integers(len(a), size=(min(100, repeats-i), len(a)))].mean(1)
                               for i in range(0, repeats, 100)])
        ci = np.quantile(boot, [.025, .975]).tolist()
    return {'source_count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p05': float(np.quantile(a, .05)), 'p95': float(np.quantile(a, .95)),
            'bootstrap_ci95': ci, 'bootstrap_unit': 'independent source',
            'bootstrap_replicates': repeats, 'seed': SEED}


def source_summary(rows, values, repeats=2000):
    _, m = macro_array(rows, values)
    return summary(m, repeats)


def effect_summary(values, repeats=2000):
    result = summary(values, repeats)
    a = np.asarray(values, dtype=float)
    result.update(positive_source_fraction=float((a > 0).mean()) if len(a) else None,
                  negative_source_fraction=float((a < 0).mean()) if len(a) else None,
                  zero_source_fraction=float((a == 0).mean()) if len(a) else None)
    return result


def validate_join(labels, positions, manifest):
    def key(r):
        return r['dataset_source'], str(r['source_id']), str(r['image_id']), tuple(r['target_ratio'])
    lookup = {key(r): r for r in labels}
    seen, output = set(), []
    for p in positions:
        k = key(p)
        if k in seen or k not in lookup:
            raise ValueError('duplicate/unknown position record')
        seen.add(k)
        r = lookup[k]
        for field in ('split', 'image_sha256', 'image_width', 'image_height', 'frame_id'):
            if field in r and p.get(field) != r[field]:
                raise ValueError('position/label mismatch: ' + field)
        if r['split'] not in ('train', 'val'):
            raise ValueError('Test data forbidden')
        for field in ('parent_identity_sha256', 'p1_base_sha256', 'tokenizer_sha256',
                      'postprocessing_config_sha256', 'spatial_entrypoint_sha256'):
            if p.get(field) != manifest[field]:
                raise ValueError('P1 identity mismatch: ' + field)
        branch = 'portrait' if r['target_ratio'] == [9, 16] else 'vertical'
        if p.get('p1_adapter_sha256') != manifest['p1_adapter_sha256_by_route'][branch]:
            raise ValueError('P1 adapter identity mismatch')
        if (p.get('position_kind') != 'P1_final_deployed' or p.get('qp_floor') != .15
                or not p.get('path_cache_sha256') or not p.get('selected_frames_sha256')):
            raise ValueError('not final deployed P1 geometry')
        center = deployed_center(p['p1_aic_bbox_xyw'], r['target_ratio'])
        base = checked_scale_action(r['image_width'], r['image_height'], r['target_ratio'], center, 1.)
        if not base['valid'] or list(base['aic_bbox_xyw']) != p['p1_aic_bbox_xyw']:
            raise ValueError('Scale=1 does not reproduce original P1 output')
        output.append({**r, **p, 'center_xy': list(center)})
    # Every completed source is required to retain every predeclared sampled
    # frame in both views. Partial sources never enter evaluation.
    sources = {(r['dataset_source'], r['source_id']) for r in output}
    expected = {key(r) for r in labels if (r['dataset_source'], r['source_id']) in sources}
    if expected != seen:
        raise ValueError('incomplete sampled source/view')
    return output


def error_values(r):
    gt = r['gt_crop_xywh']
    c = r['center_xy']
    residual = np.asarray([c[0]-gt[0]-gt[2]/2, c[1]-gt[1]-gt[3]/2])
    norm = residual / [r['image_width'], r['image_height']]
    axis = 0 if r['target_ratio'] == [9, 16] else 1
    return {'absolute_x_px': abs(residual[0]), 'absolute_y_px': abs(residual[1]),
            'absolute_active_axis_px': abs(residual[axis]),
            'absolute_x_fraction_width': abs(norm[0]), 'absolute_y_fraction_height': abs(norm[1]),
            'absolute_active_axis_fraction': abs(norm[axis]),
            'signed_x_fraction_width': norm[0], 'signed_y_fraction_height': norm[1],
            'euclidean_px': float(np.linalg.norm(residual)), 'linf_px': float(abs(residual).max())}


def center_errors(rows, repeats=2000):
    output = {}
    for split in ('train', 'val'):
        cohort = [r for r in rows if r['split'] == split]
        values = [error_values(r) for r in cohort]
        output[split] = {name: source_summary(cohort, [v[name] for v in values], repeats)
                         for name in values[0]} if values else None
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    gap = None
    if train and val:
        _, t = macro_array(train, [error_values(r)['absolute_active_axis_fraction'] for r in train])
        _, v = macro_array(val, [error_values(r)['absolute_active_axis_fraction'] for r in val])
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(repeats):
            boot.append(v[rng.integers(len(v), size=len(v))].mean() -
                        t[rng.integers(len(t), size=len(t))].mean())
        interval = np.quantile(boot, [.025, .975]).tolist() if min(len(t), len(v)) > 1 else None
        gap = {'val_minus_train_normalized_active_axis_MAE': float(v.mean()-t.mean()),
               'independent_source_bootstrap_ci95': interval,
               'positive_gap_means_train_looks_better': True,
               'statistically_resolved_positive_gap': interval[0] > 0 if interval else None,
               'causal_in_sample_optimism_proven': False,
               'limitation': 'sampled cohorts differ in content; Val was historically used for selection; source/byte exposure audited separately'}
    output['train_validation_gap'] = gap
    output['scope'] = 'source summaries of per-frame absolute errors against public human crop centers'
    return output


def action_table(rows, scales):
    scores = np.full((len(rows), len(scales)), np.nan)
    boxes, drift, all_drift, reasons = [], [], [], defaultdict(int)
    for i, r in enumerate(rows):
        local = []
        for j, scale in enumerate(scales):
            try:
                a = checked_scale_action(r['image_width'], r['image_height'], r['target_ratio'], r['center_xy'], float(scale))
            except ValueError:
                a = {'valid': False, 'status': 'center_out_of_image'}
            if a.get('position_drift_px') is not None:
                all_drift.append(a['position_drift_px'])
            if a['valid']:
                scores[i, j] = box_iou_xywh(a['crop_xywh'], r['gt_crop_xywh'])
                drift.append(a['position_drift_px'])
                local.append(tuple(a['aic_bbox_xyw']))
            else:
                reasons[a.get('status', 'invalid')] += 1
                local.append(None)
        boxes.append(local)
    return scores, boxes, {'maximum_accepted_position_drift_px': max(drift) if drift else None,
                           'maximum_projection_drift_before_rejection_px': max(all_drift) if all_drift else None,
                           'invalid_actions': dict(reasons), 'action_checks': scores.size,
                           'valid_action_rate': len(drift)/scores.size if scores.size else None}


def constants_oracle(rows, bins, repeats=2000):
    if len(bins) != 8 or len(set(bins)) != 8:
        raise ValueError('exactly eight frozen actions required')
    if len({(r['dataset_source'], tuple(r['target_ratio'])) for r in rows}) != 1:
        raise ValueError('datasets and routes must remain separate')
    tr = np.array([r['split'] == 'train' for r in rows]); va = ~tr
    if not tr.any() or not va.any():
        reason = 'both actual Train and independent Val P1 positions required'
        return {'best_constant_scale': None, 'reason': reason}, {'oracle_at_k': None, 'reason': reason}
    scales = sorted(set(CONSTANTS + list(bins)))
    q, boxes, geometry = action_table(rows, scales)
    if not np.isfinite(q).all():
        # Never inflate a metric by dropping infeasible windows.
        return {'best_constant_scale': None, 'reason': 'infeasible frozen-center action', **geometry}, {
            'oracle_at_k': None, 'reason': 'infeasible frozen-center action', **geometry}
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'val']
    _, tm = macro_array(train, q[tr]); _, vm = macro_array(val, q[va])
    ci = [scales.index(s) for s in CONSTANTS]
    best = max(ci, key=lambda j: (tm[:, j].mean(), scales[j]))
    base = scales.index(1.)
    table = []
    for j in ci:
        table.append({'scale': scales[j], 'train_source_macro_iou': float(tm[:, j].mean()),
                      'val_source_macro_iou': summary(vm[:, j], repeats),
                      'val_gain_over_scale_1': effect_summary(vm[:, j]-vm[:, base], repeats)})
    constants = {'best_constant_scale': scales[best], 'selected_on': 'Train source-macro IoU only',
        'val_source_macro_iou': summary(vm[:, best], repeats),
        'val_gain_over_scale_1': effect_summary(vm[:, best]-vm[:, base], repeats),
        'scale_1_val_source_macro_iou': summary(vm[:, base], repeats), 'all_constants': table,
        'geometry': geometry, 'metric': 'post-P1-QP fixed-center integer AIC [x,y,w] with ratio-derived height'}
    bj = [scales.index(s) for s in bins]
    ranking = sorted(bj, key=lambda j: (-tm[:, j].mean(), -scales[j]))
    curves = {}
    for k in (1, 2, 4, 8):
        _, v = macro_array(val, q[va][:, ranking[:k]].max(1))
        curves[str(k)] = {'val_source_macro_iou': summary(v, repeats),
            'gain_over_best_constant': effect_summary(v-vm[:, best], repeats),
            'gain_over_scale_1': effect_summary(v-vm[:, base], repeats)}
    winning = np.argmax(q[:, bj] + np.asarray(bins)[None, :] * 1e-14, axis=1)
    unique = np.array([len({b[j] for j in bj if b[j] is not None}) / 8 for b in boxes])
    coverage = {str(scale): {'valid_val_fraction': source_summary(val, np.isfinite(q[va, bj[i]]), repeats),
        'oracle_optimum_val_fraction': source_summary(val, winning[va] == i, repeats),
        'oracle_optimum_val_frame_count': int((winning[va] == i).sum())} for i, scale in enumerate(bins)}
    oracle = {'GT_visible_upper_bound': True, 'learned_model_greedy': None,
        'P1_greedy_scale_1': summary(vm[:, base], repeats),
        'train_selected_constant_action_greedy': {'scale': scales[ranking[0]],
            'val_source_macro_iou': summary(vm[:, ranking[0]], repeats)},
        'oracle_at_k': curves, 'candidate_order': [scales[j] for j in ranking],
        'candidate_order_selected_on': 'Train source-macro only; fixed before Val evaluation',
        'unique_deployed_candidate_ratio': source_summary(val, unique[va], repeats),
        'per_action_coverage': coverage, 'geometry': geometry,
        'comparison_warning': 'Oracle candidates are eight frozen bins; best constant is selected from the separately declared six constants'}
    sorted_quality = np.sort(q[va][:, bj], axis=1)
    oracle['top_two_action_quality_margin'] = source_summary(
        val, sorted_quality[:, -1]-sorted_quality[:, -2], repeats)
    # GT-center calculation is a labelled counterfactual for error attribution,
    # never an input to the actual policy or a replacement for P1 centers.
    counterfactual = []
    for r in val:
        gt = r['gt_crop_xywh']
        counterfactual.append(dict(r, center_xy=[gt[0]+gt[2]/2, gt[1]+gt[3]/2]))
    cq, _, cd = action_table(counterfactual, bins)
    pq = q[va][:, bj]
    common = np.isfinite(cq) & np.isfinite(pq)
    usable = common.any(1)
    pcommon = np.where(common, pq, -np.inf); ccommon = np.where(common, cq, -np.inf)
    usable_rows = [r for r, ok in zip(val, usable) if ok]
    tie = np.asarray(bins)[None, :]*1e-14
    oracle['position_counterfactual_NOT_policy'] = {
        'GT_center_oracle_minus_P1_center_oracle_common_legal': source_summary(
            usable_rows, ccommon[usable].max(1)-pcommon[usable].max(1), repeats),
        'optimal_action_changed_when_center_replaced_with_GT': source_summary(
            usable_rows, (np.argmax(pcommon+tie, 1) != np.argmax(ccommon+tie, 1))[usable], repeats),
        'mutually_legal_val_rows': int(usable.sum()), 'counterfactual_geometry': cd,
        'warning': 'diagnostic-only visible-GT upper bound; does not prove all remaining error is Scale error'}
    return constants, oracle


def center_jitter(rows, bins, repeats=2000):
    q, _, geometry = action_table(rows, bins)
    residuals = defaultdict(list)
    for r in rows:
        if r['split'] != 'val':
            continue
        gt = r['gt_crop_xywh']; c = r['center_xy']
        residuals[source_id(r)].append(np.array([c[0]-gt[0]-gt[2]/2, c[1]-gt[1]-gt[3]/2]) /
                                       [r['image_width'], r['image_height']])
    conditions = [('active_axis_' + str(s), s) for s in (-.05, -.03, -.01, .01, .03, .05)]
    conditions += [('validation_error_matched', None)]
    output = {'original_geometry': geometry, 'conditions': {},
              'empirical_fit': 'empirical signed normalized Val residuals: uniform source, uniform frame; exclude evaluated Val source; preserve bias and magnitudes',
              'use': 'sensitivity diagnostic only; no label/action-bin/policy refit using validation residuals'}
    for name, fraction in conditions:
        perturbed, available_rows, displacement = [], [], []
        original_indices = []
        for i, r in enumerate(rows):
            center = np.asarray(r['center_xy'], dtype=float)
            dims = np.array([r['image_width'], r['image_height']])
            if fraction is None:
                ids = sorted(s for s in residuals if not (r['split'] == 'val' and s == source_id(r)))
                if not ids:
                    continue
                seed = int(hashlib.sha256(f"{SEED}:{r['source_id']}:{r['image_id']}:{r['target_ratio']}".encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                source = ids[int(rng.integers(len(ids)))]; group = residuals[source]
                delta = group[int(rng.integers(len(group)))] * dims
            else:
                axis = 0 if r['target_ratio'] == [9, 16] else 1
                delta = np.zeros(2); delta[axis] = fraction * dims[axis]
            perturbed.append(dict(r, center_xy=(center+delta).tolist()))
            available_rows.append(r); original_indices.append(i)
            displacement.append(float(abs(delta).max()))
        if not perturbed:
            output['conditions'][name] = {'result': None, 'reason': 'independent Val residual distribution unavailable'}
            continue
        jq, _, jd = action_table(perturbed, bins)
        oq = q[original_indices]
        common = np.isfinite(oq) & np.isfinite(jq)
        usable = common.any(1)
        tie = np.asarray(bins)[None, :] * 1e-14
        ob = np.where(common, oq, -np.inf); jb = np.where(common, jq, -np.inf)
        original_best = np.argmax(ob+tie, 1); new_best = np.argmax(jb+tie, 1)
        original_full_best = np.argmax(np.where(np.isfinite(oq), oq, -np.inf)+tie, 1)
        best_iou = np.where(usable, jb.max(1), np.nan)
        delta_iou = np.zeros(len(oq)); delta_iou[usable] = best_iou[usable]-ob[usable].max(1)
        holding_regret = np.zeros(len(oq))
        holding_regret[usable] = best_iou[usable]-jb[np.arange(len(jb)), original_best][usable]
        local = {}
        for split in ('train', 'val'):
            mask = np.array([r['split'] == split for r in available_rows])
            valid = mask & usable
            rr = [r for r, selected in zip(available_rows, valid) if selected]
            all_rows = [r for r, selected in zip(available_rows, mask) if selected]
            local[split] = {
                'optimal_scale_changed_rate': source_summary(rr, (original_full_best != new_best)[valid], repeats),
                'optimal_scale_changed_rate_common_legal': source_summary(rr, (original_best != new_best)[valid], repeats),
                'scale_label_stability_common_legal': source_summary(rr, (original_best == new_best)[valid], repeats),
                'oracle_iou_change_common_legal': source_summary(rr, delta_iou[valid], repeats),
                'holding_original_common_best_scale_regret': source_summary(rr, holding_regret[valid], repeats),
                'original_best_action_becomes_illegal_rate': source_summary(all_rows,
                    ~np.isfinite(jq[np.arange(len(jq)), original_full_best])[mask], repeats),
                'no_legal_window_rate': source_summary(all_rows, ~usable[mask], repeats),
                'rejected_action_fraction': source_summary(all_rows, 1-np.isfinite(jq[mask]).mean(1), repeats),
                'evaluated_rows': int(mask.sum()), 'mutually_legal_rows': int(valid.sum())}
        output['conditions'][name] = {'by_split': local, 'position_drift': jd,
            'intentional_displacement_from_original_P1_max_px': max(displacement),
            'note': 'drift is projection error relative to deliberately perturbed center; deliberate shift reported separately'}
    return output


def ridge_predict(train_x, train_y, x, weights):
    mean = np.average(train_x, axis=0, weights=weights)
    std = np.sqrt(np.average((train_x-mean)**2, axis=0, weights=weights)); std[std < 1e-9] = 1.
    a = np.c_[np.ones(len(train_x)), (train_x-mean)/std]
    b = np.c_[np.ones(len(x)), (x-mean)/std]
    penalty = np.eye(a.shape[1])*10.; penalty[0, 0] = 0
    return b @ np.linalg.solve(a.T @ (weights[:, None]*a)+penalty, a.T @ (weights[:, None]*train_y))


def content_probe(rows, bins, features, repeats=2000):
    """Fixed CPU diagnostic ridge; no tuning, no SFT, no deployment claim."""
    tr = np.array([r['split'] == 'train' for r in rows]); va = ~tr
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'val']
    if len({source_id(r) for r in train}) < 4 or len({source_id(r) for r in val}) < 4:
        return {'learned_minus_best_constant': None, 'reason': 'fewer than four actual independent Train or Val sources'}
    if any(r['image_sha256'] not in features for r in rows):
        return {'learned_minus_best_constant': None, 'reason': 'hash-verified label-blind visual descriptors missing'}
    scales = sorted(set(CONSTANTS + list(bins)))
    q, _, _ = action_table(rows, scales)
    if not np.isfinite(q).all():
        return {'learned_minus_best_constant': None, 'reason': 'infeasible fixed-center action'}
    content = np.asarray([features[r['image_sha256']] for r in rows])
    # These descriptors are transpose invariant. Original pixel bytes can be
    # shared by the paired A/B view without silently changing feature meaning.
    meta = np.array([[np.log(r['image_width']), np.log(r['image_height']),
                      r['center_xy'][0]/r['image_width'], r['center_xy'][1]/r['image_height']] for r in rows])
    bj = [scales.index(s) for s in bins]; ci = [scales.index(s) for s in CONSTANTS]
    _, tm = macro_array(train, q[tr]); _, vm = macro_array(val, q[va])
    best_constant = max(ci, key=lambda j: (tm[:, j].mean(), scales[j]))
    best_grid = max(bj, key=lambda j: (tm[:, j].mean(), scales[j]))
    counts = defaultdict(int)
    for r in train:
        counts[source_id(r)] += 1
    weights = np.array([1/counts[source_id(r)] for r in train]); weights *= len(train)/weights.sum()
    results = {}
    policies = {}
    for name, x in [('geometry_only', meta), ('geometry_plus_RGB', np.c_[meta, content])]:
        pred = ridge_predict(x[tr], q[tr][:, bj], x[va], weights)
        chosen = np.argmax(pred+np.asarray(bins)[None, :]*1e-14, 1)
        vals = q[va][np.arange(va.sum()), np.asarray(bj)[chosen]]
        _, m = macro_array(val, vals)
        policies[name] = m
        results[name] = {'source_macro_deploy_iou': summary(m, repeats),
            'learned_minus_best_constant': effect_summary(m-vm[:, best_constant], repeats),
            'learned_minus_train_selected_grid_constant': effect_summary(m-vm[:, best_grid], repeats),
            'actions': {str(s): int((chosen == i).sum()) for i, s in enumerate(bins)},
            'scale_bin_coverage': len(set(chosen.tolist()))/len(bins),
            'source_macro_largest_action_fraction': max(
                source_summary(val, chosen == i, repeats)['mean'] for i in range(len(bins)))}
    return {'model': 'fixed ridge alpha=10, completion-free CPU quality-vector diagnostic; no SFT/DPO',
            'selection': 'all fitting and standardization on Train; no validation tuning',
            'target': 'eight actual deployed IoUs at frozen P1 center; not human scale regression',
            'train_sources': len({source_id(r) for r in train}), 'val_sources': len({source_id(r) for r in val}),
            'best_constant_scale': scales[best_constant], 'train_selected_grid_constant': scales[best_grid],
            'results': results, 'incremental_RGB_over_geometry': effect_summary(
                policies['geometry_plus_RGB']-policies['geometry_only'], repeats),
            'limitation': 'bounded low-level probe is not Video-ORA Scale policy; failure does not prove all visual signal absent'}
