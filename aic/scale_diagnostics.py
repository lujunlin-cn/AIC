"""CPU-only Scale diagnostics. No model loading, training, or CUDA imports.

The sampling unit is an independent public source. Historical raw-center pilots
and final-P1 deployment results must never share a metric namespace.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from aic.scale_geometry import (
    box_iou_xywh, checked_scale_action, deployed_center, maximum_window,
)

SEED = 20261108


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return (row['dataset_source'], str(row['source_id']), str(row['image_id']),
            tuple(row['target_ratio']))


def independent_source(row):
    return str(row.get('independent_source_id', row['source_id']))


def arm(row):
    ratio = row['target_ratio']
    if ratio == [9, 16]:
        return 'A'
    if ratio == [16, 9]:
        return 'B'
    raise ValueError('unexpected target ratio')


def validate_labels(rows):
    seen = set()
    source_splits, image_splits = defaultdict(set), defaultdict(set)
    for r in rows:
        if r['dataset_source'] not in {'LIVE_YT_VC', 'GNMC'} or r['split'] not in {'train', 'val'}:
            raise ValueError('only public LIVE/GNMC Train/Val accepted')
        if key(r) in seen:
            raise ValueError('duplicate public label')
        seen.add(key(r))
        source_splits[(r['dataset_source'], str(r['source_id']))].add(r['split'])
        image_splits[r['image_sha256']].add(r['split'])
        w, h = r['image_width'], r['image_height']
        x, y, bw, bh = r['gt_crop_xywh']
        if not np.isfinite([w, h, x, y, bw, bh]).all() or min(w, h, bw, bh) <= 0:
            raise ValueError('nonfinite or degenerate annotation')
        # GNMC's original annotations have minor boundary/ratio imperfections;
        # preserve and count them, do not silently rewrite the human boxes.
        arm(r)
    if any(len(v) != 1 for v in source_splits.values()):
        raise ValueError('source crosses train/validation')
    if any(len(v) != 1 for v in image_splits.values()):
        raise ValueError('identical image bytes cross train/validation')


def join_final_positions(labels, positions, parent_sha):
    """Fail closed on identities and selected-cohort coverage; never infer centers from GT."""
    validate_labels(labels)
    label_map = {key(r): r for r in labels}
    seen, result = set(), []
    for p in positions:
        k = key(p)
        if k in seen or k not in label_map:
            raise ValueError('duplicate/unknown position key')
        seen.add(k)
        r = label_map[k]
        for f in ('split', 'image_sha256', 'image_width', 'image_height'):
            if p.get(f) != r[f]:
                raise ValueError('position/label mismatch: ' + f)
        if (p.get('parent_identity_sha256') != parent_sha
                or p.get('position_kind') != 'P1_final_deployed'):
            raise ValueError('not an identity-bound final P1 center')
        box = p['p1_aic_bbox_xyw']
        center = deployed_center(box, r['target_ratio'])
        base = checked_scale_action(r['image_width'], r['image_height'], r['target_ratio'], center, 1.)
        if not base['valid'] or list(base['aic_bbox_xyw']) != list(box):
            raise ValueError('Scale=1 does not exactly reproduce original final P1 bbox')
        if (not p.get('selected_frames_sha256') or not p.get('path_cache_sha256')
                or p.get('qp_floor') != .15):
            raise ValueError('missing immutable frame-set/path-cache/QP identity')
        if r['split']=='val' and p.get('p1_validation_source_disjoint_verified') is not True:
            raise ValueError('P1 validation-source training exposure not verified')
        result.append(dict(r, center_xy=list(center), p1_aic_bbox_xyw=box,
                           position_kind='P1_final_deployed'))
    if not result:
        raise ValueError('empty P1 cache')
    # A supplied cache must completely cover each declared source/view cohort,
    # rather than cherry-pick easy frames from that source.
    cohorts = {(r['dataset_source'], r['source_id'], tuple(r['target_ratio'])) for r in result}
    expected = {key(r) for r in labels if (r['dataset_source'], r['source_id'],
                tuple(r['target_ratio'])) in cohorts}
    if seen != expected:
        raise ValueError('incomplete source/view position coverage')
    return result


def source_partition(source_ids):
    """Fixed 50:50 validation tune/confirmation split before examining any IoU."""
    ordered = sorted(set(source_ids), key=lambda s: hashlib.sha256(
        f'scale-diagnostic-{SEED}:{s}'.encode()).hexdigest())
    if len(ordered) < 4:
        raise ValueError('at least four independent validation sources required')
    return ordered[:len(ordered)//2], ordered[len(ordered)//2:]


def paired_interval(delta, repeats=2000):
    d = np.asarray(delta, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError('finite source effects required')
    rng = np.random.default_rng(SEED)
    boot = np.concatenate([d[rng.integers(len(d), size=(min(100, repeats-i), len(d)))].mean(1)
                           for i in range(0, repeats, 100)])
    return {'delta_source_macro_iou': float(d.mean()),
            'paired_source_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'sources': len(d), 'replicates': repeats, 'seed': SEED}


def describe(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'sd': float(a.std()),
            'min': float(a.min()), 'p05': float(np.quantile(a, .05)),
            'median': float(np.median(a)), 'p95': float(np.quantile(a, .95)),
            'max': float(a.max())}


def source_matrix(values, rows):
    ids = sorted({independent_source(r) for r in rows})
    grouped = defaultdict(list)
    for r, v in zip(rows, values):
        grouped[independent_source(r)].append(v)
    return ids, np.array([np.mean(grouped[s], axis=0) for s in ids])


def constant_and_oracle(rows, scales, bins, repeats=2000):
    """One dataset and geometry at a time; caller labels raw vs deployed scope."""
    if any(r['split'] != 'val' for r in rows):
        raise ValueError('constant selection accepts independent Val sources only')
    if len({(r['dataset_source'], arm(r)) for r in rows}) != 1:
        raise ValueError('do not mix datasets or A/B')
    scales = sorted(set(map(float, [*scales, *bins, 1.])))
    if any(not 0 < s <= 1 for s in scales) or len(set(bins)) != 8:
        raise ValueError('need eight distinct legal Scale Actions')
    ious, boxes, drifts, gt_ious, errors, jitter, actual_scales = [], [], [], [], [], [], []
    for r in rows:
        W, H, ratio, c = r['image_width'], r['image_height'], r['target_ratio'], r['center_xy']
        gt = r['gt_crop_xywh']
        actions = [checked_scale_action(W, H, ratio, c, s) for s in scales]
        if not all(a['valid'] for a in actions):
            raise ValueError('constant grid changes frozen Position; no row dropping allowed')
        ious.append([box_iou_xywh(a['crop_xywh'], gt) for a in actions])
        boxes.append([tuple(a['aic_bbox_xyw']) for a in actions])
        actual_scales.append([a['crop_xywh'][2]/maximum_window(W,H,ratio)[0] for a in actions])
        drifts.extend(a['position_drift_px'] for a in actions)
        gc = [gt[0]+gt[2]/2, gt[1]+gt[3]/2]
        axis = 0 if arm(r) == 'A' else 1
        errors.append([(c[axis]-gc[axis])/[W, H][axis],
                       (c[1-axis]-gc[1-axis])/[W, H][1-axis]])
        counterfactual = [checked_scale_action(W, H, ratio, gc, s) for s in bins]
        gt_ious.append([box_iou_xywh(a['crop_xywh'], gt) if a['valid'] else np.nan
                        for a in counterfactual])
        # Deliberate center perturbation is a label sensitivity diagnostic only.
        per_jitter = []
        for fraction in (-.05, -.03, -.01, .01, .03, .05):
            jc = list(c); jc[axis] = float(np.clip(jc[axis]+fraction*[W,H][axis], 0, [W,H][axis]))
            acts = [checked_scale_action(W, H, ratio, jc, s) for s in bins]
            per_jitter.append([box_iou_xywh(a['crop_xywh'], gt) if a['valid'] else np.nan for a in acts])
        jitter.append(per_jitter)
    scores = np.array(ious)
    ids, macro = source_matrix(scores, rows)
    _, scale_macro = source_matrix(actual_scales, rows)
    tune, confirm = source_partition(ids)
    ti, ci = [ids.index(s) for s in tune], [ids.index(s) for s in confirm]
    # Tie-break towards larger windows. Candidate order is fixed without GT.
    best = max(range(len(scales)), key=lambda j: (macro[ti,j].mean(), scales[j]))
    base = scales.index(1.)
    constants = []
    for j, scale in enumerate(scales):
        constants.append({'scale': scale, 'tune_source_macro_iou': float(macro[ti,j].mean()),
                          'confirmation_source_macro_iou': float(macro[ci,j].mean()),
                          'quantized_final_scale_source_macro':float(scale_macro[ci,j].mean()),
                          'quantized_final_scale_range':[float(np.min(np.asarray(actual_scales)[:,j])),
                                                         float(np.max(np.asarray(actual_scales)[:,j]))],
                          'vs_scale_1': paired_interval(macro[ci,j]-macro[ci,base], repeats)})
    chosen = {'scale': scales[best], 'confirmation_source_macro_iou': float(macro[ci,best].mean()),
              'vs_scale_1': paired_interval(macro[ci,best]-macro[ci,base], repeats)}
    selection = {'validation_sources':len(ids), 'frames':len(rows), 'tune_sources':tune,
                 'confirmation_sources':confirm, 'confirmation_is_fresh_unexposed_data':False,
                 'note':'Historical validation was already used in earlier work; internal confirmation only.',
                 'selection_criterion':'tune source-macro IoU; ties prefer larger Scale',
                 'ci_scope':'conditional on tune-selected constant, independent confirmation sources',
                 'best_constant':chosen, 'scale_1_confirmation_iou':float(macro[ci,base].mean()),
                 'all_constants':constants}
    bj = [scales.index(float(s)) for s in bins]
    grid = scores[:,bj]
    ranking = sorted(range(8), key=lambda j:(-macro[ti,bj[j]].mean(), -bins[j]))
    curves = {}
    for k in (1,2,4,8):
        vals = grid[:,ranking[:k]].max(1)
        _, m = source_matrix(vals,rows)
        curves[str(k)] = {'source_macro_iou':float(m[ci].mean()),
                         'gain_over_best_constant':paired_interval(m[ci]-macro[ci,best], repeats),
                         'gain_over_scale_1':paired_interval(m[ci]-macro[ci,base], repeats)}
    _, oracle_macro = source_matrix(grid.max(1),rows)
    confirm_rows = np.array([independent_source(r) in confirm for r in rows])
    grid_boxes = [[row[j] for j in bj] for row in boxes]
    duplicates = np.array([1-len(set(row))/8 for row in grid_boxes])
    _, duplicate_macro = source_matrix(duplicates,rows)
    winners = np.argmax(grid + np.asarray(bins)[None,:]*1e-14,axis=1)
    gt_grid = np.array(gt_ious)
    common = np.isfinite(gt_grid)
    common_current = np.where(common,grid,-np.inf)
    common_truth = np.where(common,gt_grid,-np.inf)
    usable = common.any(1)
    tie=np.asarray(bins)*1e-14
    changed = np.argmax(common_current+tie,1) != np.argmax(common_truth+tie,1)
    current_opt=common & (common_current>=common_current.max(1,keepdims=True)-1e-9)
    truth_opt=common & (common_truth>=common_truth.max(1,keepdims=True)-1e-9)
    disjoint_opt=~(current_opt & truth_opt).any(1)
    def rate_by_source(vals,mask):
        if not mask.any(): return None
        _, v = source_matrix(np.asarray(vals)[mask], [r for r,m in zip(rows,mask) if m])
        return float(v.mean())
    jitter_result = {}
    for j, fraction in enumerate((-.05,-.03,-.01,.01,.03,.05)):
        jgrid = np.asarray(jitter)[:,j,:]
        ok = np.isfinite(jgrid).any(1)
        local = np.where(np.isfinite(jgrid),grid,-np.inf)
        perturbed = np.where(np.isfinite(jgrid),jgrid,-np.inf)
        jitter_result[str(fraction)] = {
            'optimal_action_changed_source_macro_rate':rate_by_source(
                np.argmax(local,1)!=np.argmax(perturbed,1),ok & confirm_rows),
            'usable_confirmation_rows':int((ok & confirm_rows).sum())}
    result = {'fixed_grid_not_model_sampling':True, 'model_greedy_iou':None,
              'P1_greedy_scale_1_source_macro_iou':float(macro[ci,base].mean()),
              'model_greedy_missing_reason':'No frozen learned Scale predictions supplied; no SFT/DPO run.',
              'constant_policy_greedy':{'scale':bins[ranking[0]],
                'source_macro_iou':float(macro[ci,bj[ranking[0]]].mean())},
              'oracle_at_k':curves, 'candidate_prefix_rule':'global tune-ranked fixed actions; no per-frame GT ranking',
              'candidate_action_order':[bins[j] for j in ranking],
              'maximum_position_drift_px':float(max(drifts)),
              'candidate_duplicate_rate_source_macro':float(duplicate_macro[ci].mean()),
              'oracle_scale_distribution_frame_counts':dict(Counter(str(bins[j]) for j in winners[confirm_rows])),
              'oracle_scale_distribution_source_macro':{
                  str(s):rate_by_source(winners==j,confirm_rows) for j,s in enumerate(bins)},
              'position_error_normalized':{
                  'active_axis_abs_source_macro':rate_by_source(np.abs(np.array(errors)[:,0]),confirm_rows),
                  'inactive_axis_abs_source_macro':rate_by_source(np.abs(np.array(errors)[:,1]),confirm_rows)},
              'position_vs_scale':{
                  'optimal_action_changed_at_GT_center_source_macro_rate':rate_by_source(changed,usable & confirm_rows),
                  'disjoint_optimal_action_sets_source_macro_rate':rate_by_source(disjoint_opt,usable & confirm_rows),
                  'comparison':'same mutually legal actions at actual and GT centers; GT center is counterfactual only',
                  'usable_confirmation_rows':int((usable & confirm_rows).sum()),
                  'excluded_confirmation_rows_no_common_action':int((~usable & confirm_rows).sum()),
                  'center_jitter_sensitivity':jitter_result},
              'confirmation_sources':confirm}
    return selection,result


def center_bias(rows, repeats=2000):
    """Train vs Val center optimism is observational, not proof of memorization."""
    output={}
    for dataset,branch in sorted({(r['dataset_source'],arm(r)) for r in rows}):
        values={}; groups={}
        for split in ('train','val'):
            cohort=[r for r in rows if (r['dataset_source'],arm(r),r['split'])==(dataset,branch,split)]
            if not cohort:continue
            axis=0 if branch=='A' else 1
            errors=[]
            for r in cohort:
                gt=r['gt_crop_xywh'];human=gt[axis]+gt[axis+2]/2
                errors.append(abs(r['center_xy'][axis]-human)/[r['image_width'],r['image_height']][axis])
            ids,means=source_matrix(errors,cohort);groups[split]=means
            values[split]={'absolute_active_axis_error_source_macro':float(means.mean()),
                           'source_error_distribution':describe(means),'independent_sources':len(ids)}
        if set(groups)=={'train','val'}:
            rng=np.random.default_rng(SEED)
            t,v=groups['train'],groups['val'];boot=[]
            for _ in range(repeats):
                boot.append(float(v[rng.integers(len(v),size=len(v))].mean()-t[rng.integers(len(t),size=len(t))].mean()))
            values['validation_minus_train_error']={'delta':float(v.mean()-t.mean()),
                'independent_source_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
                'interpretation':'positive suggests center optimism; domain/source differences can also explain this contrast'}
        else:values['validation_minus_train_error']=None
        output[f'{dataset}/{branch}']=values
    return output


def human_distributions(rows):
    result = {}
    for dataset, branch, split in sorted({(r['dataset_source'],arm(r),r['split']) for r in rows}):
        cohort=[r for r in rows if (r['dataset_source'],arm(r),r['split'])==(dataset,branch,split)]
        scales=[]; ratio_errors=[]; outside=0
        for r in cohort:
            W,H=r['image_width'],r['image_height']; x,y,w,h=r['gt_crop_xywh']
            scales.append(w/maximum_window(W,H,r['target_ratio'])[0])
            ratio_errors.append(abs(h-w*r['target_ratio'][1]/r['target_ratio'][0]))
            outside+=int(x<0 or y<0 or x+w>W+1e-6 or y+h>H+1e-6)
        ids,means=source_matrix(np.array(scales),cohort)
        group_mean=dict(zip(ids,means))
        within=np.array([s-group_mean[independent_source(r)] for r,s in zip(cohort,scales)])
        total=float(np.var(scales))
        result[f'{dataset}/{branch}/{split}']={
            'source_mean_scale':describe(means), 'row_scale':describe(scales),
            'source_macro_mean_ci95':paired_interval(means)['paired_source_bootstrap_ci95'],
            'within_source_variance_row_weighted':float(np.mean(within**2)),
            'between_source_fraction_of_row_variance':float(1-np.mean(within**2)/total) if total else None,
            'between_source_fraction_is_not_annotator_noise':True,
            'scale_above_1_rows':int(np.sum(np.asarray(scales)>1)),
            'ratio_error_px':describe(ratio_errors), 'out_of_bounds_human_box_rows':outside,
            'transposed_geometry_augmentation':all(r['transposed_geometry_augmentation'] for r in cohort)}
    return result
