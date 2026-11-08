"""Train-only calibration of Scale bins and geometric thresholds.

An immutable source hash split inside Train separates fitting from calibration.
Validation and Test labels are rejected, including for bin-count selection.
No diagnostic GT-centered crop is ever substituted for a P1 center.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np

from aic.scale_geometry import checked_scale_action, box_iou_xywh, maximum_window

SEED=20261108


def independent_key(row):
    identity=(row.get('image_sha256') if row['dataset_source']=='GNMC' else None)
    return row['dataset_source'],str(row.get('independent_source_id') or identity or row['source_id'])


def weighted_quantiles(values, weights, quantiles):
    values=np.asarray(values,dtype=float);weights=np.asarray(weights,dtype=float)
    if len(values)!=len(weights) or not len(values) or (weights<=0).any():
        raise ValueError('aligned positive quantile weights required')
    order=np.argsort(values,kind='stable')
    centers=(np.cumsum(weights[order])-.5*weights[order])/weights.sum()
    return np.interp(quantiles,centers,values[order])


def source_weights(rows):
    counts=Counter(independent_key(r) for r in rows)
    datasets=Counter(k[0] for k in counts)
    return np.array([1/(len(datasets)*datasets[r['dataset_source']]*counts[independent_key(r)])
                     for r in rows])


def deploy_iou_matrix(rows, scales, human_center=False):
    """Vectorized exact integer AIC projection with the same 1 px drift guard."""
    if not rows or not scales:
        raise ValueError('nonempty samples and scales required')
    if any(r['split']!='train' for r in rows):
        raise ValueError('calibration accepts Train labels only')
    if any(not 0<s<=1 for s in scales):
        raise ValueError('illegal scale')
    W=np.array([r['image_width'] for r in rows],dtype=float)[:,None]
    H=np.array([r['image_height'] for r in rows],dtype=float)[:,None]
    rw=np.array([r['target_ratio'][0] for r in rows],dtype=float)[:,None]
    rh=np.array([r['target_ratio'][1] for r in rows],dtype=float)[:,None]
    gt=np.array([r['gt_crop_xywh'] for r in rows],dtype=float)
    centers=(gt[:,:2]+gt[:,2:]/2 if human_center else np.array([r['center_xy'] for r in rows],dtype=float))
    w=np.maximum(1,np.floor(np.minimum(W,H*rw/rh)*np.array(scales)[None,:]+1e-9))
    h=w*rh/rw
    x=np.minimum(np.maximum(np.rint(centers[:,0,None]-w/2),0),W-w)
    y=np.minimum(np.maximum(np.rint(centers[:,1,None]-h/2),0),np.floor(H-h+1e-9))
    drift=np.maximum(np.abs(x+w/2-centers[:,0,None]),np.abs(y+h/2-centers[:,1,None]))
    valid=(drift<=1)&(h<=H+1e-9)
    gx,gy,gw,gh=(gt[:,i,None] for i in range(4))
    inter=np.maximum(0,np.minimum(x+w,gx+gw)-np.maximum(x,gx))*np.maximum(0,np.minimum(y+h,gy+gh)-np.maximum(y,gy))
    iou=inter/(w*h+gw*gh-inter)
    return np.where(valid,iou,-1.),drift


def greedy_bins(matrix, scales, weights, count):
    if count<2 or count>len(scales):
        raise ValueError('invalid bin count')
    chosen=[scales.index(1.)]
    current=np.maximum(matrix[:,chosen[0]],0.)
    while len(chosen)<count:
        gains=(np.maximum(matrix,current[:,None])-current[:,None])*weights[:,None]
        values=gains.sum(0)
        values[chosen]=-1
        index=int(np.argmax(values))
        chosen.append(index)
        current=np.maximum(current,matrix[:,index])
    return sorted(scales[i] for i in chosen)


def source_interval(values, rows, repeats=1000):
    groups=defaultdict(list)
    for row,v in zip(rows,values):
        groups[independent_key(row)].append(float(v))
    by_dataset=defaultdict(list)
    for (dataset,_),v in groups.items():by_dataset[dataset].append(float(np.mean(v)))
    rng=np.random.default_rng(SEED)
    report={}
    for dataset,values in by_dataset.items():
        v=np.asarray(values)
        boot=[rng.choice(v,len(v),replace=True).mean() for _ in range(repeats)]
        report[dataset]=dict(source_count=len(v),source_macro_mean=float(v.mean()),
                             ci95=np.quantile(boot,[.025,.975]).tolist(),
                             positive_source_fraction=float((v>1e-9).mean()),
                             negative_source_fraction=float((v< -1e-9).mean()),
                             source_standard_deviation=float(v.std(ddof=1)) if len(v)>1 else 0.)
    return report


def calibrate(rows, old_bins):
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('only nonempty Train rows can fit bins/thresholds')
    if any(r.get('position_kind')!='P1_final_deployed' for r in rows):
        raise ValueError('actual deployed P1 centers required')
    # Human crop scale supplies candidate support. Evaluation uses actual P1
    # centers, source-balanced within each dataset; no frame-count weighting.
    source_groups=defaultdict(list)
    for i,row in enumerate(rows):source_groups[independent_key(row)].append(i)
    dataset_sources=defaultdict(list)
    for key in source_groups:dataset_sources[key[0]].append(key)
    fit_keys=set();cal_keys=set()
    for dataset,keys in dataset_sources.items():
        if len(keys)<10:raise ValueError('at least ten independent Train sources per dataset required')
        keys.sort(key=lambda x:hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest())
        n=max(2,len(keys)//5)
        cal_keys.update(keys[:n]);fit_keys.update(keys[n:])
    fit_indices=np.array([i for i,r in enumerate(rows) if independent_key(r) in fit_keys])
    cal_indices=np.array([i for i,r in enumerate(rows) if independent_key(r) in cal_keys])
    fit_rows=[rows[i] for i in fit_indices];cal_rows=[rows[i] for i in cal_indices]
    ratios=np.array([min(1.,r['gt_crop_xywh'][2]/maximum_window(r['image_width'],r['image_height'],r['target_ratio'])[0])
                     for r in fit_rows])
    candidates=sorted(set([1.,*old_bins,*np.linspace(.1,1,91).tolist(),
                           *weighted_quantiles(ratios,source_weights(fit_rows),np.linspace(0,1,33)).tolist()]))
    candidates=[float(x) for x in candidates if 0<x<=1]
    matrix,_=deploy_iou_matrix(rows,candidates)
    fitted8=greedy_bins(matrix[fit_indices],candidates,source_weights(fit_rows),8)
    fitted16=greedy_bins(matrix[fit_indices],candidates,source_weights(fit_rows),16)
    def best(bins):return np.maximum(matrix[:,[candidates.index(x) for x in bins]].max(1),0)
    old=best(old_bins);new8=best(fitted8);new16=best(fitted16)
    # Geometry discretization sensitivity: ±one pixel in a human box changes
    # IoU. This is a quantization floor, NOT an estimate of annotator noise.
    pixel_noise=[]
    for r in cal_rows:
        x,y,w,h=r['gt_crop_xywh']
        delta=max(1-box_iou_xywh([x,y,w,h],[x+dx,y+dy,w,h]) for dx,dy in [(1,0),(0,1),(-1,0),(0,-1)])
        pixel_noise.append(delta)
    resolution_floor=float(weighted_quantiles(pixel_noise,source_weights(cal_rows),[.95])[0])
    additional=source_interval((new16-new8)[cal_indices],cal_rows)
    # Prefer 16 only when every calibration dataset has a confidently positive
    # benefit exceeding the declared pixel-resolution floor.
    use16=all(v['ci95'][0]>resolution_floor for v in additional.values())
    bins=fitted16 if use16 else fitted8
    human,_=deploy_iou_matrix(cal_rows,bins,human_center=True)
    human_best=np.maximum(human.max(1),0)
    threshold=float(weighted_quantiles(human_best,source_weights(cal_rows),[.05])[0])
    if not 0<threshold<=1:raise ValueError('train human-box attainability threshold is degenerate')
    best_values=best(bins)
    labels=matrix[:,[candidates.index(x) for x in bins]].argmax(1)
    valid=best_values>=threshold
    weights=source_weights(rows)
    frequencies=np.array([weights[valid & (labels==i)].sum() for i in range(len(bins))])
    frequencies/=max(float(frequencies.sum()),1e-30)
    counts=Counter(independent_key(r) for r,v in zip(rows,valid) if v)
    effective_sources=len(counts)
    # Bound follows the empirical label majority plus source-sampling SE;
    # an essentially constant training target is reported as a blocker.
    majority=float(frequencies.max())
    collapse_bound=min(.999,majority+1.96*math.sqrt(majority*(1-majority)/max(1,effective_sources)))
    delta=max(1e-6,2*resolution_floor)
    contrast=source_interval((best_values-matrix[:,candidates.index(1.)])[cal_indices],cal_rows)
    minima={}
    for dataset,info in contrast.items():
        n=max(2,math.ceil((1.96*info['source_standard_deviation']/delta)**2))
        eligible=sum(key[0]==dataset for key in counts)
        minima[dataset]=dict(minimum_independent_sources=n,available_train_sources=len(dataset_sources[dataset]),
                             eligible_label_train_sources=eligible,
                             precision_target=delta,method='normal-approx source-effect mean CI half-width',
                             minimum_pairs=n,passed=eligible>=n)
    return dict(status='calibrated_from_train_only',values=bins,count=len(bins),
                fitting_sources=len(fit_keys),calibration_sources=len(cal_keys),
                fit_source_keys=[list(x) for x in sorted(fit_keys)],
                calibration_source_keys=[list(x) for x in sorted(cal_keys)],
                validation_or_test_used=False,old_bins=list(old_bins),fitted8=fitted8,fitted16=fitted16,
                old_max_adjacent_gap=max(np.diff(old_bins)),new_max_adjacent_gap=max(np.diff(bins)),
                fitted8_minus_old=source_interval((new8-old)[cal_indices],cal_rows),
                fitted16_minus_fitted8=additional,selected_count_reason='16 only if all calibration dataset CI lower bounds exceed resolution floor',
                quantization_resolution_floor=resolution_floor,geometry_preference_delta_iou=delta,
                minimum_position_conditioned_iou=threshold,
                mismatch_threshold_basis='Train calibration 5th percentile attainable IoU at human centers; diagnostic only, no P1 center substitution',
                gt_label_weighted_frequencies=frequencies.tolist(),majority_scale_upper_bound=collapse_bound,
                train_label_majority=majority,scale_collapse_in_supervision=effective_sources==0 or majority>=.999,
                train_valid_label_count=int(valid.sum()),train_valid_label_fraction=float(valid.mean()),
                train_valid_label_source_dataset_macro=float(weights[valid].sum()),
                source_stability_requirements=minima,calibration_oracle_over_baseline=contrast,
                minimum_samples_gate_passed=all(v['passed'] for v in minima.values()))
