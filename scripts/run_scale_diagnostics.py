"""Prepare/run E0/E1/E2 on CPU. Historical raw caches never become deployment evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

from aic.scale_data import read_jsonl
from aic.scale_diagnostics import (
    SEED, arm, center_bias, constant_and_oracle, describe, digest, human_distributions,
    join_final_positions, key, source_matrix, validate_labels,
)
from aic.scale_geometry import deployed_center, maximum_window


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def clean_public_labels(rows, out):
    """Quarantine whole Train sources with exact Val bytes; preserve originals."""
    by_hash=defaultdict(list)
    for r in rows:by_hash[r['image_sha256']].append(r)
    crossing=[v for v in by_hash.values() if len({r['split'] for r in v})>1]
    excluded={(r['dataset_source'],r['source_id']) for v in crossing for r in v if r['split']=='train'}
    cleaned=[dict(r) for r in rows if (r['dataset_source'],r['source_id']) not in excluded]
    for r in cleaned:
        if r['dataset_source']=='GNMC':r['independent_source_id']='image-sha256:'+r['image_sha256']
    validate_labels(cleaned)
    path=out/'inputs/public_labels_clean.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in cleaned:f.write(json.dumps(r,sort_keys=True)+'\n')
    audit={'cross_split_identical_image_groups':len(crossing),
           'quarantined_train_sources':[list(x) for x in sorted(excluded)],
           'removed_label_views':len(rows)-len(cleaned),'remaining_label_views':len(cleaned),
           'policy':'retain Val, quarantine entire matching Train sources and all geometric views; originals unchanged',
           'clean_labels_path':str(path),'clean_labels_sha256':digest(path),
           'GNMC_statistics_unit':'image bytes; duplicate IDs grouped before Macro/bootstrap'}
    audit['historical_bin_fit_contaminated_validation_sources']=sorted({r['source_id'] for v in crossing for r in v if r['dataset_source']=='GNMC' and r['split']=='val'})
    audit['historical_bin_policy']='Keep the exact existing eight actions; exclude these validation sources from E0/E2 because their image aliases entered the historical bin fit. E1 uses cleaned Train and all independent Val images.'
    write(out/'input_isolation_audit.json',audit)
    return cleaned,audit


def prepare_inference_cohort(labels, out, isolation):
    train_sources=sorted({r['source_id'] for r in labels if r['dataset_source']=='LIVE_YT_VC' and r['split']=='train'},
        key=lambda s:hashlib.sha256(f'scale-bias-sample-{SEED}:{s}'.encode()).hexdigest())[:200]
    selected=set(train_sources)
    cohort=[r for r in labels if
        (r['dataset_source']=='LIVE_YT_VC' and (r['split']=='val' or r['source_id'] in selected))
        or (r['dataset_source']=='GNMC' and r['split']=='val'
            and r['source_id'] not in isolation['historical_bin_fit_contaminated_validation_sources'])]
    path=out/'inputs/pending_inference_cohort.jsonl'
    with path.open('w') as f:
        for r in cohort:f.write(json.dumps(r,sort_keys=True)+'\n')
    # One fixed full-video interval per public source, not inferred hidden-test
    # temporal labels. Both A/B views are explicit; no official test media.
    source_views=400*2
    calls=source_views*24*2 + 996*2*2
    plan={
        'status':'prepared_NOT_authorized_NOT_launched','new_gpu_inference_authorized':False,
        'SFT_authorized':False,'DPO_authorized':False,'optimizer_steps':0,
        'cohort_path':str(path),'cohort_sha256':digest(path),'label_views':len(cohort),
        'LIVE_train_source_ids_for_center_bias':train_sources,
        'LIVE_train_sources':200,'LIVE_validation_sources':200,
        'GNMC_validation_sources':996,'GNMC_train_GPU_inference_requested':0,
        'training_labels_sent_to_position_model':False,
        'public_temporal_contract':'one complete public video interval fixed before all Scale comparisons; native A and transposed B; this is not evaluation of learned Temporal performance',
        'model_requests_upper_bound':calls,
        'model_request_basis':'400 LIVE videos x 2 geometry views x <=24 anchors x (subject+position), plus 996 GNMC images x 2 views x (subject+position); tiny batch parity/startup checks extra',
        'estimated_model_only_GPU_hours_by_amortized_seconds_per_request':{
            str(s):round(calls*s/3600,4) for s in (.25,.5,1.)},
        'estimate_is_measured':False,
        'additional_costs':'model loading, image/video decoding, batch parity, synchronization and retries; GNMC subject-count/environment inference is not included',
        'total_budget_hours_cap':12,
        'available_budget':'unknown until the other agent stops and its actual GPU usage is reconciled; local diagnostics used zero GPU',
        'before_launch':'wait for user GPU approval; inspect live GPU/processes; reuse verified existing caches first; pilot throughput with no optimizer/smoke; reconcile shared cumulative ledger; reduce source plan before outcomes are observed or stop if projected total exceeds remaining budget',
        'do_not_use':'--smoke, run_scale_stage, bootstrap_scale_gpu, SFT/DPO forward-backward, auto-wait-and-launch, independent reward models',
        'existing_inference_entrypoint':'scripts/generate_p1_scale_positions.py (another agent owns it; not modified or executed here)',
        'cache_requirements':['dataset_source/source_id/image_id/split/ratio/dimensions/image SHA',
            'P1_final_deployed original p1_aic_bbox_xyw and parent_identity_sha256',
            'qp_floor=.15, selected_frames_sha256, path_cache_sha256, unchanged parent',
            'archive full per-source PTS/events/anchors/proposals/P1 trajectory and QP audit',
            'prove P1 training-source separation of each claimed independent validation cohort, especially GNMC old B training images'],
        'validation_exposure_field':'p1_validation_source_disjoint_verified=true per Val record, backed by a retained training-source provenance audit; generator output alone does not establish this fact',
        'CPU_followup':'CUDA_VISIBLE_DEVICES=\'\' OPENBLAS_NUM_THREADS=1 python -m scripts.run_scale_diagnostics --positions PUBLIC_POSITIONS_JSONL',
        'eight_action_enumeration_GPU_hours':0,
        'missing_visual_features':'reuse cached primary-subject box as a size proxy; subject count and environment fraction need separately validated definitions/features; do not substitute saliency-component count',
        'stop_after':'E0/E1/E2 report; no automatic SFT/DPO transition'}
    write(out/'pending_inference_plan.json',plan)
    return plan


def historical_rows(snapshot, labels):
    label_map = {key(r):r for r in labels}
    root = snapshot/'repro/liveytvc_full_retrain_20261023'
    outputs = {}
    receipts = {}
    for branch, condition, filename in [('A','a:new_a','arm_a_gate.raw.json'),
                                         ('B','b:new_b','arm_b_gate.raw.json')]:
        path = root/'eval'/filename
        receipts[filename] = digest(path)
        values = json.loads(path.read_text())[condition]
        cohort = []
        for source, frames in values.items():
            for item in frames:
                ratio = [9,16] if branch=='A' else [16,9]
                r = label_map[('LIVE_YT_VC',source,f"{source}-frame{item['frame_id']}",tuple(ratio))]
                if not item['valid']:
                    raise ValueError('historical parse failure; no silent row exclusion')
                cx,cy = json.loads(item['raw'])['crop_center']
                W,H = r['image_width'],r['image_height']
                if not (0<=cx<=1000 and 0<=cy<=1000): raise ValueError('invalid raw center')
                w,h = maximum_window(W,H,ratio)
                # Reproduce generation.parse_center's continuous maximum-window
                # projection, not the incorrect parse_center(raw,1,1) eval metric.
                x=float(np.clip(cx*W/1000-w/2,0,W-w))
                y=float(np.clip(cy*H/1000-h/2,0,H-h))
                cohort.append(dict(r,center_xy=[x+w/2,y+h/2],
                    position_kind='historical_raw_single_frame_NOT_final_P1'))
        outputs[branch] = cohort
    return outputs,receipts


def features_from_rgb(payload):
    """Label-blind low-level descriptors. Saliency components are NOT subjects."""
    from PIL import Image
    from scipy.ndimage import uniform_filter, label
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert('RGB').resize((64,64)),dtype=float)/255
    gray = rgb.mean(2)
    dx=np.diff(gray,axis=1); dy=np.diff(gray,axis=0)
    spectrum=np.fft.fft2(gray)
    logamp=np.log(np.abs(spectrum)+1e-8)
    residual=logamp-uniform_filter(logamp,size=3,mode='wrap')
    saliency=uniform_filter(np.abs(np.fft.ifft2(np.exp(residual+1j*np.angle(spectrum))))**2,size=5)
    mask=saliency>saliency.mean()+saliency.std()
    _,count=label(mask)
    weights=saliency/(saliency.sum()+1e-12)
    yy,xx=np.mgrid[:64,:64]/63
    hist=np.histogram(gray,bins=16,range=(0,1))[0]/gray.size
    entropy=-np.sum(hist*np.log(hist+1e-12))
    return [float(gray.mean()),float(gray.std()),float((rgb.max(2)-rgb.min(2)).mean()),
            float(np.abs(dx).mean()+np.abs(dy).mean()),float(entropy),
            float(mask.mean()),float(count),float((xx*weights).sum()),float((yy*weights).sum())]


def ridge_predict(train_x,train_y,x):
    mean=train_x.mean(0); std=train_x.std(0); std[std<1e-9]=1
    a=np.c_[np.ones(len(train_x)),(train_x-mean)/std]
    b=np.c_[np.ones(len(x)),(x-mean)/std]
    penalty=np.eye(a.shape[1])*10.;penalty[0,0]=0
    return b@np.linalg.solve(a.T@a+penalty,a.T@train_y)


def visual_probe(labels, archive, cache, permutations=100):
    native=[r for r in labels if r['dataset_source']=='GNMC' and not r['transposed_geometry_augmentation']]
    if not archive.is_file():
        return {'status':'blocked_missing_GNMC_images','explained_fraction':None}
    source_hash=digest(archive)
    if cache.exists():
        stored=json.loads(cache.read_text())
        if stored['archive_sha256']!=source_hash:raise ValueError('feature archive identity changed')
        feature_map=stored['features']
    else:
        feature_map={}
        with ZipFile(archive) as z:
            for i,r in enumerate(native):
                payload=z.read(r['input_image'])
                if hashlib.sha256(payload).hexdigest()!=r['image_sha256']:
                    raise ValueError('GNMC image hash mismatch')
                feature_map[r['image_id']]=features_from_rgb(payload)
                if i%1000==0: print(f'GNMC label-blind CPU features: {i}/{len(native)}',flush=True)
        write(cache,{'archive_sha256':source_hash,'feature_version':'rgb_spectral_v1_64px',
                     'features':feature_map})
    branches={}
    for branch in ('A','B'):
        # Byte-identical image IDs are one independent observation. Aggregate
        # their human targets, never count duplicates as independent sources.
        grouped=defaultdict(list)
        for r in labels:
            if r['dataset_source']=='GNMC' and arm(r)==branch:grouped[r['image_sha256']].append(r)
        rows=[v[0] for _,v in sorted(grouped.items())]
        content=np.array([feature_map[r['image_id']] for r in rows])
        if branch=='A':content[:,[7,8]]=content[:,[8,7]]
        nuisance=np.array([[np.log(r['image_width']/r['image_height']),
                            np.log(r['image_width']*r['image_height'])] for r in rows])
        y=np.array([np.mean([q['gt_crop_xywh'][2]/maximum_window(q['image_width'],q['image_height'],q['target_ratio'])[0]
                              for q in grouped[r['image_sha256']]]) for r in rows])
        tr=np.array([r['split']=='train' for r in rows]); va=~tr
        full=np.c_[nuisance,content]
        p0=np.full(va.sum(),y[tr].mean())
        pn=ridge_predict(nuisance[tr],y[tr],nuisance[va])
        pv=ridge_predict(full[tr],y[tr],full[va])
        err0=(y[va]-p0)**2; errn=(y[va]-pn)**2; errv=(y[va]-pv)**2
        denom=np.mean(err0)
        rng=np.random.default_rng(SEED)
        indices=rng.integers(va.sum(),size=(2000,va.sum()))
        incremental=(errn[indices].mean(1)-errv[indices].mean(1))/err0[indices].mean(1)
        shuffled=[]
        for _ in range(permutations):
            pred=ridge_predict(full[tr],rng.permutation(y[tr]),full[va])
            shuffled.append(float(1-np.mean((y[va]-pred)**2)/denom))
        # Source-cross-fitting is CPU regression of scale, not P1 re-training.
        folds=np.array([int(hashlib.sha256(('gnmc-fold:'+r['image_sha256']).encode()).hexdigest(),16)%5 for r in rows])
        cv=np.full(len(rows),np.nan); cvbase=np.full(len(rows),np.nan)
        for fold in range(5):
            fit=tr&(folds!=fold); check=tr&(folds==fold)
            cv[check]=ridge_predict(full[fit],y[fit],full[check]);cvbase[check]=y[fit].mean()
        branches[branch]={
            'domain':'GNMC native 16:9 images' if branch=='B' else 'GNMC transposed 9:16 geometry augmentation',
            'train_sources':int(tr.sum()),'validation_sources':int(va.sum()),
            'target':'raw human linear scale, including annotation imperfections',
            'train_mean_scale':float(y[tr].mean()),
            'validation_r2_vs_train_constant':float(1-errv.mean()/denom),
            'geometry_metadata_only_r2':float(1-errn.mean()/denom),
            'incremental_visual_r2_over_geometry':float((errn.mean()-errv.mean())/denom),
            'incremental_visual_r2_source_bootstrap_ci95':np.quantile(incremental,[.025,.975]).tolist(),
            'train_source_crossfit_r2':float(1-np.mean((y[tr]-cv[tr])**2)/np.mean((y[tr]-cvbase[tr])**2)),
            'train_target_permutation_validation_r2':describe(shuffled),
            'visual_fraction_is_not_total_explainable_fraction':True,
            'subject_size_count_environment_regression':None,
            'limitation':'spectral saliency area/components/background are low-level proxies, not measured subject size/count/environment',
            'model':'fixed ridge alpha=10, train-only standardization; five source folds; no validation tuning',
            'independent_source_bootstrap_replicates':2000}
    return {'status':'CPU_low_level_proxy_probe_complete','feature_cache_sha256':digest(cache),
            'archive_sha256':source_hash,'branches':branches,
            'feature_names':['brightness','contrast','color_range','edges','entropy',
                             'spectral_saliency_area_proxy','saliency_component_count_NOT_subject_count',
                             'saliency_center_x','saliency_center_y'],
            'LIVE_content_probe':{'status':'blocked_no_local_frame_media_or_verified_visual_features',
                                  'explained_fraction':None},
            'single_frame_total_explained_fraction':None,
            'annotator_style_variance':None,
            'noise_identifiability':'single human crop per image/frame with no repeated rater IDs cannot identify style/noise vs content variance'}


def gaicd_support(root):
    base=root/'datasets/GAICD'
    media={r['item_id']:r for r in read_jsonl(base/'processed/media.jsonl') if r['official_split']=='train' and 'journal' in r['release']}
    totals={b:{'target_ratio_rated_candidates':0,'images_with_two_or_more':0,
               'same_image_same_center_different_scale_pairs':0} for b in ('A','B')}
    for record in read_jsonl(base/'processed/annotations.jsonl'):
        image_id=record['media_uid'].split(':')[-1]
        if record['variant']!='journal' or image_id not in media:continue
        ann=json.loads(Path(record['annotation_path']).read_text())
        for branch, ratio in [('A',[9,16]),('B',[16,9])]:
            candidates=[]
            for box,mos,valid in zip(ann['boxes_xywh'],ann['mos'],ann['valid']):
                if not valid or not 1<=mos<=5:continue
                x,y,w,h=box
                if abs(h-w*ratio[1]/ratio[0])<=1:
                    candidates.append((x+w/2,y+h/2,w))
            totals[branch]['target_ratio_rated_candidates']+=len(candidates)
            totals[branch]['images_with_two_or_more']+=int(len(candidates)>=2)
            for i,a in enumerate(candidates):
                for b in candidates[i+1:]:
                    if max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1 and abs(a[2]-b[2])>1:
                        totals[branch]['same_image_same_center_different_scale_pairs']+=1
    return {'journal_train_images':len(media),'by_arm':totals,'test_annotations_opened':False,
            'P1_center_and_action_bin_filtered_pairs':None,
            'interpretation':'MOS is within-image aesthetic preference, independent of single-box IoU supervision; exact-ratio fixed-center Scale support must be established separately.'}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/hajimi2025/AIC'))
    p.add_argument('--base',type=Path,default=Path('artifacts/scale_sft_dpo'))
    p.add_argument('--positions',type=Path)
    p.add_argument('--protocol',type=Path,default=Path('artifacts/scale_sft_dpo/diagnostics/protocol.json'))
    a=p.parse_args();start=time.monotonic();base=a.base;out=base/'diagnostics'
    protocol=json.loads(a.protocol.read_text())
    labels_path=base/'manifests/public_labels/labels.jsonl'
    labels,isolation=clean_public_labels(read_jsonl(labels_path),out)
    inference_plan=prepare_inference_cohort(labels,out,isolation)
    config=json.loads((base/'config/run_config.json').read_text())
    bins=json.loads((base/'config/scale_bins.json').read_text())['values']
    scales=protocol['constant_scales']
    snapshot=base/'remote_snapshot'
    for filename in ('recovery_path_projection.py','generation.py','contracts.py'):
        path=snapshot/'semifinal_20261107/code/method'/filename
        if digest(path)!=config['parent']['inference_config']['runtime_sources'][filename]:
            raise ValueError('frozen runtime hash mismatch: '+filename)
    pilot,receipts=historical_rows(snapshot,labels)
    e0={'status':'local_preparation_complete_deployment_pending','routes':{},'historical_raw_pilot':{}}
    e2={'status':'local_preparation_complete_deployment_pending','routes':{},'historical_raw_pilot':{}}
    for branch in ('A','B'):
        print('CPU historical raw geometry',branch,flush=True)
        constants,oracle=constant_and_oracle(pilot[branch],scales,bins,protocol['bootstrap_replicates'])
        e0['historical_raw_pilot'][branch]=constants
        e2['historical_raw_pilot'][branch]=oracle
        e0['routes'][branch]={'best_constant_scale':None,'best_constant_scale_deployed_iou':None,
            'deployed_iou_gain_over_scale_1':None,'raw_iou':None,'qp_pre_post_scale_measurements':None,
            'blocker':'Need identity-bound final P1 public source centers and full path cache.'}
        e2['routes'][branch]={'greedy_iou':None,'oracle_at_k':None,'oracle_gain_over_best_constant':None,
            'blocker':'Final P1 path absent; raw pilot is not a deployment Oracle.'}
    joined=[]
    if a.positions:
        joined=join_final_positions(labels,read_jsonl(a.positions),config['parent']['identity_sha256'])
        for dataset,branch in sorted({(r['dataset_source'],arm(r)) for r in joined if r['split']=='val'}):
            cohort=[r for r in joined if r['split']=='val' and r['dataset_source']==dataset and arm(r)==branch
                    and not (dataset=='GNMC' and r['source_id'] in isolation['historical_bin_fit_contaminated_validation_sources'])]
            constant,oracle=constant_and_oracle(cohort,scales,bins,protocol['bootstrap_replicates'])
            e0['routes'][branch].setdefault('by_dataset',{})[dataset]={
                'status':'measured_post_P1_QP_scale_CPU_geometry','results':constant,
                'qp_scale_before_after':{'before':1.,'after':1.,'basis':'hash-verified fixed-width P1 QP code'},
                'scale_insertion':'after final frozen P1 QP; quantized final scale in constant table',
                'raw_before_QP_iou':None}
            e2['routes'][branch].setdefault('by_dataset',{})[dataset]=oracle
            if (branch,dataset) in {('A','LIVE_YT_VC'),('B','GNMC')}:
                e0['routes'][branch].update(best_constant_scale=constant['best_constant']['scale'],
                    best_constant_scale_deployed_iou=constant['best_constant']['confirmation_source_macro_iou'],
                    deployed_iou_gain_over_scale_1=constant['best_constant']['vs_scale_1'],
                    primary_dataset=dataset,blocker=None)
                e2['routes'][branch].update(greedy_iou=oracle['P1_greedy_scale_1_source_macro_iou'],
                    oracle_at_k=oracle['oracle_at_k'],
                    oracle_gain_over_best_constant=oracle['oracle_at_k']['8']['gain_over_best_constant'],
                    primary_dataset=dataset,blocker=None)
        e0['status']=e2['status']='provided_cohorts_evaluated_missing_domains_remain_explicit'
    for obj in (e0,e2):
        obj['historical_bin_fit_validation_exclusion']=isolation['historical_bin_fit_contaminated_validation_sources']
        obj['metric_scope']='public source-macro spatial IoU with fixed frames; not official competition Score'
        obj['historical_raw_pilot_warning']='single-frame raw centers; no smoothing, subjects, full PTS, or QP. B is transposed LIVE, not upright B confirmation; model identity of historical raw output is not hash-bound.'
    e1={'status':'partial_CPU_diagnostics_complete','input_isolation':isolation,'human_scale_distributions':human_distributions(labels),
        'visual_predictability':visual_probe(labels,a.root/'.cache_videoora_data/GNMC/GNMC.zip',out/'cache/gnmc_features.json'),
        'P1_in_sample_center_bias':{'train_center_error':None,'train_minus_validation_center_error':None,
            'severity':None,'reason':'actual train P1 output absent; train response field is human-derived, not P1 prediction'},
        'position_effect_on_optimal_scale':{b:e2['historical_raw_pilot'][b]['position_vs_scale'] for b in ('A','B')},
        'crossfit_vs_jitter':{
            'crossfit_P1':'requires source-disjoint retraining of position parents, incompatible with this frozen-P1 no-GPU round; never call a current-P1 forward pass cross-fitting',
            'crossfit_Scale_probe':'CPU GNMC ridge cross-fitting implemented; does not repair P1 in-sample center leakage',
            'jitter':'CPU sensitivity at +/-1/3/5% active-axis dimension; empirical error-matched augmentation requires train+independent-source P1 center residuals',
            'recommendation':'measure center optimism first; jitter is feasible sensitivity/robustness work, not a replacement for out-of-sample P1 centers'},
        'GAICD_MOS':gaicd_support(a.root),
        'Scale_SFT_recommendation':'defer; deployment constant/Oracle and out-of-source visual signal remain unestablished'}
    if joined:
        e1['P1_in_sample_center_bias']={'status':'supplied_cohorts_only','by_dataset_and_arm':center_bias(joined)}
        e1['position_effect_on_optimal_scale_deployed']={b:v.get('by_dataset',{}) for b,v in e2['routes'].items()}
    runtime=snapshot/'semifinal_20261107/code/method'
    qp_hash=digest(runtime/'recovery_path_projection.py')
    if qp_hash!=config['parent']['inference_config']['runtime_sources']['recovery_path_projection.py']:
        raise ValueError('original QP code hash mismatch')
    common={'created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
            'protocol_sha256':digest(a.protocol),'public_labels_sha256':digest(labels_path),
            'clean_public_labels_sha256':isolation['clean_labels_sha256'],
            'run_config_sha256':digest(base/'config/run_config.json'),
            'scale_bins_sha256':digest(base/'config/scale_bins.json'),
            'parent_identity_sha256':config['parent']['identity_sha256'],
            'historical_raw_cache_sha256':receipts,'qp_code_sha256':qp_hash,
            'new_GPU_work_executed':False,'SFT_executed':False,'DPO_executed':False,
            'official_hidden_test_data_used':False,'positions_input_sha256':digest(a.positions) if a.positions else None}
    for filename,obj in [('constant_scale_baseline.json',e0),('scale_predictability.json',e1),('scale_oracle_analysis.json',e2)]:
        obj['provenance']=common;write(out/filename,obj)
    plot_diagnostics(out,labels,e0)
    write_report(out,e0,e1,e2)
    versions={}
    for module in ['numpy','scipy','PIL','matplotlib']:
        m=__import__(module);versions[module]=m.__version__
    write(out/'execution.json',dict(common,wall_seconds=time.monotonic()-start,
        python=sys.version,platform=platform.platform(),packages=versions,
        command=[sys.executable,'-m','scripts.run_scale_diagnostics',*sys.argv[1:]],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),
        diagnostic_code_sha256={str(q):digest(q) for q in [Path(__file__),Path('aic/scale_diagnostics.py')]},
        GPU_seconds=0,remote_operations=0))
    print(json.dumps({'status':e0['status'],'wall_seconds':time.monotonic()-start,'output':str(out)},indent=2))


def plot_diagnostics(out,labels,e0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,(dataset,branch) in enumerate([('LIVE_YT_VC','A'),('GNMC','B')]):
        ax=axes[i,0]
        for split,color,style in [('train','#0072B2','-'),('val','#D55E00','--')]:
            cohort=[r for r in labels if (r['dataset_source'],arm(r),r['split'])==(dataset,branch,split)]
            values=[r['gt_crop_xywh'][2]/maximum_window(r['image_width'],r['image_height'],r['target_ratio'])[0] for r in cohort]
            _,means=source_matrix(values,cohort)
            ax.hist(means,bins=np.linspace(.25,1.05,33),weights=np.ones(len(means))/len(means),
                    histtype='step',linewidth=1.8,linestyle=style,color=color,label=f'{split}: {len(means)} sources')
        ax.set(title=f'{dataset}: human scale per independent source',xlabel='Human linear scale',ylabel='Fraction of sources')
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8);ax.grid(alpha=.2)
    for i,branch in enumerate(('A','B')):
        ax=axes[i,1]; c=e0['historical_raw_pilot'][branch];table=c['all_constants']
        x=np.array([r['scale'] for r in table]); y=np.array([r['vs_scale_1']['delta_source_macro_iou'] for r in table])
        intervals=np.array([r['vs_scale_1']['paired_source_bootstrap_ci95'] for r in table])
        ax.plot(x,y,color='#0072B2',linewidth=1.7)
        ax.fill_between(x,intervals[:,0],intervals[:,1],color='#0072B2',alpha=.18,label='pointwise source bootstrap 95% CI')
        ax.axhline(0,color='black',linewidth=.8);ax.axvline(c['best_constant']['scale'],color='#D55E00',linestyle='--',label='constant selected on tune sources')
        ax.set(title=f'{branch}: raw pilot, 100 confirmation sources'+('\nLIVE transpose; not native B confirmation' if branch=='B' else ''),
               xlabel='Requested constant scale',ylabel='Raw IoU delta from Scale=1')
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Scale diagnostics: public sources; raw pilots are not final deployment IoU',fontsize=12)
    fig.savefig(out/'diagnostic_overview.png',dpi=160,facecolor='white')
    plt.close(fig)


def write_report(out,e0,e1,e2):
    def ci(effect):
        lo,hi=effect['paired_source_bootstrap_ci95']
        return f"{effect['delta_source_macro_iou']:+.6f} [{lo:+.6f}, {hi:+.6f}]"
    final_results=[]
    for b in ('A','B'):
        for dataset, measured in e0['routes'][b].get('by_dataset',{}).items():
            best=measured['results']['best_constant']
            final_results.append(f"| {b} | {dataset} | {best['scale']:.6f} | {best['confirmation_source_macro_iou']:.6f} | {ci(best['vs_scale_1'])} |")
    current_status=('已提供缓存的 source/view 已完成最终 P1 中心后的 CPU 几何评测；见下方新增缓存表。其余域和视觉/风格诊断缺口仍保留。'
                    if final_results else '真实部署结论仍待冻结 P1 最终中心缓存。')
    lines=['# Scale 是否值得学习：本地 E0/E1/E2 诊断', '',
      '**结论：暂不启动 Scale-SFT/DPO。CPU 准备和历史缓存诊断完成；真实部署结论仍待冻结 P1 最终中心缓存。**', '',
      '本轮 GPU 使用 0 秒，没有远程操作，没有模型推理或 SFT/DPO。GNMC 图像描述符和固定 ridge 探针只在 CPU 运行。', '',
      '![人工尺度 source 分布与常数尺度 Raw 响应；阴影为逐点 source bootstrap 95% 区间](diagnostic_overview.png)', '',
      '## E0：最优常数与 Scale=1.0', '',
      '正式 A/B 最优常数、真实部署 IoU 和收益目前均为 `null`。本地历史缓存只含单帧中心输出，缺少完整 PTS、主体锚点、轨迹和平滑/QP 输出；不能将下表称为真实部署表现。B 是 LIVE 转置，不能代表原生 B。', '',
      '| 历史 Raw 几何 pilot | 最优常数 | Scale=1 IoU | 所选常数 IoU | 相对 Scale=1 的源配对 95% CI |',
      '|---|---:|---:|---:|---|']
    lines[2]='**结论：暂不启动 Scale-SFT/DPO。'+current_status+'**'
    for branch in ('A','B'):
        c=e0['historical_raw_pilot'][branch];best=c['best_constant']
        lines.append(f"| {branch} | {best['scale']:.6f} | {c['scale_1_confirmation_iou']:.6f} | {best['confirmation_source_macro_iou']:.6f} | {ci(best['vs_scale_1'])} |")
    lines+=['','A 为 1,200 帧、B 为 2,000 帧，各 200 独立 LIVE 视频源。每路按预先固定 source 哈希分成 100 tune / 100 confirmation；只在 tune 选择尺度，confirmation 配对 bootstrap 2,000 次。此前项目已使用这些验证源，因此不是新鲜最终留出。全常数表均提供与 Scale=1 的比较；其他格点只是探索性敏感性分析。', '',
      '真实 QP 代码已按 P1 配置 SHA256 验证：它要求 baseline/candidate 宽度一致，只修改活动轴位置。Scale 保持 Position 冻结的合法接法是在最终 P1 QP 后收缩。QP 前后 P1 尺度在代码上不变；逐源观测仍缺失，不能伪造测量值。最终整数宽度/比例衍生高度单独计量。原有 `integer_predictions()` 强制最大窗口，会抹掉预先插入的 Scale；本诊断使用独立的固定中心 `[x,y,w]` 导出，禁止经过该最大窗重写。', '',
      '## E1：内容可预测性、风格噪声与中心偏差', '',
      '输入隔离发现 GNMC 4 组跨 Train/Val 的完全相同图像：隔离训练源 001183、001972、003413、009560，连同转置视图共 8 行；原文件不修改。干净 GNMC Train 6,996 个 ID 合并为 6,983 个独立图像，Val 为 1,000 图像。LIVE 沿用已隔离的 1,595/200 视频源。统计和 bootstrap 均以独立图像/视频为单位，不按帧或候选行抽样。保留原有八动作，不利用 Val 重新拟合；由于历史 bin 拟合已经读过这四组重复图像的训练标签，未来 E0/E2 额外排除 GNMC Val 000667、001961、007010、008745，使用 996 个无此重叠的图像。E1 的清洁 Train 探针仍可用全部 1,000 个独立 Val 图像。', '',
      '| 人工尺度（source-mean） | Train 均值 / 中位数 | Val 均值 / 中位数 |',
      '|---|---|---|']
    if final_results:
        lines[lines.index('正式 A/B 最优常数、真实部署 IoU 和收益目前均为 `null`。本地历史缓存只含单帧中心输出，缺少完整 PTS、主体锚点、轨迹和平滑/QP 输出；不能将下表称为真实部署表现。B 是 LIVE 转置，不能代表原生 B。')]='历史单帧表始终只表示 Raw pilot；新提供的最终 P1 缓存结果单独列在文末，不能合并域或替代官方视频评测。'
    dist=e1['human_scale_distributions']
    for dataset in ('LIVE_YT_VC','GNMC'):
        for b in ('A','B'):
            t=dist[f'{dataset}/{b}/train']['source_mean_scale'];v=dist[f'{dataset}/{b}/val']['source_mean_scale']
            lines.append(f"| {dataset}/{b} | {t['mean']:.6f} / {t['median']:.6f} | {v['mean']:.6f} / {v['median']:.6f} |")
    lines+=['','人工框的比例误差、尺度超过 1 和越界数量保留在 JSON 中，没有将原始标注偷偷改成合法动作。GNMC A、LIVE B 是同步转置增强。人工平均尺度不能直接作为最佳 IoU 常数。', '',
      '| GNMC 单帧 CPU 探针 | Val R²（相对 Train 常数） | 几何元数据 R² | 增量视觉 R² [source bootstrap 95% CI] |',
      '|---|---:|---:|---|']
    probe=e1['visual_predictability']
    for b,p in probe.get('branches',{}).items():
        lo,hi=p['incremental_visual_r2_source_bootstrap_ci95']
        lines.append(f"| {b} | {p['validation_r2_vs_train_constant']:.4f} | {p['geometry_metadata_only_r2']:.4f} | {p['incremental_visual_r2_over_geometry']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines+=['','固定 ridge alpha=10，训练侧标准化、5 折 source 交叉拟合、100 次训练标签置换。特征只读 RGB：亮度、对比度、色差、边缘、熵、频谱显著性面积/连通分量/中心。约 15% 是这个受限探针对 GNMC 人工尺度的源外预测能力，约 11 个百分点是控制图像几何后的增量；不是全部可解释信息的比例，也不是可迁移到 LIVE 的证明。置换 R² 接近零。所有特征和拟合都未使用 Val 标签调参。', '',
      '**真实主体大小、主体数量、环境占比尚无已验证特征缓存，相关性结果为 null。显著性连通分量不是主体数量。LIVE 的本地帧媒体缺失，视觉解释率为 null。** 每图/帧仅有一个人工框、无逐标注者重复测量，无法识别风格噪声方差；视频内/视频间差异都混有内容和标注因素，不能解释为标注者风格。', '',
      '**P1 样本内偏差严重程度未知。** 缺实际训练源 P1 输出，训练 JSON 中 `response` 是人工派生标签，绝不能当作 P1 样本内预测。新增缓存入口可分别计算 train/val 中心误差和独立 source bootstrap 的乐观偏差，但这个比较仍可能混有 domain 差异。', '',
      '中心与尺度的反事实使用双方共同合法动作，避免边界投影暗改中心；只作为分解，GT 中心不进入学习或策略输入：', '',
      '| 历史 pilot | 活动轴中心绝对误差 / 图像维度 | GT 中心下最佳动作改变比例（source macro） |',
      '|---|---:|---:|']
    for b,o in e2['historical_raw_pilot'].items():
        lines.append(f"| {b} | {o['position_error_normalized']['active_axis_abs_source_macro']:.4f} | {o['position_vs_scale']['optimal_action_changed_at_GT_center_source_macro_rate']:.2%} |")
    lines+=['','这说明位置误差会影响最优尺度标签；不能将其统称为 Scale 学不会。±1/3/5% 活动轴中心抖动的标签敏感性已在 JSON 分开记录，这些扰动不改变正式 P1。真正交叉拟合 P1 要按源重训位置模型，本轮不执行；CPU Scale 探针交叉拟合不等于 P1 交叉拟合。抖动增强成本低，但只能在测得源外中心残差后校准，不能修复或证明样本内偏差已消失。', '',
      '## E2：八动作精确 Oracle', '',
      '| 历史 Raw pilot | 常数策略 Greedy | Oracle@1 | Oracle@2 | Oracle@4 | Oracle@8 | Oracle@8−最佳常数 [95% CI] |',
      '|---|---:|---:|---:|---:|---:|---|']
    for b,o in e2['historical_raw_pilot'].items():
        curve=o['oracle_at_k']
        lines.append(f"| {b} | {o['constant_policy_greedy']['source_macro_iou']:.6f} | "+' | '.join(f"{curve[str(k)]['source_macro_iou']:.6f}" for k in (1,2,4,8))+f" | {ci(curve['8']['gain_over_best_constant'])} |")
    lines+=['','Oracle@K 的候选顺序仅由 tune 源上的全局动作得分决定，不按每帧 GT 重排。每个候选都经整数 `[x,y,w]` 与比例衍生高度计算 IoU，按最终框去重；pilot 两路候选重复率均为 0，最大 Position Drift ≤0.5 px。动作分布、GT 中心反事实及抖动敏感性见 JSON。此处 Greedy 是调优后的**常数策略**，学习型模型 Greedy 未提供，保留 null。', '',
      'Oracle 的约 0.025/0.032 Raw 空间是标签可见条件下的几何上界，不是单帧模型可实现提升，更不是官方 Score。真实部署 Oracle 和相对最佳常数的空间仍为 null。精确八动作只消耗 CPU，不预留 3 小时 GPU。', '',
      '## 八个问题的当前答案', '',
      '1. **A/B 最优常数？** 真实部署尚未知；历史 Raw pilot 为 A=0.975、B=0.948719，B 仅为 LIVE 转置。',
      '2. **常数收缩的真实部署增益？** 尚未测得；Raw pilot 为 A +0.006689、B +0.007653 IoU，不能替代部署增益。',
      '3. **人工尺度有多少可由内容解释？** 受限 GNMC 探针有约 0.15 源外 R²，控制几何后约 +0.11；总解释比例、LIVE 比例和标注者风格噪声比例无法识别。',
      '4. **P1 样本内偏差多严重？** 未知，缺真实训练源推理；不能从训练标签推断。',
      f"5. **位置误差改变最优 Scale 吗？** 历史 Raw 反事实中 A {e2['historical_raw_pilot']['A']['position_vs_scale']['optimal_action_changed_at_GT_center_source_macro_rate']:.1%}、B {e2['historical_raw_pilot']['B']['position_vs_scale']['optimal_action_changed_at_GT_center_source_macro_rate']:.1%} 的 source-macro 最优动作改变（平值统一优先较大 Scale），支持存在耦合；最终部署需复核。",
      '6. **Oracle@K 相对最优常数还有多少空间？** Raw Oracle@8 约 +0.024803/+0.032103；最终部署空间未知，且上界不等于可学习信息。',
      '7. **值得启动 Scale-SFT 吗？** 当前不启动。先补最终 P1 中心，确认最佳常数、原生 B、中心偏差和源外内容策略相对最佳常数的部署收益；GNMC 弱正信号支持继续小成本诊断，不能据此批准 SFT。',
      '8. **GAICD MOS 是否有独立价值？** 有，MOS 测量审美排序，与单框 IoU 监督不同；但当前固定中心、精确画幅 Scale 可用覆盖很小，不能直接推导 DPO 有益。', '',
      'GAICD 只打开 Journal Train 标注：A/B 有效目标比例候选 54/2,082，至少两个候选的图像 6/192；同图同中心（≤1 px）、不同尺度的候选对仅 3/107，尚未按 P1 中心和八动作过滤，也未筛选 MOS 差值。这些是几何支持计数，不是 DPO 偏好数据量，未生成偏好文件。', '',
      '## 后续输入、复现与停止点', '',
      '运行命令、环境、输入/代码 SHA、Git 状态摘要见 `execution.json`；统计方案见 `protocol.json`；原始常数全表、source 划分和 source-bootstrap 区间见三个 JSON。缓存及干净 public manifest 保留本地且不入 Git。', '',
      '```bash',
      "CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 python -m scripts.run_scale_diagnostics",
      '# 取得身份绑定且覆盖完整 source/view 的最终 P1 缓存后，仅运行 CPU：',
      "CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 python -m scripts.run_scale_diagnostics --positions PATH_TO_PUBLIC_P1_POSITIONS_JSONL",
      '```', '',
      '缓存要求、已有 GPU 程序的使用边界和待批准推理预算见 `pending_inference_plan.json`。本轮不运行该程序，不调用远程 GPU，不自动开始后续阶段。']
    if final_results:
        lines+=['','## 新提供的最终 P1 缓存结果（覆盖上述未测问题的已提供域）','',
                '| 路线 | 数据源 | 最佳常数 | 确认源部署 IoU | vs Scale=1 [95% CI] |',
                '|---|---|---:|---:|---|',*final_results,'',
                '这些是冻结最终 P1 中心后、固定时间集合的空间几何结果；GNMC 是单图投影而非视频部署验证。Q1/Q2 以本表对应域为准，Q4 以 scale_predictability.json 中实际 train/val 中心误差为准，Q6 以 scale_oracle_analysis.json 的 routes/by_dataset 为准。缺失域与缺失的完整 Raw/QP 中间测量仍为 null。']
    (out/'diagnostic_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
