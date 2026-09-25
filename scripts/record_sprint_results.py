#!/usr/bin/env python3
"""Append completed evidence to registry without deleting historical failures."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='artifacts');ap.add_argument('--registry',default='experiments/registry.jsonl');a=ap.parse_args()
    root=Path(a.root);path=Path(a.registry);old=[json.loads(x) for x in path.read_text().splitlines()];seen={x['run_id'] for x in old};new=[]
    base={'date':'2026-09-25','host':'supie','official_f_video':None,'competition_score':None}
    for p in sorted((root/'RG_NCV_001').glob('*/*_outer/metrics.json')):
        d=json.loads(p.read_text());job=d['job'];run=p.parent.parent.name;cfg=json.loads((p.parent.parent/'config.json').read_text());model=job['model']
        metrics={k:v['mean'] if v else None for k,v in d['statistics'].items()}
        new.append({**base,'run_id':run,'dataset':'TVSum','dataset_version':'proxy-v2 + TVSUM_SUMMARY_V1_FIXED + TVSUM_RANKING_V2',
            'architecture':model+' frozen encoder + shared TemporalUNet','config':str(p.parent.parent/'config.json'),
            'config_sha256':sha(p.parent.parent/'config.json'),'git_commit':cfg['git_commit'],'gpu_ids':{'A0':[2],'DeiT_S':[4],'ViT_B':[5]}[model],
            'seed':job['seed'],'split':cfg['fold'],'batch_size':1,'optimizer':'AdamW','learning_rate':cfg['learning_rate'],
            'scheduler':None,'epochs':cfg['epochs'],'checkpoint_rule':'minimum inner-dev continuous BCE','threshold':job['threshold'],
            'threshold_source':'inner_dev only','training_time_seconds':job['training_seconds'],'best_checkpoint':job['checkpoint'],
            'checkpoint_sha256':d['checkpoint_sha256'],'manifest_sha256':d['manifest_sha256'],'metrics':metrics,
            'head_fp16_bytes':job['head_weight_bytes'],'full_bundle_bytes':None,
            'bundle_note':'head experiment; full deployment bundles separately audited; head alone is NOT total inference size',
            'environment':f'/data/aic/experiments/RG_NCV_001/{run}/environment.txt',
            'command':f'/data/aic/experiments/RG_NCV_001/{run}/command.txt',
            'result':'completed_paired_nested_cv','decision':'aggregate source-paired evidence; no individual outer-fold winner selection'})
    for group in ['SUMME_OOD_002','SUMME_OOD_003','SUMME_CV_OOD_001','SUMME_CV_OOD_002']:
        for p in sorted((root/group).glob('*/metrics.json')):
            d=json.loads(p.read_text());m=d['model'];cv='runs' in d
            native_keys=['native_mean_user_f1','native_max_user_f1','spearman','kendall_tau_b','ndcg_at_15pct']
            if cv:
                import numpy as np
                metrics={k:float(np.mean([v[k] for run in d['runs'] for v in run['per_video'] if v[k] is not None])) for k in native_keys}
            else:metrics=d['means']
            new.append({**base,'run_id':group+'_'+m,'dataset':'SumMe','config':d['config'],'git_commit':d['config']['git_commit'],
                'source_metrics_sha256':sha(p),'gpu_ids':{'A0':[2],'DeiT_S':[4],'ViT_B':[5]}[m],
                'training_time_seconds':0,'result':'completed_zero_shot_ood','no_ood_tuning':True,
                'strict_video_count':len(d['runs'][0]['per_video']) if cv else d['strict_video_count'],
                'heads':len(d['runs']) if cv else 1,'weight_bytes':d.get('weights'),'parameter_count':d.get('parameter_count'),
                'latency_seconds':d.get('latency_total_seconds'),'peak_vram_bytes':d.get('peak_vram_bytes'),
                'metrics':metrics,'validator':d.get('validator'),'decode_protocol':'ANNOTATED_CFR_DECODE_ORDINAL_V1',
                'caveat':'size-biased public pilot, exact decoded-count/FPS alignment only; cannot infer full dataset performance',
                'artifact_root':'/data/aic/experiments/'+group+'/'+m})
    for group in ['ENGINEERING_RELEASE_002','ENGINEERING_RELEASE_003']:
        for p in sorted((root/group).glob('*.report.json')):
            d=json.loads(p.read_text())
            new.append({**base,'run_id':group+'_'+d['candidate_id'],'dataset':'DHF1K subset3' if group.endswith('002') else 'TVSum historical DEV subset2',
                'result':'completed_frozen_raw_video_release_verification','git_commit':'bc6ca2d','gpu_ids':[6],
                'training_time_seconds':0,'config':d['candidate'],'source_metrics_sha256':sha(p),
                'inventory':d['inventory'],'weight_bytes':d['loaded_weight_bytes'],
                'latency_seconds':d['elapsed_seconds'],'peak_vram_bytes':d['peak_vram_bytes'],
                'metrics':{'empty_prediction_rate':d['empty_prediction_rate'],'per_video':d['per_video']},
                'validator':d['validation'],'artifact_root':'/data/aic/experiments/'+group,
                'decision':'integration evidence only; no temporal GT in DHF1K; no threshold changes'})
    new.append({**base,'run_id':'ENGINEERING_RELEASE_001','result':'failed_cuda_statistics_before_device_initialization',
                'training_time_seconds':0,'git_commit':'bc6ca2d (fix included)','gpu_ids':[6],
                'decision':'retained failure log; explicitly initialize selected logical CUDA device; rerun as 002'})
    p=Path('reports/representation_generalization/cross_dataset_duplicate_audit.json')
    if p.exists():
        d=json.loads(p.read_text())
        new.append({**base,'run_id':'CROSS_DATASET_DUPLICATE_001','result':'completed_sparse_duplicate_screen',
                    'training_time_seconds':0,'gpu_ids':[],'source_metrics_sha256':sha(p),
                    'metrics':{k:d[k] for k in ['n_tvsum','n_summe','n_pairs','exact_duplicates','review_candidates']},
                    'decision':'no flagged pair; sparse screen is not proof of zero near-duplicate or pretraining overlap'})
    p=root/'SUMME_BASELINES_001'/'metrics.json'
    if p.exists():
        d=json.loads(p.read_text())
        new.append({**base,'run_id':'SUMME_BASELINES_001','result':'completed_fixed_budget_nonlearning_controls',
                    'dataset':'SumMe strict-alignment 14 videos','config':d['config'],'training_time_seconds':0,
                    'gpu_ids':[],'seed':d['config']['random_seed'],'weight_bytes':0,
                    'source_metrics_sha256':sha(p),'metrics':d['means'],'runtime_seconds':d['elapsed_seconds'],
                    'decision':'A0 positive paired margin over random; current DeiT/ViT margins uncertain'})
    for group in ['SPATIAL_GT_001','SPATIAL_GT_NATIVE_002','DENSE_E2E_001']:
        p=root/group/'metrics.json'
        if p.exists():
            d=json.loads(p.read_text());new.append({**base,'run_id':group,'dataset':'DHF1K/RetargetVid','training_time_seconds':0,
                'result':'completed_dense_crop_benchmark' if group.startswith('SPATIAL') else 'completed_raw_jsonl_spatial_equivalence',
                'metrics_source':str(p),'source_metrics_sha256':sha(p),'artifact_root':'/data/aic/experiments/'+group,
                'metrics':d.get('methods'),'protocol':d['protocol'],'detector_bytes':d.get('detector_bytes'),
                'git_commit':'1ba1b9a' if group=='SPATIAL_GT_001' else 'd088c31',
                'decision':'no stable IoU gain claim; source-video bootstrap CI crosses zero'})
    new.append({**base,'run_id':'SUMME_OOD_001','result':'failed_source_nonmonotonic_pts','dataset':'SumMe',
                'training_time_seconds':0,'git_commit':'1ba1b9a','decision':'preserve strict competition PTS; separate explicit audited annotation-ordinal benchmark in SUMME_OOD_002'})
    unique=[x for x in new if x['run_id'] not in seen]
    with path.open('a') as f:
        for x in unique:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    print('appended',len(unique),'total',len(old)+len(unique))
if __name__=='__main__':main()
