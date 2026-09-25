"""Source-paired head-capacity analysis; report all preregistered models/seeds."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.analyze_nested_cv import paired_bootstrap,stats


def collect(root, model):
    observations=defaultdict(list); fold_rows=[]
    for p in sorted(Path(root).glob('*/*_outer/metrics.json')):
        d=json.loads(p.read_text())
        if d['job']['model']!=model:continue
        fold_rows.append(d)
        for row in d['per_video']:observations[row['video_id']].append(row)
    if len(fold_rows)!=30 or len(observations)!=50 or any(len(v)!=6 for v in observations.values()):
        raise ValueError('Expected 30 outer-fold runs and six observations per source')
    keys=['f1','spearman','ndcg_at_15pct','summary_f1','prediction_rate','empty_prediction']
    videos={vid:{k:stats([r[k] for r in rows])['mean'] for k in keys} for vid,rows in observations.items()}
    metrics={k:stats([float(np.mean([r[k] for r in d['per_video'] if r[k] is not None])) for d in fold_rows]) for k in keys}
    return videos,metrics,fold_rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--control',required=True);ap.add_argument('--linear',required=True)
    ap.add_argument('--ood',required=True);ap.add_argument('--ood-control',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();out={'models':{},'official_f_video':None,'competition_score':None}
    ood_control=json.loads(Path(a.ood_control).read_text())
    for model in ['A0','DeiT_S']:
        control,control_stats,control_runs=collect(a.control,model)
        linear,linear_stats,runs=collect(a.linear,model)
        # Match all split/seed identities, then check all actual source arrays.
        def identity(d):return tuple(d['job'][k] for k in ['repeat','fold_id','seed'])
        c={identity(d):d for d in control_runs};l={identity(d):d for d in runs}
        if c.keys()!=l.keys():raise ValueError('unpaired fold/seed')
        checks=0
        for key,d in l.items():
            cr=c[key]
            cfg=json.loads((Path(a.linear)/d['job']['run_id'].removesuffix('_outer')/'config.json').read_text())
            ccfg=json.loads((Path(a.control)/cr['job']['run_id'].removesuffix('_outer')/'config.json').read_text())
            for field in ['fold','epochs','learning_rate','weight_decay','thresholds','seed']:
                if cfg[field]!=ccfg[field]:raise ValueError('changed control variable '+field)
            for row in d['per_video']:
                vid=row['video_id'];lp=Path(a.linear)/d['job']['run_id'].removesuffix('_outer')/d['job']['run_id']/(vid+'.npz')
                cp=Path(a.control)/cr['job']['run_id'].removesuffix('_outer')/cr['job']['run_id']/(vid+'.npz')
                with np.load(lp) as lz,np.load(cp) as cz:
                    for field in ['labels','mask','frame_indices','timestamps']:
                        if not np.array_equal(lz[field],cz[field]):raise ValueError('prediction inputs mismatch')
                checks+=1
        ids=sorted(control)
        paired={k:paired_bootstrap([control[v][k] for v in ids],[linear[v][k] for v in ids]) for k in control[ids[0]]}
        od=json.loads((Path(a.ood)/model/'metrics.json').read_text())
        okeys=['native_mean_user_f1','native_max_user_f1','spearman','ndcg_at_15pct']
        ov={vid:{k:float(np.mean([v[k] for run in od['runs'] for v in run['per_video'] if v['video_id']==vid and v[k] is not None])) for k in okeys} for vid in sorted({v['video_id'] for r in od['runs'] for v in r['per_video']})}
        oc=ood_control['per_video'][model]
        if ov.keys()!=oc.keys():raise ValueError('unpaired OOD source coverage')
        op={k:paired_bootstrap([oc[v][k] for v in sorted(ov)],[ov[v][k] for v in sorted(ov)]) for k in okeys}
        out['models'][model]={'control_fold_statistics':control_stats,'linear_fold_statistics':linear_stats,
            'linear_minus_unet':paired,'linear_ood_means':{k:float(np.mean([v[k] for v in ov.values()])) for k in okeys},
            'linear_minus_unet_ood':op,'per_video':linear,'ood_per_video':ov,'paired_input_checks':checks,
            'training_total_seconds':sum(d['job']['training_seconds'] for d in runs),
            'max_training_seconds':max(d['job']['training_seconds'] for d in runs),
            'head_parameter_count':runs[0]['job']['head_parameter_count'],'head_fp16_bytes':runs[0]['job']['head_weight_bytes']}
    out['protocol']='LINEAR_HEAD_CAPACITY_CONTROL_V1'
    out['interpretation']='Same nested protocol; only head changed. SumMe is already exposed exploratory confirmation, not fresh test. No full inference bundle or promotion claimed.'
    Path(a.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for model,d in out['models'].items():print(model,json.dumps({k:d[k] for k in ['linear_fold_statistics','linear_minus_unet','linear_ood_means','linear_minus_unet_ood']}))


if __name__=='__main__':main()
