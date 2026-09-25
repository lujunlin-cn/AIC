#!/usr/bin/env python3
"""Paired source-video inference for repeated nested CV, not independent frames."""
import argparse,csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS=['f1','precision','recall','spearman','kendall_tau_b','ndcg','ndcg_at_15pct',
         'summary_f1','prediction_rate','target_rate','empty_prediction','continuous_mae']

def stats(values):
    v=np.asarray([x for x in values if x is not None],float)
    return {'mean':float(v.mean()),'std':float(v.std(ddof=1)) if len(v)>1 else 0.,
            'median':float(np.median(v)),'n':len(v)} if len(v) else None

def paired_bootstrap(a,b,seed=20260925,nboot=20000):
    """One independent sampling unit per source video, after seed/repeat means."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a):raise ValueError('paired vectors required')
    delta=b-a;rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),(nboot,len(delta)))].mean(axis=1)
    return {'delta':float(delta.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'n_videos':len(delta),'positive_video_fraction':float((delta>0).mean()),
            'bootstrap_replicates':nboot,'seed':seed}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    models=defaultdict(list);sources=[];long=[];vids=defaultdict(lambda:defaultdict(list))
    for p in sorted(root.glob('*/*_outer/metrics.json')):
        d=json.loads(p.read_text());job=d['job'];model=job['model'];models[model].append(d)
        sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        for row in d['per_video']:
            rec={**row,**{k:job[k] for k in ['model','seed','repeat','fold_id','threshold']},'run_id':job['run_id']}
            long.append(rec);vids[model][row['video_id']].append(rec)
    if set(models)!=set(['A0','DeiT_S','ViT_B']):raise ValueError('missing representation')
    aggregated={}
    for model,by_video in vids.items():
        if len(by_video)!=50 or any(len(x)!=6 for x in by_video.values()):raise ValueError('expected 50 videos x 2 repeats x 3 seeds')
        aggregated[model]={}
        for vid,rows in by_video.items():
            d={k:rows[0][k] for k in ['video_id','category','title','nframes']}
            d.update({k:stats([r[k] for r in rows])['mean'] if stats([r[k] for r in rows]) else None for k in METRICS})
            d['score_quantiles']={q:float(np.mean([r['score_quantiles'][q] for r in rows])) for q in rows[0]['score_quantiles']}
            d['threshold_mean']=float(np.mean([r['threshold'] for r in rows]));aggregated[model][vid]=d
    paired={}
    for baseline,challenger in [('A0','DeiT_S'),('A0','ViT_B'),('DeiT_S','ViT_B')]:
        pair=challenger+'-minus-'+baseline;paired[pair]={}
        for key in METRICS:
            common=[v for v in sorted(aggregated[baseline]) if all(aggregated[m][v][key] is not None for m in [baseline,challenger])]
            paired[pair][key]=paired_bootstrap([aggregated[baseline][v][key] for v in common],[aggregated[challenger][v][key] for v in common])
    summary={m:{k:stats([float(np.mean([r[k] for r in d['per_video'] if r[k] is not None])) for d in runs]) for k in METRICS} for m,runs in models.items()}
    categories={m:{c:{k:stats([r[k] for r in rows.values() if r['category']==c]) for k in METRICS} for c in sorted({r['category'] for r in rows.values()})} for m,rows in aggregated.items()}
    deltas=[]
    for vid,x in aggregated['A0'].items():
        b=aggregated['DeiT_S'][vid];d={k:x[k] for k in ['video_id','category','title','nframes']}
        for k in METRICS:
            d['A0_'+k]=x[k];d['DeiT_S_'+k]=b[k];d['delta_'+k]=None if x[k] is None or b[k] is None else b[k]-x[k]
        deltas.append(d)
    report={'protocol':'TVSUM_PAIRED_NESTED_CV_ANALYSIS_V1','run_count':len(sources),
        'unit':'50 source videos; average six repeated/seed observations per video before bootstrap',
        'limitations':'Development-exposed TVSum; bootstrap conditional on trained folds, not full training-population uncertainty. Metrics exploratory, no multiplicity adjustment.',
        'fold_seed_statistics':summary,'paired_video_bootstrap':paired,'categories':categories,
        'per_video':aggregated,'fold_seed_rows':[{'job':d['job'],'statistics':d['statistics']} for rs in models.values() for d in rs],
        'sources':sources,'official_f_video':None,'competition_score':None}
    (out/'nested_cv_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'nested_cv_per_observation.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in long))
    with (out/'a0_deit_per_video.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(sorted(deltas,key=lambda r:r['delta_summary_f1'],reverse=True))
    for m in summary:print(m,{k:summary[m][k] for k in ['f1','spearman','ndcg_at_15pct','summary_f1','empty_prediction']})
    for pair in paired:print(pair,{k:paired[pair][k] for k in ['f1','spearman','ndcg_at_15pct','summary_f1']})
if __name__=='__main__':main()
