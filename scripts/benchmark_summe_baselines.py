"""Fixed-budget non-learning SumMe controls, no GT-derived selection budget."""
import argparse
import json
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
from aic.temporal_metrics import fixed_segments, ranking_report, summary_mask


def native_f1(mask, human):
    den = human.sum(0) + mask.sum()
    values = np.divide(2 * (human & mask[:, None]).sum(0), den,
                       out=np.zeros(human.shape[1]), where=den > 0)
    return {'native_mean_user_f1': float(values.mean()), 'native_max_user_f1': float(values.max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    rows=[]; start=time.perf_counter()
    for cohort_id, cohort in enumerate(cfg['cohorts'], 1):
        for p in sorted(Path(cohort['predictions']).glob('*.npz')):
            with np.load(p) as z:
                idx=z['frame_indices']; n=len(z['relevance'])
            gt=loadmat(Path(cohort['root'])/'GT'/(p.stem+'.mat'))
            human=gt['user_score']>0; relevance=gt['gt_score'].ravel()
            segments=fixed_segments(n); count=max(1,int(len(segments)*cfg['summary_budget']))
            positions=np.unique(np.rint(np.linspace(0,len(segments)-1,count)).astype(int))
            uniform=np.zeros(n,dtype=bool); remaining=int(n*cfg['summary_budget'])
            for pos in positions:
                lo,hi=segments[pos]
                if hi-lo<=remaining:uniform[lo:hi]=True; remaining-=hi-lo
            constant=summary_mask(np.zeros(n),budget=cfg['summary_budget'])
            for name,mask in [('constant_first_budget',constant),('uniform_segments',uniform)]:
                rows.append({'video_id':p.stem,'cohort':cohort_id,'method':name,**native_f1(mask,human),
                             'selection_rate':float(mask.mean()),'spearman':None,'ndcg_at_15pct':None})
            seed=cfg['random_seed']+int(hashlib.sha256(p.stem.encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed); draws=[]
            for _ in range(cfg['random_repeats']):
                full=np.interp(np.arange(n),idx,rng.random(len(idx)))
                mask=summary_mask(full,budget=cfg['summary_budget'])
                draws.append({**native_f1(mask,human),**ranking_report(full,relevance),'selection_rate':float(mask.mean())})
            keys=['native_mean_user_f1','native_max_user_f1','spearman','ndcg_at_15pct','selection_rate']
            rows.append({'video_id':p.stem,'cohort':cohort_id,'method':'random_sample_scores',
                         **{k:float(np.mean([x[k] for x in draws])) for k in keys},
                         'repeats':len(draws),'seed':seed,
                         'random_summary_std':float(np.std([x['native_mean_user_f1'] for x in draws],ddof=1))})
    means={}
    for method in cfg['baselines']:
        selected=[r for r in rows if r['method']==method]
        means[method]={k:float(np.mean([r[k] for r in selected if r[k] is not None]))
                       if any(r[k] is not None for r in selected) else None
                       for k in ['native_mean_user_f1','native_max_user_f1','spearman','ndcg_at_15pct','selection_rate']}
    report={'config':cfg,'per_video':rows,'means':means,'elapsed_seconds':time.perf_counter()-start,
            'note':'No GT count used to select frames. Constant native fallback uses ceil budget. Random means average 32 fixed draws, never choose best seed.',
            'official_f_video':None,'competition_score':None}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
