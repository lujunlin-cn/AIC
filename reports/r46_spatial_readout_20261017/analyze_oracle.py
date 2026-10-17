#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def ci(x, n=20000, seed=20261017):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    means=np.empty(n)
    for i in range(n): means[i]=rng.choice(x,len(x),replace=True).mean()
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    d=pd.read_csv(a.csv)
    # Same source appears twice under seed views; do not pool seeds as independent.
    d['source']=d['vid'].astype(str)
    pool_rows=[]
    # primary source independent views: live_confirmation B3_s0 and live_dev B3_s0 and rv_diag_exposed.
    for pool, x in d.groupby(['pool','arm'], sort=True):
        p,arm=pool
        # source macro over rows; a video contributes equally regardless frame count
        per=x.groupby('source')[['head_iou','center_iou','oracle_iou']].mean()
        for col in ['head_iou','center_iou','oracle_iou']:
            pass
        def delta(a1,a2):
            z=per[a1]-per[a2]
            return {'mean':float(z.mean()),'ci95':ci(z),'wins':int((z>0).sum()),'losses':int((z<0).sum()),'n_sources':int(len(z))}
        pool_rows.append({'pool':p,'arm':arm,'rows':int(len(x)),'sources':int(len(per)),
           'mean_source_iou':{k:float(per[k].mean()) for k in ['head_iou','center_iou','oracle_iou']},
           'head_minus_center':delta('head_iou','center_iou'),
           'oracle_minus_head':delta('oracle_iou','head_iou'),
           'oracle_minus_center':delta('oracle_iou','center_iou'),
           'gap_positive_frame_share':float((x.gap>1e-4).mean()),
           'gap_frame_mean':float(x.gap.mean())})
    # Paired seed consistency on same live_confirmation sources.
    x0=d[(d.pool=='live_confirmation')&(d.arm=='B3_s0')].groupby('source')[['head_iou']].mean()
    x1=d[(d.pool=='live_confirmation_s1')&(d.arm=='B3_s1')].groupby('source')[['head_iou']].mean()
    seed_delta=(x1.head_iou-x0.head_iou).dropna()
    out={'protocol':'R46 spatial candidate-utility oracle audit',
      'input_sha256':None,
      'claim_boundary':'Public source-level diagnostic only; oracle uses public crop GT and cannot be converted to official score.',
      'rows':int(len(d)), 'pool_results':pool_rows,
      'seed_consistency':{'n_sources':int(len(seed_delta)),'mean_s1_minus_s0':float(seed_delta.mean()),'ci95':ci(seed_delta),'corr':float(np.corrcoef(x0.head_iou.loc[seed_delta.index],x1.head_iou.loc[seed_delta.index])[0,1])},
      'interpretation':{
       'supports':'Candidate geometry is generally adequate on these public pools and current B3 candidate ranking leaves substantial attainable utility on the table; a readout/representation/label transfer intervention is more justified than changing candidate count.',
       'does_not_identify':['whether the error is frozen visual features, pooling, head optimization, or domain/label policy','whether a new readout transfers to the 426 competition videos','whether replacing boxes changes joint official F'],
       'next_minimal_experiment':'Train a constrained readout on the same 129 candidate features with source-disjoint public splits, retain B3 candidate geometry, and require source-level gain over frozen B3 and center on both live and RV held-out pools before packaging.'}}
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
