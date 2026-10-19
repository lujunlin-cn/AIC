#!/usr/bin/env python3
"""R96: static vs short-context observability on the R42 public human labels.

Source-group leave-one-out logistic probes. This is a mechanism diagnostic only:
R42 is a 12-source, 1fps, 30s human protocol and is not AIC GT.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss

VALID_DEP={'当前帧足够','需要前后过程','需要声音或语言','无法判断'}

def make_features(e):
    # e [31,D], labels use t=0..29; anchor 30 is future context for t=29.
    cur=e[:30]; prev=np.vstack([e[0:1],e[:29]]); nxt=e[1:31]
    return {'static':cur, 'context':np.concatenate([cur,cur-prev,nxt-cur,(prev+nxt)/2-cur],1), 'difference':np.concatenate([cur-prev,nxt-cur],1)}

def metrics(y,p):
    out={'n':len(y),'positive':int(y.sum()),'ap':float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)),'f1':float(f1_score(y,p>=.5,zero_division=0))}
    try: out['logloss']=float(log_loss(y,np.clip(p,1e-5,1-1e-5),labels=[0,1]))
    except Exception: out['logloss']=None
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs='+',type=Path,required=True); ap.add_argument('--embeddings',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=[]
    for p in a.labels:
        d=pd.read_csv(p); d['label_file']=str(p); rows.append(d)
    d=pd.concat(rows,ignore_index=True); d['source']=d['video_id'].astype(str)
    d['y'] = d['generic_highlight'].astype(str).str.extract(r'([0-3])')[0].astype(int).ge(2).astype(int)
    d['needs_context_valid']=d['needs_context'].astype(str).isin(VALID_DEP)
    if len(d)!=len(d['source'].unique())*30 and False: raise ValueError('unexpected row count')
    records=[]; Xs={k:[] for k in ('static','context','difference')}; ys=[]; groups=[]; deps=[]; phases=[]
    for vid,g in d.groupby('source',sort=True):
        ep=a.embeddings/f'{vid}.npz'
        if not ep.exists(): raise FileNotFoundError(str(ep))
        z=np.load(ep); e=z['embeddings'].astype(np.float32); anchors=np.asarray(z['anchors'])
        if len(e)<31: raise ValueError(f'{vid}: need 31 anchors')
        # Rows are 0..29 seconds; use nearest anchor, preserving exact 1fps contract.
        gg=g.sort_values('time_sec'); idx=np.clip(np.rint(gg['time_sec'].to_numpy()).astype(int),0,30)
        f=make_features(e)
        for k in Xs: Xs[k].append(f[k][idx])
        ys.extend(gg['y'].to_numpy().tolist()); groups.extend([vid]*len(gg)); deps.extend(gg['needs_context'].astype(str).tolist()); phases.extend(gg['event_phase'].astype(str).tolist())
    y=np.asarray(ys,np.int64); groups=np.asarray(groups); deps=np.asarray(deps); phases=np.asarray(phases)
    X={k:np.concatenate(v,0) for k,v in Xs.items()}; results={}
    for name,x in X.items():
        fold=[]; pred_all=np.zeros(len(y),float)
        for held in sorted(set(groups)):
            tr=groups!=held; te=~tr
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=20261019))
            model.fit(x[tr],y[tr]); p=model.predict_proba(x[te])[:,1]; pred_all[te]=p
            m=metrics(y[te],p); m['held_source']=held; fold.append(m)
        results[name]={'source_macro':{k:float(np.mean([f[k] for f in fold if f[k] is not None])) for k in ('ap','auc','balanced_accuracy','f1','logloss')},'folds':fold}
        # Dependency subgroup is descriptive, only positives with a valid protocol choice.
        sub={}
        for cat in ('当前帧足够','需要前后过程','需要声音或语言','无法判断'):
            m=(y==1)&(deps==cat)
            if m.sum()>=3: sub[cat]={'n':int(m.sum()),'mean_score':float(pred_all[m].mean()),'mean_rank':float(((pred_all.argsort().argsort()[m]+1).mean())/len(pred_all))}
        results[name]['positive_dependency_probe']=sub
    paired={}
    for metric in ('ap','auc','f1'):
        a0={z['held_source']:z[metric] for z in results['static']['folds'] if z[metric] is not None}
        a1={z['held_source']:z[metric] for z in results['context']['folds'] if z[metric] is not None}
        ids=sorted(set(a0)&set(a1)); dd=np.asarray([a1[i]-a0[i] for i in ids],float); rng=np.random.default_rng(20261019)
        bs=np.asarray([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(10000)]) if len(dd) else np.zeros(0)
        paired[metric]={'sources':len(dd),'mean':float(dd.mean()) if len(dd) else None,'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))] if len(dd) else None,'per_source':{i:float(v) for i,v in zip(ids,dd)}}
    out={'experiment_id':'R96_EVENT_CONTEXT_OBSERVABILITY','official_test_labels_used':False,'protocol':'R42 V1.3 public human diagnostic, source-group leave-one-source-out, fixed C=.1 logistic probes','rows':len(y),'sources':len(set(groups)),'positive':int(y.sum()),'dependency_values':{k:int((deps==k).sum()) for k in sorted(set(deps))},'phase_values':{k:int((phases==k).sum()) for k in sorted(set(phases))},'embedding_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.embeddings.glob('*.npz'))},'results':results,'paired_static_context':paired,'decision':{'context_gate_delta_ap':0.03,'context_over_static_ap':results['context']['source_macro']['ap']-results['static']['source_macro']['ap'],'context_over_static_auc':results['context']['source_macro']['auc']-results['static']['source_macro']['auc'],'use_for_official_package':False,'note':'diagnostic only; labels are not AIC GT'}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'rows':out['rows'],'sources':out['sources'],'positive':out['positive'],'summary':{k:v['source_macro'] for k,v in results.items()},'decision':out['decision']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
