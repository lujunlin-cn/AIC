#!/usr/bin/env python3
"""Source-held-out test of a semantic transition residual.

The 4 fps probe showed that local state changes help only when conditioned on
the frozen InternVideo2 event anchor.  This script tests that mechanism
directly by adding fixed products between the anchor and the change channels;
there is no coefficient sweep or official-label access.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import GRADE, decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import _rank01, proxy_f, select

def rows(paths):
    out=[]
    for p in paths:
        with p.open(newline='') as f: out += list(csv.DictReader(f))
    return out

def make_x(base, sem):
    # base has 20 high-frequency features.  Channels 9:20 are local change
    # and before/after state; their product with semantic relevance is a
    # deliberately small interaction block.
    s=np.asarray(sem,dtype=np.float32)[:,None]
    return np.c_[base,s,base[:,9:20]*s]

def oof(x,y,g,c):
    p=np.zeros(len(y),float)
    for v in sorted(set(g)):
        tr=g!=v; te=g==v; sx=StandardScaler().fit(x[tr])
        A=np.c_[np.ones(tr.sum()),c[tr]]; coef=np.linalg.lstsq(A,y[tr],rcond=None)[0]
        m=Ridge(alpha=10.).fit(sx.transform(x[tr]),y[tr]-A@coef)
        p[te]=m.predict(sx.transform(x[te]))
    return p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True)
    ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--scores',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rs=rows(a.labels); by={}
    for r in rs: by.setdefault(r['video_id'],[]).append(r)
    scores=json.loads(a.scores.read_text())['videos']; cache={}; X=[];Y=[];G=[];T=[]
    for v in sorted(by):
        fr,_,_=decode_highfreq(a.videos/f'{v}.mp4'); cache[v]=highfreq_features(fr)
        for r in by[v]:
            t=min(int(float(r['time_sec'])),29); j=min(round(t*4),len(cache[v])-1)
            sem=scores[v]['event_score']; X.append(cache[v][j]); Y.append(GRADE[r['generic_highlight']]); G.append(v); T.append(t)
    X=np.asarray(X,np.float32); Y=np.asarray(Y,float); G=np.asarray(G); T=np.asarray(T,int)
    sem=np.asarray([scores[v]['event_score'][min(t,len(scores[v]['event_score'])-1)] for v,t in zip(G,T)],np.float32)
    c=np.asarray([1-abs(t-14.5)/14.5 for t in T],float)
    arms={'D4_LOCAL_HF_IV2':np.c_[X,sem], 'D5_SEMANTIC_TRANSITION':make_x(X,sem)}
    metrics={}; preds={}
    for name,x in arms.items():
        p=oof(x,Y,G,c); preds[name]=p
        folds=[]
        for v in sorted(by):
            te=G==v; folds.append({'video_id':v,'mae':float(np.mean(np.abs(p[te]-Y[te])) )})
        metrics[name]={'macro_mae':float(np.mean([z['mae'] for z in folds])),'folds':folds}
    # Fixed-set proxy, one first view per second and eta=.20.
    proxy={}
    for name,p in preds.items():
        vals=[]
        for v in sorted(by):
            ix=np.flatnonzero(G==v); times=T[ix]; u,k=np.unique(times,return_index=True); y=Y[ix][k]; rr=p[ix][k]; n=len(y); kk=int(round(.85*n)); c0=1-np.abs(np.arange(n)-(n-1)/2)/max((n-1)/2,1)
            kc=select(c0,kk); f=_rank01(c0)+.2*(_rank01(rr)-.5); kr=select(f,kk)
            vals.append({'video_id':v,'center':proxy_f(kc,y,True),'residual':proxy_f(kr,y,True),'delta':proxy_f(kr,y,True)-proxy_f(kc,y,True),'swaps':int(len(set(kc.tolist())^set(kr.tolist()))//2)})
        proxy[name]={'macro':float(np.mean([z['residual'] for z in vals])),'center_macro':float(np.mean([z['center'] for z in vals])),'positive':int(sum(z['delta']>0 for z in vals)),'source':vals}
    out={'experiment_id':'R46_SEMANTIC_TRANSITION_PROBE','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rs),'sources':len(by),'alpha':10.0,'eta':.20,'metrics':metrics,'proxy':proxy,'feature_contract':'4fps local state + frozen IV2 anchor + fixed semantic x change interactions'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'metrics':metrics,'proxy_summary':{k:{q:v[q] for q in ('macro','center_macro','positive')} for k,v in proxy.items()}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
