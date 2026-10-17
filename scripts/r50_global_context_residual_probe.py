#!/usr/bin/env python3
"""R50 probe: add full-clip state context to the 4-fps local residual.

D3 observes only a short neighbourhood around each frame.  This arm asks
whether a frame's position in the event trajectory (change accumulated before
and after it, and displacement from the clip's start/end/median state) adds
information that local residuals cannot see.  It is a single fixed feature
contract with source-held-out Ridge; no eta/alpha/model sweep is performed.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import GRADE, _rank01, proxy_f, select

def rows(paths):
    out=[]
    for p in paths:
        with Path(p).open(newline="") as f: out += list(csv.DictReader(f))
    return out

def global_features(frames, local):
    g=np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32)/255. for x in frames])
    n=len(g)
    d=np.mean(np.abs(g[1:]-g[:-1]),axis=(1,2)) if n>1 else np.zeros(0,np.float32)
    cum=np.r_[0.,np.cumsum(d)]
    total=float(cum[-1]) if len(cum) else 0.
    med=np.median(g,axis=0)
    first,last=g[0],g[-1]
    z=[]
    for i in range(n):
        before=float(cum[i]/max(total,1e-6)); after=float((total-cum[i])/max(total,1e-6))
        z.append([float(np.mean(np.abs(g[i]-first))),
                  float(np.mean(np.abs(g[i]-last))),
                  float(np.mean(np.abs(g[i]-med))),
                  before,after,before-after,
                  float(i/max(n-1,1)),
                  float(np.mean(d[max(0,i-8):i+1])) if len(d) else 0.,
                  float(np.mean(d[i:min(len(d),i+8)])) if len(d) else 0.])
    return np.asarray(z,np.float32)

def fit_predict(xtr,ytr,ctr,xte):
    sc=StandardScaler().fit(xtr)
    A=np.c_[np.ones(len(ctr)),ctr]
    coef=np.linalg.lstsq(A,ytr,rcond=None)[0]
    m=Ridge(alpha=10.).fit(sc.transform(xtr),ytr-A@coef)
    return m.predict(sc.transform(xte))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--eta',type=float,default=.20); a=ap.parse_args()
    rs=rows(a.labels); by={}; data={}
    for r in rs: by.setdefault(r['video_id'],[]).append(r)
    for v in sorted(by):
        fr,_,_=decode_highfreq(a.videos/f'{v}.mp4'); h=highfreq_features(fr).astype(np.float32); data[v]=np.c_[h,global_features(fr,h)]
    source=[]
    for held in sorted(by):
        train=[r for r in rs if r['video_id']!=held]; X=[];Y=[];C=[]
        for r in train:
            v=r['video_id']; t=min(int(float(r['time_sec'])),29); X.append(data[v][min(4*t,len(data[v])-1)]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(t-14.5)/14.5)
        X=np.asarray(X,np.float32); Y=np.asarray(Y,float); C=np.asarray(C,float)
        # Aggregate overlapping annotators at the held-out source/time.
        grouped={}
        for r in by[held]: grouped.setdefault(int(float(r['time_sec'])),[]).append(r)
        ts=np.asarray(sorted(grouped),int); y=np.asarray([np.mean([GRADE[r['generic_highlight']] for r in grouped[t]]) for t in ts],float)
        xte=data[held][np.minimum(4*ts,len(data[held])-1)]; p=fit_predict(X,Y,C,xte)
        c=1-np.abs(ts-(len(ts)-1)/2)/max((len(ts)-1)/2,1)
        k=int(round(.85*len(ts)))
        kc=select(c,k); kr=select(_rank01(c)+a.eta*(_rank01(p)-.5),k)
        source.append({'video_id':held,'n':len(ts),'center':proxy_f(kc,y,True),'global':proxy_f(kr,y,True),'delta':proxy_f(kr,y,True)-proxy_f(kc,y,True),'center_binary':proxy_f(kc,y,False),'global_binary':proxy_f(kr,y,False),'delta_binary':proxy_f(kr,y,False)-proxy_f(kc,y,False),'swaps':int(len(set(kc.tolist())^set(kr.tolist()))//2)})
    out={'experiment_id':'R50_GLOBAL_CONTEXT_RESIDUAL_PROBE','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rs),'sources':len(source),'eta':a.eta,'alpha':10.,'feature_contract':'4fps D3 local state + clip trajectory: distance to start/end/median, cumulative change before/after, normalized change position, 2-second motion density','aggregation':'overlap annotations averaged per held-out source/time','official_gt':False,'macro_center':float(np.mean([x['center'] for x in source])),'macro_global':float(np.mean([x['global'] for x in source])),'macro_delta':float(np.mean([x['delta'] for x in source])),'positive_sources':int(sum(x['delta']>0 for x in source)),'macro_center_binary':float(np.mean([x['center_binary'] for x in source])),'macro_global_binary':float(np.mean([x['global_binary'] for x in source])),'positive_sources_binary':int(sum(x['delta_binary']>0 for x in source)),'source':source}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ('experiment_id','macro_center','macro_global','macro_delta','positive_sources','macro_center_binary','macro_global_binary','positive_sources_binary')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
