#!/usr/bin/env python3
"""Deploy the fixed 4 fps local-state residual without a semantic anchor.

This is the predeclared D3 arm from the public source-held-out probe.  It is a
mechanism comparison against the D4 semantic-anchor candidate, not a sweep.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import _rank01, select
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

GRADE={'0普通':0.0,'1有信息':1.0,'2值得保留':2.0,'3高光核心':3.0}

def rows(paths):
 out=[]
 for p in paths:
  import csv
  with p.open(newline='') as f: out += list(csv.DictReader(f))
 return out

def fit_model(x,y,c):
 sx=StandardScaler().fit(x); A=np.c_[np.ones(len(c)),c]; coef=np.linalg.lstsq(A,y,rcond=None)[0]
 m=Ridge(alpha=10.).fit(sx.transform(x),y-A@coef); return sx,m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--public-video-root',type=Path,required=True); ap.add_argument('--video-root',type=Path,required=True); ap.add_argument('--parent',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--eta',type=float,default=.20); ap.add_argument('--keep',type=float,default=.85); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=None); a=ap.parse_args()
    rs=rows(a.labels); by={}
    for r in rs: by.setdefault(r['video_id'],[]).append(r)
    X=[];Y=[];C=[]
    for v in sorted(by):
        fr,_,_=decode_highfreq(a.public_video_root/f'{v}.mp4'); h=highfreq_features(fr)
        for r in by[v]:
            t=min(int(float(r['time_sec'])),29); X.append(h[min(round(t*4),len(h)-1)]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(t-14.5)/14.5)
    sx,m=fit_model(np.asarray(X,np.float32),np.asarray(Y,float),np.asarray(C,float))
    out=[]; stats=[]
    parent_rows=[json.loads(x) for x in a.parent.read_text().splitlines() if x.strip()]
    parent_rows=sorted(parent_rows,key=lambda r:int(str(r['video_id'])))
    end=a.end if a.end is not None else len(parent_rows)
    parent_rows=parent_rows[a.start:end]
    for r in parent_rows:
        v=str(r['video_id']); fr,fps,n=decode_highfreq(a.video_root/f'{v}.mp4'); h=highfreq_features(fr); rr=m.predict(sx.transform(h)); ht=np.arange(len(h))/4.; ft=np.arange(n)/max(fps,1e-6); res=np.interp(ft,ht,rr); ids=np.asarray(sorted(int(p['frame']) for p in r['predictions']),int)
        if len(ids)!=n: raise RuntimeError(f'{v}: parent coverage {len(ids)} != {n}')
        center=1-np.abs(ft/max(ft[-1],1e-6)-.5)*2; fused=_rank01(center[ids])+a.eta*(_rank01(res[ids])-.5); k=int(round(a.keep*n)); chosen=ids[select(fused,k)]; old={int(p['frame']):p for p in r['predictions']}; q=dict(r); q['predictions']=[old[int(i)] for i in sorted(chosen)]; q['model_size_mb']=450.283377; q['model_params_b']=.450283377; out.append(q); stats.append({'video_id':v,'n':n,'k':k,'changed':int([p['frame'] for p in q['predictions']] != [p['frame'] for p in r['predictions']])})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for q in out:f.write(json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n')
    a.output.with_suffix('.summary.json').write_text(json.dumps({'experiment_id':'R46_HIGHFREQ_LOCAL_RESIDUAL_B3','parent':'R26_POS_CENTER_EXACT085_B3','videos':len(out),'predictions':sum(len(q['predictions']) for q in out),'changed_videos':sum(x['changed'] for x in stats),'eta':a.eta,'alpha':10.,'keep':a.keep,'feature_contract':'4fps local state only; source-held-out Ridge','boxes':'copied from parent'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'videos':len(out),'predictions':sum(len(q['predictions']) for q in out),'changed_videos':sum(x['changed'] for x in stats)},ensure_ascii=False))

if __name__=='__main__': main()
