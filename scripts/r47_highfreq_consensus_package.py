#!/usr/bin/env python3
"""Build the fixed cross-scale-consensus temporal candidate on full B3 rows.

The public-label fit is frozen once over the 12 annotated source videos.  At
deployment, a frame receives the 4 fps residual only when the 1 fps and 4 fps
readouts agree in rank sign.  This script changes frame membership only and
passes through the audited full-frame B3 boxes.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import GRADE, _rank01, select


def read_rows(paths):
    out=[]
    for p in paths:
        with Path(p).open(newline="") as f: out += list(csv.DictReader(f))
    return out


def multiscale(video):
    frames, fps, n = decode_highfreq(video)
    h3 = highfreq_features(frames).astype(np.float32)
    gray = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0 for x in frames])
    h2=[]
    for j in range(0,len(frames),4):
        b=max(0,j-4); a=min(len(frames)-1,j+4)
        prev=float(np.mean(np.abs(gray[j]-gray[b]))); nxt=float(np.mean(np.abs(gray[a]-gray[j])))
        h2.append(np.r_[h3[j,:9],prev,nxt,abs(prev-nxt)])
    return h2 and np.asarray(h2,np.float32), h3, fps, n


def fit(X,Y,C):
    sc=StandardScaler().fit(X); A=np.c_[np.ones(len(C)),C]; coef=np.linalg.lstsq(A,Y,rcond=None)[0]
    return sc, Ridge(alpha=10.).fit(sc.transform(X),Y-A@coef)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--public-videos',type=Path,required=True); ap.add_argument('--b3-allframes',type=Path,required=True); ap.add_argument('--parent-center',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--eta',type=float,default=.20); a=ap.parse_args()
    labels=read_rows(a.labels); vids=sorted({r['video_id'] for r in labels}); pub={}
    X2=[]; X3=[]; Y=[]; C=[]
    for v in vids:
        h2,h3,_,_=multiscale(a.public_videos/f'{v}.mp4'); pub[v]=(h2,h3)
        for r in labels:
            if r['video_id']!=v: continue
            t=min(int(float(r['time_sec'])),len(h2)-1); X2.append(h2[t]); X3.append(h3[min(4*t,len(h3)-1)]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(t-14.5)/14.5)
    sc2,m2=fit(np.asarray(X2,np.float32),np.asarray(Y,float),np.asarray(C,float)); sc3,m3=fit(np.asarray(X3,np.float32),np.asarray(Y,float),np.asarray(C,float))
    full=[json.loads(x) for x in a.b3_allframes.read_text().splitlines() if x.strip()]
    parent=[json.loads(x) for x in a.parent_center.read_text().splitlines() if x.strip()]
    pm={str(x['video_id']):x for x in parent}; out=[]; stats=[]
    for row in sorted(full,key=lambda x:int(str(x['video_id']))):
        v=str(row['video_id']); h2,h3,fps,n=multiscale(a.videos/f'{v}.mp4');
        if len(row['predictions'])!=n: raise RuntimeError(f'{v}: full B3 {len(row["predictions"])} != decode {n}')
        ids=np.arange(n,dtype=np.int64); t3=np.arange(len(h3),dtype=float)/4.; tf=ids/max(fps,1e-6); t2=np.arange(len(h2),dtype=float); p3=m3.predict(sc3.transform(h3)); p2=m2.predict(sc2.transform(h2)); r3=np.interp(tf,t3,p3); r2=np.interp(tf,t2,p2)
        c=1-np.abs(tf/max(tf[-1],1e-6)-.5)*2; rc=_rank01(c); a2=_rank01(r2); a3=_rank01(r3); agree=((a2-.5)*(a3-.5)>=0).astype(float); fused=rc+a.eta*agree*(a3-.5); k=int(round(.85*n)); chosen=ids[select(fused,k)]
        by={int(p['frame']):p for p in row['predictions']}; q=dict(row); q['predictions']=[by[int(i)] for i in sorted(chosen)]; q['model_size_mb']=450.283377; q['model_params_b']=.450283377; out.append(q)
        base=pm[v]; bset={int(p['frame']) for p in base['predictions']}; cset=set(map(int,chosen)); stats.append({'video_id':v,'n_full':n,'n_keep':k,'agreement_fraction':float(np.mean(agree)),'changed_vs_center':int(len(bset^cset)//2),'changed_vs_center_overlap_boxes':int(sum(by[i].get('bboxes')=={int(p['frame']):p for p in base['predictions']}.get(i,{}).get('bboxes') for i in bset&cset))})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for q in out:f.write(json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n')
    a.output.with_suffix('.summary.json').write_text(json.dumps({'experiment_id':'R47_HIGHFREQ_CROSS_SCALE_CONSENSUS_B3','parent':'R26_POS_CENTER_EXACT085_B3','videos':len(out),'predictions':sum(len(q['predictions']) for q in out),'eta':a.eta,'ridge_alpha':10.,'feature_contract':'1fps local state and 4fps local state; signed D3 residual only under same-side rank consensus','boxes':'full-frame B3 copied','parameter_count':450283377,'stats':stats},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'videos':len(out),'predictions':sum(len(q['predictions']) for q in out),'mean_changed_vs_center':float(np.mean([s['changed_vs_center'] for s in stats]))},ensure_ascii=False))

if __name__=='__main__': main()
