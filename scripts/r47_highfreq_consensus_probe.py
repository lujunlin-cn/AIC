#!/usr/bin/env python3
"""Source-held-out cross-scale consistency probe for the time residual.

The 1 fps local descriptor and the 4 fps descriptor observe the same content
at different temporal scales.  A fixed consensus gate keeps the D3 residual
only when both independent readouts rank the frame on the same side of their
within-video median.  This is a mechanism test for suppressing sub-second
compression/cut noise; eta=.20 and Ridge alpha=10 remain fixed and there is
no gate/weight sweep.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys
import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import GRADE, frame_descriptor, _rank01, proxy_f, select


def read_rows(paths):
    out=[]
    for p in paths:
        with Path(p).open(newline="") as f: out += list(csv.DictReader(f))
    return out


def multiscale(video):
    frames, _, _ = decode_highfreq(video)
    h = highfreq_features(frames)
    gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for x in frames]
    # One-second descriptors at exact 4 fps anchors: static appearance plus
    # state changes over the preceding/following second.
    one=[]
    for j in range(0, len(frames), 4):
        b=max(0,j-4); a=min(len(frames)-1,j+4)
        prev=float(np.mean(np.abs(gray[j]-gray[b])))
        nxt=float(np.mean(np.abs(gray[a]-gray[j])))
        one.append(np.r_[h[j,:9], prev, nxt, abs(prev-nxt)])
    return h.astype(np.float32), np.asarray(one,np.float32)


def fit_predict(Xtr,Ytr,Ctr,Xte):
    sc=StandardScaler().fit(Xtr)
    A=np.c_[np.ones(len(Ctr)),Ctr]
    coef=np.linalg.lstsq(A,Ytr,rcond=None)[0]
    m=Ridge(alpha=10.).fit(sc.transform(Xtr),Ytr-A@coef)
    return m.predict(sc.transform(Xte))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--eta',type=float,default=.20); a=ap.parse_args()
    labels=read_rows(a.labels); vids=sorted({r['video_id'] for r in labels}); data={}
    for v in vids: data[v]=multiscale(a.videos/f'{v}.mp4')
    by={}
    for r in labels: by.setdefault(r['video_id'],[]).append(r)
    src=[]
    for v in vids:
        tr=[r for r in labels if r['video_id']!=v]
        X2=[]; X3=[]; Y=[]; C=[]
        for r in tr:
            u=r['video_id']; t=min(int(float(r['time_sec'])),len(data[u][1])-1); X3.append(data[u][0][min(4*t,len(data[u][0])-1)]); X2.append(data[u][1][t]); Y.append(GRADE[r['generic_highlight']]); C.append(1-abs(t-14.5)/14.5)
        Y=np.asarray(Y,float); C=np.asarray(C,float)
        p2te=[]; p3te=[]
        # Fit separately to preserve distinct scale contracts.
        h2=data[v][1]; h3=data[v][0]
        times, idx=np.unique(np.asarray([int(float(r['time_sec'])) for r in by[v]]),return_index=True)
        y=np.asarray([GRADE[by[v][i]['generic_highlight']] for i in idx],float); n=len(y); k=int(round(.85*n))
        # Refit per fold, then evaluate this held-out source.
        x2tr=np.asarray(X2,np.float32); x3tr=np.asarray(X3,np.float32)
        p2=fit_predict(x2tr,Y,C,h2[times])
        p3=fit_predict(x3tr,Y,C,h3[np.minimum(4*times,len(h3)-1)])
        c=1-np.abs(np.arange(n)-(n-1)/2)/max((n-1)/2,1); rc=_rank01(c); r2=_rank01(p2); r3=_rank01(p3)
        agree=((r2-.5)*(r3-.5)>=0).astype(float)
        full=rc+a.eta*(r3-.5); cons=rc+a.eta*agree*(r3-.5)
        kc,kf,kg=select(rc,k),select(full,k),select(cons,k)
        src.append({'video_id':v,'n':n,'agree_count':int(agree.sum()),'center_soft':proxy_f(kc,y,True),'full_d3_soft':proxy_f(kf,y,True),'consensus_soft':proxy_f(kg,y,True),'delta_full_soft':proxy_f(kf,y,True)-proxy_f(kc,y,True),'delta_consensus_soft':proxy_f(kg,y,True)-proxy_f(kc,y,True),'center_binary':proxy_f(kc,y,False),'full_d3_binary':proxy_f(kf,y,False),'consensus_binary':proxy_f(kg,y,False),'delta_full_binary':proxy_f(kf,y,False)-proxy_f(kc,y,False),'delta_consensus_binary':proxy_f(kg,y,False)-proxy_f(kc,y,False),'full_swaps':int(len(set(kc.tolist())^set(kf.tolist()))//2),'consensus_swaps':int(len(set(kc.tolist())^set(kg.tolist()))//2)})
    out={'experiment_id':'R47_HIGHFREQ_CROSS_SCALE_CONSENSUS','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(labels),'sources':len(vids),'eta':a.eta,'ridge_alpha':10.,'gate':'signed D3 residual enabled iff rank(D2 local residual) and rank(D3 4fps residual) are on same side of within-video median','official_gt':False,'source':src}
    for k in ['center_soft','full_d3_soft','consensus_soft','center_binary','full_d3_binary','consensus_binary']:
        out['macro_'+k]=float(np.mean([z[k] for z in src]))
    for k in ['delta_full_soft','delta_consensus_soft','delta_full_binary','delta_consensus_binary']:
        out['positive_'+k]=int(sum(z[k]>0 for z in src))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ['experiment_id','macro_center_soft','macro_full_d3_soft','macro_consensus_soft','macro_center_binary','macro_full_d3_binary','macro_consensus_binary','positive_delta_full_soft','positive_delta_consensus_soft']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
