#!/usr/bin/env python3
"""Fixed shot-aware temporal prior probe (no learned weights).

At 4 fps, a per-video 95th-percentile grayscale-difference threshold defines
cut boundaries.  Within each resulting shot, frames receive a triangular
centre score.  The deployed score is the equal geometric mean of the global
centre rank and the shot-centre rank (implemented as the arithmetic mean in
rank space); K remains exactly round(.85*N).  This is a direct alternative to
the content residual and uses no labels to fit thresholds or weights.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq
from scripts.r44_local_state_residual import GRADE, _rank01, proxy_f, select


def read_rows(paths):
    out=[]
    for p in paths:
        with Path(p).open(newline="") as f: out += list(csv.DictReader(f))
    return out


def shot_score(frames):
    gray=np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32)/255. for x in frames])
    d=np.mean(np.abs(gray[1:]-gray[:-1]),axis=(1,2)) if len(frames)>1 else np.zeros(0)
    q=float(np.quantile(d,.95)) if len(d) else 0.
    cuts=(np.flatnonzero(d>=q)+1).tolist() if len(d) else []
    cuts=sorted(set([x for x in cuts if 0<x<len(frames)]))
    bounds=[0]+cuts+[len(frames)]
    s=np.zeros(len(frames),float)
    for a,b in zip(bounds[:-1],bounds[1:]):
        t=np.arange(a,b,dtype=float)
        s[a:b]=1.-np.abs(t-(a+b-1)/2.)/max((b-a-1)/2.,1.)
    return s,q,cuts


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    labels=read_rows(a.labels); by={}
    for r in labels: by.setdefault(r['video_id'],[]).append(r)
    src=[]
    for v in sorted(by):
        frames,_,_=decode_highfreq(a.videos/f'{v}.mp4'); ss,q,cuts=shot_score(frames); times,idx=np.unique(np.asarray([int(float(r['time_sec'])) for r in by[v]]),return_index=True); y=np.asarray([GRADE[by[v][i]['generic_highlight']] for i in idx],float); n=len(y); k=int(round(.85*n)); j=np.minimum(times*4,len(ss)-1); global_s=1.-np.abs(np.arange(n)-(n-1)/2.)/max((n-1)/2.,1.); shot_s=ss[j]; fused=.5*_rank01(global_s)+.5*_rank01(shot_s); kc=select(global_s,k); ks=select(fused,k); src.append({'video_id':v,'n':n,'k':k,'cuts_4fps':cuts,'cut_threshold':q,'center_soft':proxy_f(kc,y,True),'shot_soft':proxy_f(ks,y,True),'delta_soft':proxy_f(ks,y,True)-proxy_f(kc,y,True),'center_binary':proxy_f(kc,y,False),'shot_binary':proxy_f(ks,y,False),'delta_binary':proxy_f(ks,y,False)-proxy_f(kc,y,False),'swaps':int(len(set(kc.tolist())^set(ks.tolist()))//2)})
    out={'experiment_id':'R47_SHOT_AWARE_TEMPORAL_PRIOR','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(labels),'sources':len(src),'threshold':'per-video 95th percentile of 4fps adjacent grayscale mean absolute difference','merge':'equal arithmetic mean of global-center and per-shot-center ranks','keep':.85,'official_gt':False,'source':src}
    for k in ['center_soft','shot_soft','center_binary','shot_binary']: out['macro_'+k]=float(np.mean([z[k] for z in src]))
    out['positive_soft']=int(sum(z['delta_soft']>0 for z in src)); out['positive_binary']=int(sum(z['delta_binary']>0 for z in src)); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ['experiment_id','sources','macro_center_soft','macro_shot_soft','macro_center_binary','macro_shot_binary','positive_soft','positive_binary']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
