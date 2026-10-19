#!/usr/bin/env python3
"""Source held-out test of the predeclared 4 fps boundary intervention.

This is a diagnostic only.  It evaluates the same 30 second public clips used
for the annotation calibration, with source-held-out predictions.  The
deployment rule is fixed to the R26 geometry (keep=.85, 5 percent bands) and
eta=.20; no official labels or package score are read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

# Allow direct execution from the repository root (``python scripts/...``).
# The project keeps scripts as a source directory rather than a packaged
# module, so relying on the current interpreter's module path is brittle.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.r45_highfreq_semantic_residual import (
    GRADE, decode_highfreq, highfreq_features,
)
from scripts.r44_local_state_residual import _rank01, proxy_f, select


def rows(paths):
    out=[]
    for p in paths:
        with p.open(newline="") as f:
            out += list(csv.DictReader(f))
    return out


def boundary_set(n, score, eta=.20, keep=.85, band_fraction=.05):
    k=int(round(keep*n)); lo=(n-k)//2; hi=lo+k
    band=max(1,int(round(band_fraction*n)))
    protected=np.arange(min(hi,lo+band),max(min(hi,lo+band),hi-band),dtype=int)
    allowed=np.arange(max(0,lo-band),min(n,hi+band),dtype=int)
    free=np.setdiff1d(allowed,protected,assume_unique=True)
    need=k-len(protected)
    center=1-np.abs(np.arange(n)-(n-1)/2)/max((n-1)/2,1)
    fused=_rank01(center)+eta*(_rank01(score)-.5)
    return np.sort(np.r_[protected,free[select(fused[free],need)]])


def fit_oof(x,y,g,c):
    pred=np.zeros(len(y),float)
    for v in sorted(set(g)):
        tr=g!=v; te=g==v
        sx=StandardScaler().fit(x[tr])
        A=np.c_[np.ones(tr.sum()),c[tr]]
        coef=np.linalg.lstsq(A,y[tr],rcond=None)[0]
        model=Ridge(alpha=10.).fit(sx.transform(x[tr]),y[tr]-A@coef)
        pred[te]=model.predict(sx.transform(x[te]))
    return pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True)
    ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--scores',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rs=rows(args.labels); by={}
    for r in rs: by.setdefault(r['video_id'],[]).append(r)
    scores=json.loads(args.scores.read_text())['videos']
    X=[]; Y=[]; G=[]; T=[]; tracks={}
    for v in sorted(by):
        frames,_,_=decode_highfreq(args.videos/f'{v}.mp4'); tracks[v]=highfreq_features(frames)
        for r in by[v]:
            t=min(int(float(r['time_sec'])),29); j=min(round(t*4),len(tracks[v])-1)
            sem=scores[v]['event_score']; X.append(np.r_[tracks[v][j],sem[min(t,len(sem)-1)]])
            Y.append(GRADE[r['generic_highlight']]); G.append(v); T.append(t)
    X=np.asarray(X,np.float32); Y=np.asarray(Y,float); G=np.asarray(G); T=np.asarray(T,int)
    # Public contract has one annotation per second per stream.  Retain the
    # first view for the selection proxy so duplicate streams cannot double
    # weight one source; report both stream rows in the input metadata.
    pred=fit_oof(X,Y,G,np.asarray([1-abs(t-14.5)/14.5 for t in T]))
    source=[]
    for v in sorted(by):
        ix=np.flatnonzero(G==v); times=T[ix]; uniq,keep=np.unique(times,return_index=True)
        y=Y[ix][keep]; p=pred[ix][keep]; n=len(y); k=int(round(.85*n))
        c=1-np.abs(np.arange(n)-(n-1)/2)/max((n-1)/2,1)
        sfull=_rank01(c)+.2*(_rank01(p)-.5)
        kc=select(c,k); kf=select(sfull,k); kb=boundary_set(n,p)
        # Count where each rule changes the parent and how much public proxy F
        # moves.  The boundary rule is fixed before looking at these labels.
        source.append({'video_id':v,'n':n,'center_proxy':proxy_f(kc,y,True),
          'full_hf_proxy':proxy_f(kf,y,True),'boundary_hf_proxy':proxy_f(kb,y,True),
          'delta_full':proxy_f(kf,y,True)-proxy_f(kc,y,True),
          'delta_boundary':proxy_f(kb,y,True)-proxy_f(kc,y,True),
          'full_swaps':int(len(set(kc.tolist())^set(kf.tolist()))//2),
          'boundary_swaps':int(len(set(kc.tolist())^set(kb.tolist()))//2),
          'boundary_changed_positions':sorted(set(kb.tolist())^set(kc.tolist()))})
    out={'experiment_id':'R45_HIGHFREQ_BOUNDARY_PROBE','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED',
         'rows':len(rs),'sources':len(by),'keep':.85,'eta':.20,'band_fraction':.05,
         'feature_contract':'4fps local state + frozen IV2 event score; source-held-out Ridge',
         'macro_center':float(np.mean([z['center_proxy'] for z in source])),
         'macro_full':float(np.mean([z['full_hf_proxy'] for z in source])),
         'macro_boundary':float(np.mean([z['boundary_hf_proxy'] for z in source])),
         'positive_full':int(sum(z['delta_full']>0 for z in source)),
         'positive_boundary':int(sum(z['delta_boundary']>0 for z in source)),
         'source':source}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('experiment_id','rows','sources','macro_center','macro_full','macro_boundary','positive_full','positive_boundary')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
