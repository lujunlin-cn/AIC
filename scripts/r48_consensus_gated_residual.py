#!/usr/bin/env python3
"""Build R48: fixed consensus-gated 4fps temporal residual candidate.

The low-frequency (1fps) and high-frequency (4fps) readouts are fit on the
same public labels.  A high-frequency rank adjustment is allowed only when
both readouts are on the same side of their per-video median.  This is a
single predeclared mechanism, not an eta/threshold sweep.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.r45_highfreq_semantic_residual import decode_highfreq, highfreq_features
from scripts.r44_local_state_residual import _rank01, select

def read_rows(paths):
    out=[]
    for p in paths:
        with p.open(newline='') as f: out += list(csv.DictReader(f))
    return out
def fit(x,y):
    s=StandardScaler().fit(x); m=Ridge(alpha=10.).fit(s.transform(x),y); return s,m
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--public-video-root',type=Path,required=True); ap.add_argument('--video-root',type=Path,required=True); ap.add_argument('--parent',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=None); ap.add_argument('--eta',type=float,default=.20); args=ap.parse_args()
    labels=read_rows(args.labels); grade={'0普通':0.,'1有信息':1.,'2值得保留':2.,'3高光核心':3.}; X2=[]; X3=[]; Y=[]; cache={}
    for v in sorted({r['video_id'] for r in labels}):
        fr,_,_=decode_highfreq(args.public_video_root/f'{v}.mp4'); cache[v]=highfreq_features(fr)
    for r in labels:
        v=r['video_id']; h=cache[v]; t=min(int(float(r['time_sec'])),29); z=h[min(round(t*4),len(h)-1)]
        X2.append(np.r_[z[:9],z[11],z[12],abs(z[15])]); X3.append(z); Y.append(grade[r['generic_highlight']])
    s2,m2=fit(np.asarray(X2,np.float32),np.asarray(Y,float)); s3,m3=fit(np.asarray(X3,np.float32),np.asarray(Y,float))
    rows=[json.loads(x) for x in args.parent.read_text().splitlines() if x.strip()]; rows=sorted(rows,key=lambda r:int(str(r['video_id'])) if str(r['video_id']).isdigit() else str(r['video_id']))
    end=args.end if args.end is not None else len(rows); rows=rows[args.start:end]; out=[]; changed=0; swaps=0
    for r in rows:
        v=str(r['video_id']); fr,fps,n=decode_highfreq(args.video_root/f'{v}.mp4'); h=highfreq_features(fr)
        p2=m2.predict(s2.transform(np.c_[h[:,:9],h[:,11],h[:,12],np.abs(h[:,15])]))
        p3=m3.predict(s3.transform(h)); anchors=np.arange(len(h),dtype=float)/4.; ft=np.arange(n,dtype=float)/max(fps,1e-9); ai=np.minimum(np.rint(ft*4).astype(int),len(h)-1); q2=p2[ai]; q3=p3[ai]
        r2=_rank01(q2); r3=_rank01(q3); same=((r2-.5)*(r3-.5)>=0).astype(float); center=1.-np.abs(ft/max(ft[-1],1e-9)-.5)*2.; fused=_rank01(center)+args.eta*(r3-.5)*same
        ids=np.asarray(sorted(int(p['frame']) for p in r['predictions']),int); k=int(round(.85*n)); chosen=ids[select(fused[ids],k)]; old={int(p['frame']):p for p in r['predictions']}; q=dict(r); q['predictions']=[old[int(i)] for i in sorted(chosen)]; out.append(q); parent_keep=np.asarray(sorted(ids[select(1.-np.abs(np.arange(n,dtype=float)/max(n-1,1)-.5)*2, k)]),int); changed += int(set(chosen)!=set(parent_keep)); swaps += len(set(chosen)-set(parent_keep))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(''.join(json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n' for q in out)); args.output.with_suffix('.summary.json').write_text(json.dumps({'experiment_id':'R48_CONSENSUS_GATED_RESIDUAL_B3','parent':'R26_POS_CENTER_EXACT085_B3','videos':len(out),'predictions':sum(len(q['predictions']) for q in out),'changed_videos':changed,'swaps_added':swaps,'eta':args.eta,'alpha':10.,'feature_contract':'1fps local + 4fps local state; highfreq residual active only when readout ranks agree','boxes':'copied from parent'},indent=2)+'\n'); print(json.dumps({'videos':len(out),'predictions':sum(len(q['predictions']) for q in out),'changed_videos':changed,'swaps_added':swaps}))
if __name__=='__main__': main()
