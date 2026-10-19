#!/usr/bin/env python3
"""R48 public probe: trust 4-fps residual only under a low-frequency consensus.

This is a mechanism test, not a parameter sweep.  D3 improves fixed-set proxy
ranking while its absolute MAE is worse than the 1-fps local model.  The
predeclared intervention therefore keeps the centre rank and applies the D3
residual only when the 1-fps and 4-fps readouts point to the same side of their
within-video medians.  An onset-only arm is reported as a competing mechanism.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
GRADE={'0普通':0.0,'1有信息':1.0,'2值得保留':2.0,'3高光核心':3.0}
def _rank01(x):
    return np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort').astype(float)/max(len(x)-1,1)
def select(x,k): return np.argsort(-x,kind='mergesort')[:max(1,min(int(k),len(x)))]
def proxy_f(ix,y,soft=True):
    z=y/3 if soft else (y>=2).astype(float); s=float(z[ix].sum()); m=float(z.sum()); return 2*s/(len(ix)+m) if len(ix)+m else 1.
def frame_descriptor(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(np.float32)
    return np.asarray([gray.mean()/255,gray.std()/255,cv2.Laplacian(gray,cv2.CV_32F).var()/10000,cv2.Canny(im,80,160).mean()/255,im[:,:,0].mean()/255,im[:,:,1].mean()/255,im[:,:,2].mean()/255,hsv[:,:,1].mean()/255,hsv[:,:,2].std()/255],np.float32)
def decode_highfreq(video):
    cap=cv2.VideoCapture(str(video)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 30); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur=max((n-1)/fps,0); m=max(1,int(np.floor(dur*4+1e-7))+1); target=np.minimum(np.rint(np.arange(m)*fps/4).astype(np.int64),n-1); wanted={}
    for j,i in enumerate(target.tolist()): wanted.setdefault(int(i),[]).append(j)
    ims=[None]*m; i=0
    while i<=int(target[-1]):
        ok,im=cap.read()
        if not ok: break
        if i in wanted:
            for j in wanted[i]: ims[j]=im.copy()
        i+=1
    cap.release(); fb=next((x for x in ims if x is not None),np.zeros((224,224,3),np.uint8)); return np.stack([x if x is not None else fb.copy() for x in ims])
def highfreq_features(frames):
    desc=np.stack([frame_descriptor(x) for x in frames]).astype(np.float32); gray=np.stack([cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32)/255 for x in frames]); diff=np.mean(np.abs(gray[1:]-gray[:-1]),axis=(1,2)); out=[]
    for c in range(len(frames)):
        lo,hi=max(0,c-2),min(len(frames),c+3); local=diff[max(0,lo-1):min(len(diff),hi)]; before=diff[max(0,c-2):c]; after=diff[c:min(len(diff),c+2)]; bm=float(before.mean()) if before.size else 0.; am=float(after.mean()) if after.size else 0.; bx=float(before.max()) if before.size else 0.; ax=float(after.max()) if after.size else 0.; im=desc[c]; ib=desc[max(0,c-2)]; ia=desc[min(len(desc)-1,c+2)]
        out.append(np.r_[im,float(local.max()) if local.size else 0.,float(local.mean()) if local.size else 0.,bm,am,bx,ax,am-bm,ax-bx,float(np.abs(im-ib).mean()),float(np.abs(ia-im).mean()),float(np.abs(ia-im).mean())-float(np.abs(im-ib).mean())])
    return np.asarray(out,np.float32)
def read_rows(paths):
    out=[]
    for stream,p in enumerate(paths):
        with p.open(newline='') as f:
            for r in csv.DictReader(f):
                q=dict(r); q['_stream']=stream; out.append(q)
    return out

def fit_oof(x2,x3,y,g):
    p2=np.full(len(y),np.nan); p3=np.full(len(y),np.nan)
    for v in sorted(set(g)):
        tr,te=g!=v,g==v
        def pred(a,b):
            mu=a.mean(0); sd=a.std(0); sd[sd<1e-6]=1; z=(a-mu)/sd; zt=(b-mu)/sd; A=np.c_[np.ones(len(z)),z]; coef=np.linalg.solve(A.T@A+10*np.eye(A.shape[1]),A.T@y[tr]); return np.c_[np.ones(len(zt)),zt]@coef
        p2[te]=pred(x2[tr],x2[te]); p3[te]=pred(x3[tr],x3[te])
    return p2,p3

def select_score(c,p2,p3,feat,eta=.20):
    r2=_rank01(p2); r3=_rank01(p3); delta=r3-.5
    same=((r2-.5)*(r3-.5)>=0).astype(float)
    # A fixed physical gate: the 4-fps readout is trusted on rising local
    # motion only.  It is a competing hypothesis, not tuned after scoring.
    onset=(feat[:,12] >= feat[:,11]).astype(float)
    return {
      'D2':_rank01(c)+eta*(r2-.5),
      'D3':_rank01(c)+eta*delta,
      'CONSENSUS':_rank01(c)+eta*delta*same,
      'ONSET':_rank01(c)+eta*delta*onset,
      'CONSENSUS_ONSET':_rank01(c)+eta*delta*same*onset,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',nargs=2,type=Path,required=True); ap.add_argument('--videos',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=read_rows(args.labels); by={}
    for r in rows: by.setdefault(r['video_id'],[]).append(r)
    X2=[]; X3=[]; Y=[]; G=[]; S=[]; T=[]; F={}
    for v in sorted(by):
        fr=decode_highfreq(args.videos/f'{v}.mp4'); h=highfreq_features(fr); F[v]=h
        for r in by[v]:
            t=min(int(float(r['time_sec'])),29); z=h[min(round(t*4),len(h)-1)]
            X2.append(np.r_[z[:9],z[11],z[12],abs(z[15])]); X3.append(z)
            Y.append(GRADE[r['generic_highlight']]); G.append(v); S.append(int(r.get('_stream',0))); T.append(t)
    X2=np.asarray(X2,np.float32); X3=np.asarray(X3,np.float32); Y=np.asarray(Y,float); G=np.asarray(G); S=np.asarray(S); T=np.asarray(T)
    p2,p3=fit_oof(X2,X3,Y,G)
    source={}; all_d={k:[] for k in ('D2','D3','CONSENSUS','ONSET','CONSENSUS_ONSET')}
    for v in sorted(by):
        ixv=np.flatnonzero(G==v)
        for st in sorted(set(S[ixv].tolist())):
            ix=ixv[S[ixv]==st]; times=T[ix]; times,keep=np.unique(times,return_index=True); ix=ix[keep]
            y=Y[ix]; n=len(ix); k=int(round(.85*n)); c=1-np.abs(np.arange(n)-(n-1)/2)/max((n-1)/2,1)
            scores=select_score(c,p2[ix],p3[ix],X3[ix]); base=proxy_f(select(c,k),y,True)
            rec={'video_id':v,'stream':st,'base':base}
            for name,sc in scores.items():
                val=proxy_f(select(sc,k),y,True); rec[name]=val; rec[name+'_delta']=val-base; all_d[name].append(val-base)
            source[f'{v}|{st}']=rec
    out={'experiment_id':'R48_CONSENSUS_GATED_RESIDUAL_PROBE','status':'PUBLIC_SOURCE_HELD_OUT_COMPLETED','rows':len(rows),'sources':len(by),'views':len(source),'eta':.20,'alpha':10.,'feature_contract':'D2 1fps local state; D3 4fps local state; source-held-out Ridge','hypothesis':'high-frequency residual is useful only when its direction agrees with 1fps local state','macro_delta_soft':{k:float(np.mean(v)) for k,v in all_d.items()},'positive_views':{k:int(sum(x>0 for x in v)) for k,v in all_d.items()},'source':source}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({k:out[k] for k in ('experiment_id','views','macro_delta_soft','positive_views')},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
