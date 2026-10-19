#!/usr/bin/env python3
"""R90: low-parameter temporal/motion ranking probe on TVSum.

The metric matches the competition decoder more closely than binary F1: for
each held-out video, rank frames and retain the fixed top 85%, then average the
continuous human importance.  Variants isolate static visual semantics,
parameter-free motion/quality/composition features, and their short temporal
context.  This is a proxy diagnostic; it never reads competition labels.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load_manifest(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'): rows.append(json.loads(line))
    return rows


def build(rows):
    out=[]
    for r in rows:
        with np.load(r['path'],allow_pickle=False) as z:
            x=z['features'].astype(np.float32); aux=z['aux'].astype(np.float32); y=z['labels'].astype(np.float32); mask=z['mask'].astype(bool)
        x=x[mask]; aux=aux[mask]; y=y[mask]
        if len(y)<4: continue
        d=np.diff(aux,axis=0,prepend=aux[:1]); dd=np.diff(d,axis=0,prepend=d[:1]);
        sm=(np.roll(aux,1,axis=0)+aux+np.roll(aux,-1,axis=0))/3.; sm[0]=(aux[0]+aux[1])/2.; sm[-1]=(aux[-2]+aux[-1])/2.
        td=np.concatenate([aux,d,dd,sm],axis=1)
        xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-6); prev=np.concatenate([xn[:1],xn[:-1]],axis=0); nxt=np.concatenate([xn[1:],xn[-1:]],axis=0)
        novelty=(1-(xn*prev).sum(1))[:,None]; speed=np.linalg.norm(x-prev,axis=1)[:,None]/np.sqrt(x.shape[1]); accel=np.linalg.norm(x-2*prev+np.concatenate([prev[:1],prev[:-1]],axis=0),axis=1)[:,None]/np.sqrt(x.shape[1]); sem_motion=np.concatenate([x,novelty,speed,accel],axis=1)
        out.append({'id':str(r.get('video_id',Path(r['path']).stem)),'x':x,'aux':aux,'td':td,'hybrid':np.concatenate([x,td],axis=1),'semantic_motion':sem_motion,'y':y})
    return out


def fit(items, key, alpha=10.):
    X=np.concatenate([q[key] for q in items]); y=np.concatenate([q['y'] for q in items]);
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-5]=1.; Z=(X-mu)/sd; A=np.concatenate([Z,np.ones((len(Z),1),np.float32)],1)
    # Equal source weight, but avoid materializing string labels per row.
    counts=np.concatenate([np.full(len(q['y']),len(q['y']),np.float32) for q in items]); counts=np.maximum(counts,1.)
    w=1./counts
    G=(A.T*w)@A; G[:-1,:-1]+=alpha*np.eye(G.shape[0]-1); b=(A.T*w)@y; beta=np.linalg.solve(G,b).astype(np.float32)
    return beta,mu,sd


def pred(q,key,m):
    beta,mu,sd=m; z=(q[key]-mu)/sd; return np.concatenate([z,np.ones((len(z),1),np.float32)],1)@beta


def rho(a,b):
    ra=np.argsort(np.argsort(a,kind='mergesort'),kind='mergesort'); rb=np.argsort(np.argsort(b,kind='mergesort'),kind='mergesort')
    return float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else 0.


def eval_variant(items,key,m):
    vals=[]; all_r=[]; all_y=[]; per={}
    for q in items:
        s=pred(q,key,m); k=max(1,int(round(.85*len(s)))); idx=np.argsort(-s,kind='mergesort')[:k]; v=float(q['y'][idx].mean()); vals.append(v); all_r.append(rho(s,q['y'])); all_y.append(float(q['y'].mean())); per[q['id']]=v
    return {'videos':len(items),'source_macro_top85':float(np.mean(vals)),'source_macro_random':float(np.mean(all_y)),'delta_vs_random':float(np.mean(vals)-np.mean(all_y)),'mean_spearman':float(np.mean(all_r)),'per_video':per}


def paired_ci(a,b,seed=20261019):
    ids=sorted(set(a)&set(b)); d=np.asarray([a[k]-b[k] for k in ids],np.float64); rng=np.random.default_rng(seed); means=np.asarray([d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)])
    return {'sources':len(ids),'mean_delta':float(d.mean()),'ci95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',type=Path,required=True); ap.add_argument('--dev',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    tr=build(load_manifest(a.train)); dv=build(load_manifest(a.dev));
    variants={'static':'x','bank':'aux','temporal_bank':'td','hybrid':'hybrid','semantic_motion':'semantic_motion'}; models={k:fit(tr,v) for k,v in variants.items()}; reports={k:eval_variant(dv,v,models[k]) for k,v in variants.items()}
    for k in reports: reports[k].pop('per_video')
    per={k:eval_variant(dv,v,models[k])['per_video'] for k,v in variants.items()}
    out={'protocol':'R90_TEMPORAL_STRUCTURE_MOTION_RANK','train_videos':len(tr),'dev_videos':len(dv),'variants':reports,'paired_vs_static':{k:paired_ci(per[k],per['static']) for k in variants if k!='static'}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
