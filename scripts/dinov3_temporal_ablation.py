#!/usr/bin/env python3
"""Cheap T-0/T-1 DINOv3 temporal ablation on the completed PHD2 cache.

Modes are deliberately small and matched: ``dino`` uses CLS features,
``dino_pos`` appends normalized within-fragment time, ``delta`` appends first
differences, and ``pos`` is a position-only negative control.  Source split,
fixed 85% decoder and union/soft supervision are held constant.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def fixed85(scores, labels, srcs):
    by = defaultdict(list)
    for i, s in enumerate(srcs):
        n = len(scores[i]); k=max(1,int(round(.85*n))); ix=np.argsort(-scores[i])[:k]
        pred=np.zeros(n,bool); pred[ix]=1; y=labels[i] > .5
        by[str(s)].append(float(2*(pred&y).sum()/max(1,pred.sum()+y.sum())))
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=3000); ap.add_argument('--seed',type=int,default=20261020); ap.add_argument('--device',default='cuda')
    a=ap.parse_args(); root=Path(a.cache); rows=[json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x.strip()]
    X=[]; Yu=[]; Ys=[]; T=[]; src=[]
    for r in rows:
        z=np.load(r['path']); x=z['pooled'].astype(np.float32); ts=z['timestamps'].astype(np.float32);
        t=(ts-ts.min())/max(1e-6,float(ts.max()-ts.min()));
        X.append(x); Yu.append(z['union'].astype(np.float32)); Ys.append(z['soft'].astype(np.float32)); T.append(t); src.append(r['src'])
    X=np.stack(X); Yu=np.stack(Yu); Ys=np.stack(Ys); T=np.stack(T); src=np.asarray(src)
    uniq=np.asarray(sorted(set(src))); rng=np.random.default_rng(a.seed); rng.shuffle(uniq); val=set(uniq[:max(1,int(round(.06*len(uniq))))]); va=np.asarray([s in val for s in src]); tr=~va
    import torch
    torch.manual_seed(a.seed)
    class TCN(torch.nn.Module):
        def __init__(self,d):
            super().__init__(); self.proj=torch.nn.Conv1d(d,128,1); self.convs=torch.nn.ModuleList([torch.nn.Conv1d(128,128,3,padding=q,dilation=q) for q in (1,2,4,8)]); self.out=torch.nn.Conv1d(128,1,1)
        def forward(self,x):
            h=self.proj(x.transpose(1,2))
            for c in self.convs: h=torch.nn.functional.gelu(c(h)+h)
            return self.out(h).squeeze(1)
    results={'protocol':'same cache/source split/decoder; DINO CLS plus explicit temporal controls','rows':len(rows),'val_sources':len(val),'results':[]}
    for mode in ('pos','dino','dino_pos','delta'):
        if mode=='pos': A=T[...,None]
        elif mode=='dino': A=X
        elif mode=='dino_pos': A=np.concatenate([X,T[...,None]],-1)
        else:
            D=np.concatenate([np.zeros_like(X[:,:1]),np.diff(X,axis=1)],1); A=np.concatenate([X,D,T[...,None]],-1)
        xt=torch.from_numpy(A[tr]).float(); yt=torch.from_numpy(Yu[tr]).float(); xv=torch.from_numpy(A[va]).float().to(a.device)
        m=TCN(A.shape[-1]).to(a.device).float(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01); rr=np.random.default_rng(a.seed); bs=min(64,len(xt))
        for step in range(a.steps):
            ix=rr.integers(0,len(xt),bs); p=m(xt[ix].to(a.device)); loss=torch.nn.functional.binary_cross_entropy_with_logits(p,yt[ix].to(a.device)); opt.zero_grad(); loss.backward(); opt.step()
        with torch.inference_mode(): pred=torch.sigmoid(m(xv)).cpu().numpy()
        f,n=fixed85(pred,Yu[va],src[va]); results['results'].append({'mode':mode,'input_dim':A.shape[-1],'params':sum(p.numel() for p in m.parameters()),'source_macro_f':f,'sources':n,'last_loss':float(loss.detach().cpu()),'steps':a.steps})
        print(json.dumps(results['results'][-1]),flush=True)
    Path(a.out).write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
