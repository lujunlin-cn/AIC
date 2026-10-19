#!/usr/bin/env python3
"""R95 H2: explicit shot structure on the frozen RGB target-domain probe.

Uses the existing R92 PHD2 public weak-label cache. Shot boundaries are derived
from adjacent frozen RGB ResNet18 embeddings within each 8-frame fragment, with
no official/test labels. The control and structure arms share the exact model
and optimizer; the control receives zero shot channels. This is a mechanism
probe, not an official-score estimate.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np


def source_bootstrap(deltas, seed=20261019, n=5000):
    d=np.asarray([deltas[k] for k in sorted(deltas)],float); rng=np.random.default_rng(seed)
    b=np.asarray([d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)])
    return {'sources':len(d),'mean':float(d.mean()),'ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}

def fixed_f(scores,y,keep=.85):
    k=max(1,int(round(keep*len(y)))); idx=np.argsort(-scores,kind='mergesort')[:k]
    return float(2*y[idx].sum()/(k+y.sum()))

def eval_scores(scores,y,src,mask):
    by={}
    for i,s in enumerate(src[mask]): by.setdefault(str(s),[]).append(fixed_f(scores[mask][i],y[mask][i]))
    per={s:float(np.mean(v)) for s,v in by.items()}
    return {'rows':int(mask.sum()),'sources':len(per),'macro_f':float(np.mean(list(per.values()))),'per_source':per}

def shot_features(rgb):
    # rgb [B,T,D], feature-space scene changes, no frame labels.
    d=np.linalg.norm(rgb[:,1:]-rgb[:,:-1],axis=2)
    out=np.zeros((len(rgb),rgb.shape[1],4),np.float32)
    for b in range(len(rgb)):
        dd=d[b]; q=float(np.quantile(dd,.95)) if len(dd) else 0.; cuts=np.flatnonzero(dd>=q)+1 if len(dd) else np.zeros(0,int)
        cuts=np.unique(cuts[(cuts>0)&(cuts<rgb.shape[1])]); bounds=np.r_[0,cuts,rgb.shape[1]]
        for a,z in zip(bounds[:-1],bounds[1:]):
            t=np.arange(a,z,dtype=float); cen=(a+z-1)/2.; half=max((z-a-1)/2.,1.)
            out[b,a:z,0]=1.-np.abs(t-cen)/half
            out[b,a:z,2]=(z-a)/rgb.shape[1]
        if len(cuts):
            for t in range(rgb.shape[1]): out[b,t,1]=min(abs(t-int(c)) for c in cuts)/rgb.shape[1]
            out[b,cuts.clip(0,rgb.shape[1]-1),3]=1.
        out[b,:,0]=np.nan_to_num(out[b,:,0]); out[b,:,1]=np.nan_to_num(out[b,:,1])
    return out

class Net:
    @staticmethod
    def build(input_dim=516):
        import torch, torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.proj=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,64),nn.GELU()); self.temporal=nn.Conv1d(64,128,1)
                self.blocks=nn.ModuleList([nn.Conv1d(128,128,3,padding=d,dilation=d) for d in (1,2,4)]); self.norm=nn.GroupNorm(8,128)
                self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,1))
            def forward(self,x):
                z=self.proj(x); h=self.temporal(z.transpose(1,2))
                for c in self.blocks: h=torch.nn.functional.gelu(self.norm(c(h)+h))
                return self.head(torch.cat([h.transpose(1,2),z],dim=-1)).squeeze(-1)
        return M()
    
def train_arm(X,Y,src,split,seed,use_shot,steps,device):
        import torch
        torch.manual_seed(seed); rng=np.random.default_rng(seed); X=X.copy(); X[:,:,512:]=0 if not use_shot else X[:,:,512:]
        tr=split=='train'; ids=np.flatnonzero(tr); net=Net.build(X.shape[-1]).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=.01); best=(-1,None); hist=[]; Xt=torch.from_numpy(X); Yt=torch.from_numpy(Y)
        for step in range(1,steps+1):
            sel=ids[rng.integers(0,len(ids),64)]; xb=Xt[sel].to(device); yb=Yt[sel].to(device); pred=net(xb)
            wt=torch.where(yb.sum(1)>0,torch.ones(len(yb),device=device),torch.full((len(yb),),.15,device=device)); loss=(torch.nn.functional.binary_cross_entropy_with_logits(pred,yb,reduction='none').mean(1)*wt).mean()
            mixed=(yb.sum(1)>0)&(yb.sum(1)<yb.shape[1])
            if mixed.any():
                hi,lo=yb.argmax(1),yb.argmin(1); ar=torch.arange(len(yb),device=device)[mixed]; loss=loss+.25*torch.nn.functional.softplus(-(pred[ar,hi[mixed]]-pred[ar,lo[mixed]])).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            if step%100==0 or step==steps:
                net.eval();
                with torch.inference_mode(): sc=net(Xt.to(device)).cpu().numpy()
                m=eval_scores(sc,Y,src,~tr); hist.append({'step':step,'loss':float(loss.detach().cpu()),'dev_macro_f':m['macro_f']})
                if m['macro_f']>best[0]: best=(m['macro_f'],{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
                net.train()
        net.load_state_dict(best[1]); net.eval();
        with torch.inference_mode(): sc=net(Xt.to(device)).cpu().numpy()
        dev=eval_scores(sc,Y,src,~tr); train=eval_scores(sc,Y,src,tr)
        return net,sc,{'use_shot':use_shot,'seed':seed,'params':sum(p.numel() for p in net.parameters()),'dev':dev,'train':train,'history':hist}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--seed',type=int,default=20261019); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    import torch
    z=np.load(a.cache,allow_pickle=False); rgb=z['features'][:,:,0,:].astype(np.float32); Y=z['labels'].astype(np.float32); src=z['src'].astype(str); split=z['split'].astype(str); sf=shot_features(rgb); X=np.concatenate([rgb,sf],axis=2)
    results=[]
    for use in (False,True):
        net,sc,meta=train_arm(X,Y,src,split,a.seed,use,a.steps,a.device); results.append(meta); torch.save({'state_dict':net.state_dict(),'use_shot':use,'params':meta['params']},a.output/('shot.pt' if use else 'control.pt'))
    d0=results[0]['dev']['per_source']; d1=results[1]['dev']['per_source']; delta={s:d1[s]-d0[s] for s in sorted(set(d0)&set(d1))}
    out={'experiment_id':'R95_SHOT_STRUCTURE_RGB','hypothesis':'Explicit local shot structure adds target-domain utility beyond same-capacity frozen RGB temporal head','cache':str(a.cache),'cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'rows':len(Y),'dev_sources':len(delta),'feature_contract':'frozen ResNet18 RGB embedding + 4 feature-space shot channels (centre, distance-to-cut, shot-length, boundary flag)','results':results,'paired_dev':source_bootstrap(delta),'official_f_video':None,'official_test_labels_used':False}
    (a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'results':[{k:v for k,v in r.items() if k not in ('history','train')} for r in results],'paired_dev':out['paired_dev']},indent=2))
if __name__=='__main__': main()
