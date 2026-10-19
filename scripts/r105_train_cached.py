#!/usr/bin/env python3
"""Train/evaluate R105 heads from the cached AViMoS target features."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def macro(pred, y, src, mask):
    vals=[]
    for s in sorted(set(src[mask])):
        q=mask&(src==s); vals.append(float(y[q,np.argmax(pred[q],axis=1)].mean()))
    return float(np.mean(vals)) if vals else float('nan')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--seed',type=int,default=20261019); a=ap.parse_args()
    import torch, torch.nn as nn
    z=np.load(a.features,allow_pickle=True); x=z['x'].astype(np.float32); y=z['y'].astype(np.float32); extra=z['stats'].astype(np.float32); src=z['source'].astype(str); names=z['name'].astype(str); train=np.char.endswith(names,'train'); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); device=torch.device(a.device)
    def run(use_extra, permuted=False):
      rng=np.random.default_rng(a.seed); e=np.zeros_like(extra) if not use_extra else extra.copy()
      if permuted: e=e[rng.permutation(len(e))]
      xx=np.concatenate([x,e],axis=2); X=torch.from_numpy(xx); Y=torch.from_numpy(y); ids=np.flatnonzero(train)
      class Head(nn.Module):
       def __init__(self):
        super().__init__(); self.p=nn.Sequential(nn.LayerNorm(xx.shape[-1]),nn.Linear(xx.shape[-1],256),nn.GELU(),nn.Linear(256,128)); self.q=nn.MultiheadAttention(128,4,batch_first=True); self.o=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
       def forward(self,z): h=self.p(z); a,_=self.q(h,h,h); return self.o(h+a).squeeze(-1)
      torch.manual_seed(a.seed); net=Head().to(device); opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3); best=(-1,None)
      for st in range(1,a.steps+1):
        take=rng.choice(ids,min(64,len(ids)),False); p=net(X[take].to(device)); loss=((p-Y[take].to(device))**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st%100==0 or st==a.steps:
          net.eval(); pp=[]
          with torch.inference_mode():
            for j in range(0,len(X),256): pp.append(net(X[j:j+256].to(device)).cpu().numpy())
          pp=np.concatenate(pp); sc=macro(pp,y,src,~train); print(json.dumps({'variant':('avimos' if use_extra else 'control'),'step':st,'loss':float(loss.detach().cpu()),'dev_source_macro':sc}),flush=True)
          if sc>best[0]: best=(sc,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
          net.train()
      net.load_state_dict(best[1]); net.eval(); pp=[]
      with torch.inference_mode():
       for j in range(0,len(X),256): pp.append(net(X[j:j+256].to(device)).cpu().numpy())
      pp=np.concatenate(pp); return {'dev_source_macro':macro(pp,y,src,~train),'train_source_macro':macro(pp,y,src,train),'best_dev':best[0],'params':sum(p.numel() for p in net.parameters())}
    center=[]
    for s in sorted(set(src[~train])):
      q=(~train)&(src==s); center.append(float(y[q,y.shape[1]//2].mean()))
    report={'experiment_id':'R105_AVIMOS_AIC_UTILITY','rows':int(len(y)),'dev_sources':int(len(set(src[~train]))),'center_dev_source_macro':float(np.mean(center)),'control':run(False),'avimos':run(True),'avimos_permuted':run(True,True),'official_test_labels_used':False,'audio':'excluded'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
