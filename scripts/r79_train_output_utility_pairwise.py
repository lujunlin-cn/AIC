#!/usr/bin/env python3
"""Pairwise ranking readout for the frozen V9 deployed-candidate IoU target."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

class UtilityHead(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def rank(a): return np.argsort(np.argsort(a,kind='mergesort'),kind='mergesort')
def macro(v,y,src,nm,mask):
 vals=[]
 for s in sorted(set(src[mask])):
  q=mask&(src==s); ids=np.where(q)[0]; k=max(1,int(round(.85*len(ids)))); ii=ids[np.argsort(-v[q],kind='mergesort')[:k]]; vals.append(float(y[ii].mean()))
 return float(np.mean(vals))
def rho(a,b):
 ra=rank(a);rb=rank(b);return float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--norm',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--seed',type=int,default=20261019);ap.add_argument('--device',default='cuda');a=ap.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 z=np.load(a.input,allow_pickle=True);x=z['x'].astype(np.float32);y=z['y'].astype(np.float32);base=z['base'].astype(np.float32);src=z['src'].astype(str);nm=z['names'].astype(str); ck=torch.load(a.norm,map_location='cpu',weights_only=False);x=(x-ck['mu'])/ck['sd']; xt=torch.from_numpy(x);yt=torch.from_numpy(y)
 tr=np.char.endswith(nm,'train'); rv=nm=='rv_dev'; live=nm=='live_dev'; groups={s:np.where(tr&(src==s))[0] for s in sorted(set(src[tr]))}; sources=list(groups)
 m=UtilityHead(x.shape[1]).to(a.device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-3);rng=np.random.default_rng(a.seed);best=(-1,None);hist=[]
 for st in range(1,a.steps+1):
  ss=rng.choice(sources,size=min(a.batch,len(sources)),replace=False) if len(sources)>=a.batch else rng.choice(sources,size=a.batch,replace=True); ids=[]; pair_ids=[]
  for s in ss:
   g=groups[s]; ii=rng.choice(g,size=2,replace=len(g)<2); ids.extend(ii.tolist()); pair_ids.append((len(ids)-2,len(ids)-1))
  ids=np.asarray(ids,dtype=np.int64); p=m(xt[ids].to(a.device)); losses=[]
  for i,j in pair_ids:
   yi,yj=float(y[ids[i]]),float(y[ids[j]])
   if yi==yj: continue
   sign=1. if yi>yj else -1.; gap=min(abs(yi-yj),1.)
   losses.append(F.softplus(-sign*(p[i]-p[j]))*(.5+gap))
  if not losses: continue
  loss=torch.stack(losses).mean()+.1*F.smooth_l1_loss(p,yt[ids].to(a.device),beta=.1);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
  if st%100==0 or st==a.steps:
   with torch.no_grad(): pred=m(xt.to(a.device)).cpu().numpy()
   rr={'step':st,'loss':float(loss),'rv':macro(pred,y,src,nm,rv),'live':macro(pred,y,src,nm,live),'rv_rho':rho(pred[rv],y[rv]),'live_rho':rho(pred[live],y[live])};hist.append(rr);print(json.dumps(rr),flush=True)
   if rr['rv']>best[0]:best=(rr['rv'],{k:v.detach().cpu().clone() for k,v in m.state_dict().items()})
 if best[1] is not None:m.load_state_dict(best[1])
 with torch.no_grad():pred=m(xt.to(a.device)).cpu().numpy()
 rep={'protocol':'source-balanced pairwise ranking on frozen V9 selected-candidate native IoU','steps':a.steps,'batch':a.batch,'params':sum(p.numel() for p in m.parameters()),'best_rv':best[0],'rv':{'direct':macro(pred,y,src,nm,rv),'v9_score':macro(base,y,src,nm,rv),'rho_direct':rho(pred[rv],y[rv]),'rho_v9':rho(base[rv],y[rv])},'live':{'direct':macro(pred,y,src,nm,live),'v9_score':macro(base,y,src,nm,live),'rho_direct':rho(pred[live],y[live]),'rho_v9':rho(base[live],y[live])},'history':hist}
 a.out.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'mu':ck['mu'],'sd':ck['sd'],'config':{'input_dim':x.shape[1],'target':'V9 selected candidate native IoU'}},a.out/'pairwise_utility_head.pt');(a.out/'metrics.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2),flush=True)
if __name__=='__main__':main()
