#!/usr/bin/env python3
"""R97: target GIF interval phase auxiliary supervision on R92 RGB cache.

State labels are derived only from public PHD2 train.json intervals: before,
in, and after the observed selected interval. Rows with no interval have no
phase loss and remain weak/unknown for binary supervision. Source split is the
frozen R92 split. This is a public diagnostic, not AIC GT.
"""
from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np

def fixed_f(scores,y,keep=.85):
 k=max(1,int(round(keep*len(y)))); i=np.argsort(-scores,kind='mergesort')[:k]; return float(2*y[i].sum()/(k+y.sum()))
def eval_scores(s,y,src,mask):
 by={}
 for i,v in enumerate(src[mask]): by.setdefault(str(v),[]).append(fixed_f(s[mask][i],y[mask][i]))
 p={k:float(np.mean(v)) for k,v in by.items()}; return {'rows':int(mask.sum()),'sources':len(p),'macro_f':float(np.mean(list(p.values()))),'per_source':p}
def bootstrap_delta(a,b,seed=20261019,n=10000):
 ids=sorted(set(a)&set(b)); d=np.asarray([b[k]-a[k] for k in ids]); rng=np.random.default_rng(seed); bs=np.asarray([d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]); return {'sources':len(d),'mean':float(d.mean()),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]}
def load_phase(index_path,ann_path,frame_root,video_ids):
    idx={json.loads(l)['video_id']:json.loads(l) for l in index_path.read_text().splitlines() if l.strip()}
    ann=json.loads(ann_path.read_text()); out={}
    for vid in video_ids:
        r=idx[vid]; src=r['src']; users=ann.get(src,{})
        iv=[]
        for rows in users.values():
            for q in rows:
                a,b=float(q['t0']),float(q['t1'])
                if b>a: iv.append((a,b))
        iv=sorted(iv); times=float(r['t0'])+float(r['L'])*np.arange(8)/8.
        ph=np.full(8,-1,np.int64); y=np.zeros(8,np.float32)
        if iv:
            lo=min(a for a,b in iv); hi=max(b for a,b in iv)
            for j,t in enumerate(times):
                if any(a<=t<b for a,b in iv): ph[j]=1; y[j]=1
                elif t<lo: ph[j]=0
                elif t>=hi: ph[j]=2
                else: ph[j]=0 if t < (lo+hi)/2 else 2
        out[vid]=(y,ph,iv)
    return out
class Net:
    @staticmethod
    def build():
        import torch,torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.proj=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64),nn.GELU()); self.temporal=nn.Conv1d(64,128,1); self.blocks=nn.ModuleList([nn.Conv1d(128,128,3,padding=d,dilation=d) for d in (1,2,4)]); self.norm=nn.GroupNorm(8,128); self.head=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,1)); self.phase=nn.Linear(192,3)
            def forward(self,x):
                z=self.proj(x); h=self.temporal(z.transpose(1,2))
                for c in self.blocks: h=torch.nn.functional.gelu(self.norm(c(h)+h))
                q=torch.cat([h.transpose(1,2),z],-1); return self.head(q).squeeze(-1),self.phase(q)
        return M()
def train(X,Y,P,src,split,weight,seed,steps,device):
 import torch
 torch.manual_seed(seed); rng=np.random.default_rng(seed); net=Net.build().to(device); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=.01); tr=split=='train'; ids=np.flatnonzero(tr); Xt=torch.from_numpy(X); Yt=torch.from_numpy(Y); Pt=torch.from_numpy(P); best=(-1,None); hist=[]
 for st in range(1,steps+1):
  sel=ids[rng.integers(0,len(ids),64)]; xb=Xt[sel].to(device); yb=Yt[sel].to(device); pb=Pt[sel].to(device); logits,ph=net(xb)
  roww=torch.where(yb.sum(1)>0,torch.ones(len(yb),device=device),torch.full((len(yb),),.15,device=device)); loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,yb,reduction='none').mean(1)*roww).mean()
  mask=pb>=0
  if weight>0 and mask.any(): loss=loss+weight*torch.nn.functional.cross_entropy(ph[mask],pb[mask])
  opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
  if st%100==0 or st==steps:
   net.eval();
   with torch.inference_mode(): q,_=net(Xt.to(device)); sc=q.sigmoid().cpu().numpy()
   m=eval_scores(sc,Y,src,~tr); hist.append({'step':st,'loss':float(loss.detach().cpu()),'dev_macro_f':m['macro_f']})
   if m['macro_f']>best[0]: best=(m['macro_f'],{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
   net.train()
 net.load_state_dict(best[1]); net.eval();
 with torch.inference_mode(): q,ph=net(Xt.to(device)); sc=q.sigmoid().cpu().numpy(); pc=ph.argmax(-1).cpu().numpy()
 return net,sc,pc,{'phase_weight':weight,'params':sum(p.numel() for p in net.parameters()),'dev':eval_scores(sc,Y,src,~tr),'train':eval_scores(sc,Y,src,tr),'history':hist}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--annotations',type=Path,required=True); ap.add_argument('--frames-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
 import torch
 z=np.load(a.cache,allow_pickle=False); X=z['features'][:,:,0,:].astype(np.float32); Y=z['labels'].astype(np.float32); src=z['src'].astype(str); split=z['split'].astype(str); ph=load_phase(a.index,a.annotations,a.frames_root,z['video_id'].astype(str)); P=np.stack([ph[v][1] for v in z['video_id'].astype(str)]); interval_rows=np.asarray([bool(ph[v][2]) for v in z['video_id'].astype(str)])
 models=[]
 for w in (0.,.3):
  net,sc,pc,m=train(X,Y,P,src,split,w,20261019, a.steps,a.device); models.append(m); torch.save({'state_dict':net.state_dict(),'phase_weight':w,'params':m['params']},a.output/('control.pt' if w==0 else 'state.pt'))
 base=models[0]['dev']['per_source']; state=models[1]['dev']['per_source']; out={'experiment_id':'R97_EVENT_STATE_AUX_TCN','official_test_labels_used':False,'protocol':'R92 frozen RGB cache + public PHD2 train.json interval phase mask; rows without intervals phase-masked','cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'rows':len(Y),'sources':len(set(src)),'interval_rows':int(interval_rows.sum()),'interval_sources':int(len(set(src[interval_rows]))),'models':models,'paired_dev':bootstrap_delta(base,state),'decision':{'gate_ci_lower_gt_zero':False,'official_f_video':None,'note':'diagnostic only; no package until independent source-level fixed-budget gain'}}; (a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'models':[{k:v for k,v in m.items() if k not in ('history','train')} for m in models],'paired_dev':out['paired_dev']},indent=2))
if __name__=='__main__': main()
