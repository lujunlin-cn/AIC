#!/usr/bin/env python3
"""Train a direct utility readout for the deployed V9/B3 candidate.

The R70 feature shards contain V9 vision features and native IoU for all 129
legal windows.  A frozen V9 A0/B3 head first selects the candidate that the
spatial deployment would emit.  The target for this experiment is the native
IoU of that selected candidate, rather than max IoU over the candidate pool.
A small source-balanced scalar head then predicts that target from the
selected candidate feature and frozen head confidence signals.

This is a public source-disjoint diagnostic. It never reads official labels.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC=129

class CandidateHead(nn.Module):
    def __init__(self,d=2305,ch=128,heads=4,ff=256):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.Linear(512,ch)); self.h=heads
        self.qkv=nn.ModuleList([nn.Linear(ch,3*ch) for _ in range(2)]); self.proj_o=nn.ModuleList([nn.Linear(ch,ch) for _ in range(2)])
        self.ln1=nn.ModuleList([nn.LayerNorm(ch) for _ in range(2)]); self.ln2=nn.ModuleList([nn.LayerNorm(ch) for _ in range(2)])
        self.ff1=nn.ModuleList([nn.Linear(ch,ff) for _ in range(2)]); self.ff2=nn.ModuleList([nn.Linear(ff,ch) for _ in range(2)])
        self.drop=nn.Dropout(0.1); self.out=nn.Linear(ch,1)
    def forward(self,x):
        h=self.proj(x)
        for i in range(2):
            B,N,C=h.shape; q,k,v=self.qkv[i](h).reshape(B,N,3,self.h,C//self.h).permute(2,0,3,1,4).unbind(0)
            a=torch.softmax(q@k.transpose(-1,-2)/(C//self.h)**.5,dim=-1)
            h=self.ln1[i](h+self.drop(self.proj_o[i]((a@v).transpose(1,2).reshape(B,N,C))))
            h=self.ln2[i](h+self.drop(self.ff2[i](F.gelu(self.ff1[i](h)))))
        return self.out(h).squeeze(-1)

class UtilityHead(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def load_rows(root:Path, v9_path:Path, device:str):
    ck=torch.load(v9_path,map_location='cpu',weights_only=False)
    v9=CandidateHead(); v9.load_state_dict(ck['state_dict']); v9.to(device).eval()
    feats=[]; targets=[]; base_scores=[]; sources=[]; names=[]; selected=[]
    t0=time.time(); paths=sorted((root/'shards').glob('shard_*.npz'))
    with torch.inference_mode():
      for si,p in enumerate(paths):
        z=np.load(p,allow_pickle=True); x=z['x'].astype(np.float32); y=z['y'].astype(np.float32)
        src=z['source'].astype(str); nm=z['name'].astype(str)
        out=[]
        for st in range(0,len(x),256): out.append(v9(torch.from_numpy(x[st:st+256]).to(device)).float().cpu().numpy())
        pred=np.concatenate(out); j=pred.argmax(1); row=np.arange(len(x))
        # Direct target: native IoU of the V9/B3 candidate actually selected.
        feats.append(x[row,j]); targets.append(y[row,j]); base_scores.append(pred[row,j])
        # side features are frozen V9 confidence signals and selected position.
        srt=np.sort(pred,axis=1); gap=srt[:,-1]-srt[:,-2]; cen=pred[:,NC//2]; pos=j.astype(np.float32)/(NC-1)
        side=np.stack([pred[row,j],gap,cen,pos,pred.mean(1)],axis=1).astype(np.float32)
        feats[-1]=np.concatenate([feats[-1],side],axis=1)
        sources.append(src); names.append(nm); selected.append(j)
        if si%10==0: print(json.dumps({'shard':si,'rows':int(sum(len(a) for a in targets)),'elapsed_s':round(time.time()-t0,1)}),flush=True)
    x=np.concatenate(feats); y=np.concatenate(targets); base=np.concatenate(base_scores); src=np.concatenate(sources); nm=np.concatenate(names); jj=np.concatenate(selected)
    print(json.dumps({'loaded_rows':len(y),'features':x.shape[1],'sources':len(set(src)),'elapsed_s':round(time.time()-t0,1)}),flush=True)
    return x,y,base,src,nm,jj

def source_macro(v,target,src,nm,mask,frac=.85):
    vals=[]
    for s in sorted(set(src[mask])):
        q=mask&(src==s); n=int(q.sum()); k=max(1,int(round(frac*n))); ids=np.where(q)[0][np.argsort(-v[q],kind='mergesort')[:k]]; vals.append(float(target[ids].mean()))
    return float(np.mean(vals)) if vals else None

def spearman(a,b):
    a=np.asarray(a);b=np.asarray(b)
    ra=np.argsort(np.argsort(a,kind='mergesort'),kind='mergesort'); rb=np.argsort(np.argsort(b,kind='mergesort'),kind='mergesort')
    return float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else None

def eval_report(pred,y,base,src,nm,mask):
    out={'rows':int(mask.sum()),'sources':int(len(set(src[mask])))}
    for label,v in [('v9_score',base),('direct_head',pred)]:
        out[label]={'rho_target':spearman(v[mask],y[mask]),'row_mean_selected_iou':float(y[mask].mean()),'top85_selected_iou':source_macro(v,y,src,nm,mask),'top85_target_gain_over_random':None}
    # random source-macro reference is the mean target over all rows per source.
    vals=[]
    for s in sorted(set(src[mask])):
        q=mask&(src==s); vals.append(float(y[q].mean()))
    out['random_source_macro']=float(np.mean(vals)) if vals else None
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--v9-head',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--steps',type=int,default=1200); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--seed',type=int,default=20261019); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    x,y,base,src,nm,jj=load_rows(a.root,a.v9_head,a.device)
    tr=np.char.endswith(nm,'train'); rv=np.equal(nm,'rv_dev'); live=np.equal(nm,'live_dev')
    mu=x[tr].mean(0); sd=x[tr].std(0); sd[sd<1e-5]=1.; xn=(x-mu)/sd
    model=UtilityHead(x.shape[1]).to(a.device).float(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    # Source-balanced sampling avoids letting long videos dominate the target.
    train_sources=sorted(set(src[tr])); by={s:np.flatnonzero(tr&(src==s)) for s in train_sources}; rng=np.random.default_rng(a.seed)
    xt=torch.from_numpy(xn); yt=torch.from_numpy(y)
    best=(-1.,None); hist=[]
    for step in range(1,a.steps+1):
        chosen=rng.choice(train_sources,size=min(a.batch,len(train_sources)),replace=False) if len(train_sources)>=a.batch else rng.choice(train_sources,size=a.batch,replace=True)
        ids=np.array([rng.choice(by[s]) for s in chosen],dtype=np.int64)
        p=model(xt[ids].to(a.device)); loss=F.smooth_l1_loss(p,yt[ids].to(a.device),beta=.1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        if step%100==0 or step==a.steps:
            with torch.inference_mode(): pred=model(xt.to(a.device)).float().cpu().numpy()
            rep=eval_report(pred,y,base,src,nm,rv); hist.append({'step':step,'loss':float(loss),'rv_dev':rep})
            print(json.dumps({'step':step,'loss':float(loss),'rv_dev':rep}),flush=True)
            if rep['direct_head']['top85_selected_iou']>best[0]: best=(rep['direct_head']['top85_selected_iou'],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    if best[1] is not None:model.load_state_dict(best[1])
    with torch.inference_mode(): pred=model(xt.to(a.device)).float().cpu().numpy()
    report={'protocol':'V9 frozen A0/B3 candidate selects j; target=y[j]; source-balanced direct scalar utility head; best selected on RV source-macro top85 IoU','seed':a.seed,'steps':a.steps,'batch':a.batch,'input_dim':int(x.shape[1]),'utility_head_params':int(sum(p.numel() for p in model.parameters())),'train_rows':int(tr.sum()),'train_sources':len(train_sources),'rv_dev':eval_report(pred,y,base,src,nm,rv),'live_dev':eval_report(pred,y,base,src,nm,live),'all':eval_report(pred,y,base,src,nm,np.ones(len(y),bool)),'history':hist}
    a.out.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'mu':mu,'sd':sd,'config':{'input_dim':int(x.shape[1]),'v9_head':str(a.v9_head),'target':'native IoU of frozen V9 argmax candidate'},'report':report},a.out/'direct_utility_head.pt'); (a.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n'); np.savez_compressed(a.out/'selected_public.npz',x=x,y=y,base=base,src=src,names=nm,selected=jj); print(json.dumps({k:report[k] for k in ['utility_head_params','rv_dev','live_dev','all']},indent=2),flush=True)
if __name__=='__main__':main()
