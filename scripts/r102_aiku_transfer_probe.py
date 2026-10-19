#!/usr/bin/env python3
"""R102: AIKU summary supervision -> PHD/GIF target temporal transfer.

AIKU and AIC use different feature spaces, so only the temporal trunk and
readout are transferred; the input projection is re-initialized for the 512D
PHD2 RGB features.  This is an honest head-transfer test, not a claim that
AIKU features are deployable on AIC videos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fixed_f(scores, y, keep=.85):
    k = max(1, int(round(keep * len(y)))); idx = np.argsort(-scores, kind="mergesort")[:k]
    return float(2 * y[idx].sum() / max(k + y.sum(), 1e-12))


def eval_target(scores, y, src, mask):
    per = {}
    for i, s in enumerate(src[mask]): per.setdefault(str(s), []).append(fixed_f(scores[mask][i], y[mask][i]))
    per = {k: float(np.mean(v)) for k, v in per.items()}
    return {"sources": len(per), "macro_f": float(np.mean(list(per.values()))), "per_source": per}


class Net:
    @staticmethod
    def build(in_dim):
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.GELU())
                self.temporal = nn.Conv1d(64, 128, 3, padding=1)
                self.blocks = nn.ModuleList([nn.Conv1d(128, 128, 3, padding=d, dilation=d) for d in (1, 2, 4)])
                self.norm = nn.GroupNorm(8, 128); self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
                self.summary = nn.Linear(128, 1)
            def forward(self, x):
                z = self.in_proj(x); h = self.temporal(z.transpose(1, 2))
                for c in self.blocks: h = __import__('torch').nn.functional.gelu(self.norm(c(h) + h))
                h = h.transpose(1, 2); return self.head(h).squeeze(-1), self.summary(h).squeeze(-1)
        return M()


def pad_batch(items, device):
    import torch
    xs = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x) for x, _, _ in items], batch_first=True).to(device)
    ys = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(y) for _, y, _ in items], batch_first=True).to(device)
    masks = torch.nn.utils.rnn.pad_sequence([torch.ones(len(y), dtype=torch.bool) for _, y, _ in items], batch_first=True).to(device)
    return xs, ys, masks


def load_aiku(path):
    import h5py
    f = h5py.File(path, "r"); out=[]
    for vid in f:
        g=f[vid]; out.append((g["features"][:].astype(np.float32), g["gtscore"][:].astype(np.float32), g["gtsummary"][:].astype(np.float32)))
    f.close(); return out


def train_aiku(data, device, epochs, seed):
    import torch
    torch.manual_seed(seed); rng=np.random.default_rng(seed); net=Net.build(1024).to(device); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=.01)
    ids=np.arange(len(data)); rng.shuffle(ids); split=int(.8*len(ids)); hist=[]
    for ep in range(epochs):
        rng.shuffle(ids)
        for j in range(0,split,8):
            batch=[data[i] for i in ids[j:min(j+8,split)]]; x,y,m=pad_batch([(a,b,c) for a,b,c in batch],device); logits,summary=net(x)
            loss=((logits-y)**2*m).sum()/m.sum()+.2*(torch.nn.functional.binary_cross_entropy_with_logits(summary, torch.zeros_like(summary), reduction='none')*m).sum()/m.sum()
            # Use gtsummary as a second masked task; retain continuous gtscore as main target.
            target_summary=torch.nn.utils.rnn.pad_sequence([torch.from_numpy(data[i][2]) for i in ids[j:min(j+8,split)]],batch_first=True).to(device)
            loss=((logits-y)**2*m).sum()/m.sum()+.2*(torch.nn.functional.binary_cross_entropy_with_logits(summary,target_summary,reduction='none')*m).sum()/m.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
        hist.append({'epoch':ep+1,'loss':float(loss.detach().cpu())})
    return net, {'epochs':epochs,'train_videos':split,'val_videos':len(ids)-split,'history':hist}


def train_target(X,Y,src,split,device,steps,seed,init=None):
    import torch
    torch.manual_seed(seed); rng=np.random.default_rng(seed); net=Net.build(512).to(device)
    if init is not None:
        own=net.state_dict();
        for k,v in init.items():
            if k in own and not k.startswith('in_proj.') and own[k].shape==v.shape: own[k]=v
        net.load_state_dict(own)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=.01); tr=split=='train'; ids=np.flatnonzero(tr); xt=torch.from_numpy(X); yt=torch.from_numpy(Y); best=(-1,None); hist=[]
    for st in range(1,steps+1):
        take=ids[rng.integers(0,len(ids),64)]; logits,_=net(xt[take].to(device)); yb=yt[take].to(device); loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yb)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
        if st%100==0 or st==steps:
            net.eval();
            with torch.inference_mode(): p=net(xt.to(device))[0].sigmoid().cpu().numpy()
            ev=eval_target(p,Y,src,~tr); hist.append({'step':st,'loss':float(loss.detach().cpu()),'dev_macro_f':ev['macro_f']})
            if ev['macro_f']>best[0]: best=(ev['macro_f'],{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
            net.train()
    net.load_state_dict(best[1]); net.eval();
    with torch.inference_mode(): p=net(xt.to(device))[0].sigmoid().cpu().numpy()
    return net, {'dev':eval_target(p,Y,src,~tr),'train':eval_target(p,Y,src,tr),'best_dev':best[0],'history':hist,'params':sum(p.numel() for p in net.parameters())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--aiku-h5',type=Path,required=True); ap.add_argument('--target-cache',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--aiku-epochs',type=int,default=5); ap.add_argument('--target-steps',type=int,default=1200); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    import torch
    data=load_aiku(a.aiku_h5); aiku,ainfo=train_aiku(data,a.device,a.aiku_epochs,20261019); torch.save(aiku.state_dict(),a.out/'aiku_pretrain.pt')
    z=np.load(a.target_cache,allow_pickle=False); X=z['features'][:,:,0,:].astype(np.float32); Y=z['labels'].astype(np.float32); src=z['src'].astype(str); split=z['split'].astype(str)
    init=torch.load(a.out/'aiku_pretrain.pt',map_location='cpu',weights_only=True)
    models={}
    for name,st in [('target_only',None),('aiku_transfer',init)]:
        net,met=train_target(X,Y,src,split,a.device,a.target_steps,20261019,st); models[name]=met; torch.save(net.state_dict(),a.out/(name+'.pt'))
    out={'experiment_id':'R102_AIKU_TRANSFER_PROBE','protocol':'AIKU gtscore/gtsummary pretrain -> R92 PHD2/GIF RGB target; input projection reinitialized, temporal trunk transferred','official_test_labels_used':False,'audio':'excluded_due_to_project_negative_transfer_evidence','aiku_sha256':hashlib.sha256(a.aiku_h5.read_bytes()).hexdigest(),'target_cache_sha256':hashlib.sha256(a.target_cache.read_bytes()).hexdigest(),'aiku':ainfo,'models':models}
    (a.out/'metrics.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:{'dev':v['dev'],'best_dev':v['best_dev']} for k,v in models.items()},indent=2))

if __name__=='__main__': main()
