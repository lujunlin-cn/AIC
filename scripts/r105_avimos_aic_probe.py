#!/usr/bin/env python3
"""R105: use AViMoS visual saliency as a native crop utility feature.

Saliency is predicted in batches on the same V9 keyframes used by R70; no
official labels are used.  The final comparison is source-disjoint native
crop IoU, with a row-permuted saliency control.
"""
from __future__ import annotations
import argparse, importlib.util, json, pickle
from pathlib import Path
import numpy as np

def load_r70(path):
    s=importlib.util.spec_from_file_location("r70",path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def load_r70_rows(path):
    ps=sorted((Path(path)/"shards").glob("shard_*.npz")); z=[np.load(p,allow_pickle=True) for p in ps]
    return {k:np.concatenate([q[k] for q in z]) for k in ("x","y","source","name")}

def saliency_stats(smap, W, H, ratio, r70):
    off, w, h, axis = r70.candidate_offsets(W,H,ratio); n=129; yy,xx=np.mgrid[0:smap.shape[0],0:smap.shape[1]]
    mass=[]; dist=[]; ent=[]; area=[]; total=float(smap.sum())
    cx=float((smap*xx).sum()/max(total,1e-9))/(smap.shape[1]-1); cy=float((smap*yy).sum()/max(total,1e-9))/(smap.shape[0]-1)
    for o in off:
        if axis==0: x1,x2,y1,y2=o,o+w,0,h
        elif axis==1: x1,x2,y1,y2=0,w,o,o+h
        else: x1,x2,y1,y2=0,W,0,H
        ix1=int(np.floor(x1/W*smap.shape[1])); ix2=max(int(np.ceil(x2/W*smap.shape[1])),ix1+1); iy1=int(np.floor(y1/H*smap.shape[0])); iy2=max(int(np.ceil(y2/H*smap.shape[0])),iy1+1)
        z=smap[iy1:iy2,ix1:ix2]; m=float(z.sum()/max(total,1e-9)); mass.append(m)
        qx=((x1+x2)/2)/W; qy=((y1+y2)/2)/H; dist.append((qx-cx)**2+(qy-cy)**2)
        zz=z.reshape(-1); zz=zz/max(zz.sum(),1e-9); ent.append(float(-(zz*np.log(zz+1e-9)).sum())); area.append(float(z.size/smap.size))
    return np.stack([mass,dist,ent,area],1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--r70-script',required=True); ap.add_argument('--r70-out',required=True); ap.add_argument('--t5-dir',required=True); ap.add_argument('--rv-frames',required=True); ap.add_argument('--decoder',required=True); ap.add_argument('--mobilenet',required=True); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--steps',type=int,default=1600); ap.add_argument('--seed',type=int,default=20261019); a=ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    from torchvision.models import mobilenet_v2
    from PIL import Image
    r70=load_r70(a.r70_script); rows=load_r70_rows(a.r70_out); device=torch.device(a.device); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    backbone=mobilenet_v2(weights=None).features.to(device).eval(); backbone.load_state_dict({k:v for k,v in torch.load(a.mobilenet,map_location='cpu',weights_only=True).items() if k.startswith('features.')},strict=False)
    dec=nn.Sequential(nn.Conv2d(1280,256,1),nn.GELU(),nn.Conv2d(256,64,3,padding=1),nn.GELU(),nn.Conv2d(64,1,1)).to(device); dec.load_state_dict(torch.load(a.decoder,map_location='cpu',weights_only=True)); dec.eval(); mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    stats=[]; names=[]; sources=[]; t5=Path(a.t5_dir)
    for name in ['rv_train','live_train','rv_dev','live_dev']:
      units=pickle.loads((t5/'tables'/f'{name}.pkl').read_bytes())
      for unit in units:
        valid=[int(j) for j in unit['J'] if int(unit['keys'][int(j)]) in {int(f) for f in unit['gf']}]; imgs=[]; keys=[]
        for j in valid:
          kf=int(unit['keys'][j]); p=(t5/('keyframes_val' if name=='live_confirm3' else 'keyframes_train')/str(unit['vid'])/f'{kf}.png') if unit['ds']=='live' else (Path(a.rv_frames)/'d1'/str(unit['vid'])/f'{kf}.png')
          im=np.asarray(Image.open(p).convert('RGB').resize((224,128)),np.float32)/255.; imgs.append(torch.from_numpy(im).permute(2,0,1)); keys.append(j)
        with torch.inference_mode():
          pred=[]
          for st in range(0,len(imgs),a.batch):
            z=torch.stack(imgs[st:st+a.batch]).to(device); z=(z-mean)/std; h=backbone(z); sm=F.interpolate(dec(h),size=(32,56),mode='bilinear',align_corners=False).squeeze(1).flatten(1).softmax(1).reshape(-1,32,56).cpu().numpy(); pred.extend(sm)
        for j,sm in zip(keys,pred): stats.append(saliency_stats(sm,unit['W'],unit['H'],unit['ratio'],r70)); names.append(name); sources.append(f"{unit['ds']}:{unit['vid']}")
    stats=np.stack(stats); sources=np.asarray(sources); names=np.asarray(names)
    if len(stats)!=len(rows['y']) or not np.array_equal(sources,rows['source'].astype(str)) or not np.array_equal(names,rows['name'].astype(str)):
      raise RuntimeError(f'row alignment failed stats={len(stats)} rows={len(rows["y"])}')
    np.savez_compressed(out/'features.npz',stats=stats,source=sources,name=names,y=rows['y'],x=rows['x'])
    x=rows['x'].astype(np.float32); y=rows['y'].astype(np.float32); train=names.astype(str).astype('<U50').astype(str); train_mask=np.char.endswith(train,'train'); src=sources
    def run(extra, perm=False):
      xx=np.concatenate([x, extra if not perm else extra[np.random.default_rng(a.seed+7).permutation(len(extra))]],axis=2).astype(np.float32); X=torch.from_numpy(xx); Y=torch.from_numpy(y); rng=np.random.default_rng(a.seed); ids=np.flatnonzero(train_mask)
      class Head(nn.Module):
       def __init__(self):
        super().__init__(); self.p=nn.Sequential(nn.LayerNorm(xx.shape[-1]),nn.Linear(xx.shape[-1],256),nn.GELU(),nn.Linear(256,128)); self.q=nn.MultiheadAttention(128,4,batch_first=True); self.o=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
       def forward(self,z): h=self.p(z); a,_=self.q(h,h,h); return self.o(h+a).squeeze(-1)
      net=Head().to(device); opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3); best=(-1,None)
      def score(p,mask):
       vals=[]
       for s in sorted(set(src[mask])):
        q=mask&(src==s); vals.append(float(y[q,np.argmax(p[q],1)].mean()))
       return float(np.mean(vals))
      for st in range(1,a.steps+1):
       take=rng.choice(ids,min(64,len(ids)),False); p=net(X[take].to(device)); loss=((p-Y[take].to(device))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
       if st%100==0:
        net.eval();
        with torch.inference_mode(): pp=net(X.to(device)).cpu().numpy()
        sc=score(pp,~train_mask)
        if sc>best[0]: best=(sc,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        net.train()
      net.load_state_dict(best[1]); net.eval();
      with torch.inference_mode(): pp=net(X.to(device)).cpu().numpy()
      return {'dev_source_macro':score(pp,~train_mask),'train_source_macro':score(pp,train_mask),'best_dev':best[0],'params':sum(p.numel() for p in net.parameters())}
    result={'experiment_id':'R105_AVIMOS_AIC_UTILITY','official_test_labels_used':False,'audio':'excluded','center_dev_source_macro':float(np.mean([y[(~train_mask)&(src==s),64].mean() for s in set(src[~train_mask])]))}
    result['control']=run(np.zeros_like(stats)); result['avimos']=run(stats); result['avimos_permuted']=run(stats,True); (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
