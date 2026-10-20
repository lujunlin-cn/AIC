#!/usr/bin/env python3
"""One target-aligned ranking test for DINOv3 spatial candidates.

The first DINOv3 spatial probe used SmoothL1 regression, which can collapse
the 129-window distribution toward the center.  This script keeps the same
visual features, legal windows and source split but changes only the
candidate objective: pairwise ranking and listwise soft targets.  It is a
diagnostic, not a parameter sweep; both objectives run once with the same
seed and training budget.
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from dinov3_runtime import encode_rgb, load_model
from dinov3_spatial_patch_probe import table_rows, candidate_features, candidate_offsets, candidate_boxes, iou

def source_macro(vals, src):
    d=defaultdict(list)
    for v,s in zip(vals,src): d[str(s)].append(float(v))
    return float(np.mean([np.mean(v) for v in d.values()])), len(d)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--t5-root',required=True); ap.add_argument('--rv-frames',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--dinov3-repo',required=True); ap.add_argument('--weights',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--dtype',default='float16')
    ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--steps',type=int,default=4000); ap.add_argument('--seed',type=int,default=20261020)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); root=Path(a.t5_root)
    rows,images=table_rows(root,['rv_train','live_train','rv_dev','live_dev'],Path(a.rv_frames)); model,meta=load_model(a.dinov3_repo,a.weights,a.device,a.dtype)
    keys=list(images); cache={}; t0=time.time()
    for st in range(0,len(keys),a.batch_size):
        kk=keys[st:st+a.batch_size]; frames=[np.asarray(Image.open(images[k]).convert('RGB'),np.uint8) for k in kk]
        e=encode_rgb(model,frames,a.device,a.batch_size,224,'letterbox',True)
        for i,k in enumerate(kk): cache[k]=(e['patch_tokens'][i].astype(np.float16),e['pooled'][i].astype(np.float16),e['transforms'][i])
        if st==0 or st+len(kk)==len(keys) or st//a.batch_size%10==0: print(json.dumps({'frames':st+len(kk),'total':len(keys),'elapsed_s':round(time.time()-t0,1)}),flush=True)
    X=[]; Y=[]; S=[]; D=[]; split=[]; rng=np.random.default_rng(a.seed)
    for r in rows:
        g,c,tr=cache[r['key']]; x=candidate_features(g,c,tr,r['W'],r['H'],r['ratio'],'patch',rng); off=candidate_offsets(r['W'],r['H'],r['ratio'])[0]; y=iou(candidate_boxes(off,r['W'],r['H'],r['ratio']),np.asarray(r['gt'])).mean(1)
        X.append(x); Y.append(y.astype(np.float32)); S.append(r['source']); D.append(r['source'].split(':',1)[0]); split.append(r['split'])
    X=np.stack(X); Y=np.stack(Y); S=np.asarray(S); D=np.asarray(D); split=np.asarray(split); tr=split=='train'; va=~tr
    import torch
    torch.manual_seed(a.seed)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(X.shape[-1],256),torch.nn.GELU(),torch.nn.Linear(256,1))
        def forward(self,z): return self.net(z).squeeze(-1)
    results={'protocol':'DINOv3 patch; fixed 129 legal windows; same source split as regression probe; one pairwise and one listwise objective','model':meta,'rows':len(rows),'dev_sources':len(set(S[va])),'results':[]}
    xt=torch.from_numpy(X[tr]).float(); yt=torch.from_numpy(Y[tr]).float(); xv=torch.from_numpy(X[va]).float().to(a.device); yv=Y[va]
    for objective in ('pairwise','listwise'):
        m=Head().to(a.device); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-2); rr=np.random.default_rng(a.seed); bs=min(32,len(xt))
        for step in range(a.steps):
            ix=rr.integers(0,len(xt),bs); z=xt[ix].to(a.device); target=yt[ix].to(a.device); p=m(z)
            if objective=='pairwise':
                j=torch.from_numpy(rr.integers(0,129,bs)).to(a.device); k=torch.from_numpy(rr.integers(0,129,bs)).to(a.device); bi=torch.arange(bs,device=a.device); d=(target[bi,j]-target[bi,k]); keep=torch.abs(d)>.03
                if keep.any(): loss=torch.nn.functional.softplus(-(torch.sign(d[keep])*(p[bi,j][keep]-p[bi,k][keep]))).mean()
                else: loss=(p*0).sum()
            else:
                loss=torch.nn.functional.kl_div(torch.log_softmax(p/0.07,dim=1),torch.softmax(target/0.07,dim=1),reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.inference_mode(): pred=m(xv).cpu().numpy(); top=pred.argmax(1)
        top_iou=yv[np.arange(len(top)),top]; center=yv[:,64]; oracle=yv.max(1); sm,n=source_macro(top_iou,S[va]); cm,_=source_macro(center,S[va]);
        by={}
        for dom in ('rv','live'):
            mask=D[va]==dom; by[dom]={'rows':int(mask.sum()),'sources':len(set(S[va][mask])),'top_iou':float(top_iou[mask].mean()),'center_iou':float(center[mask].mean()),'source_macro_top':source_macro(top_iou[mask],S[va][mask])[0],'source_macro_center':source_macro(center[mask],S[va][mask])[0]}
        rec={'objective':objective,'params':sum(q.numel() for q in m.parameters()),'steps':a.steps,'source_macro_top_iou':sm,'source_macro_center_iou':cm,'delta_vs_center':sm-cm,'row_top_iou':float(top_iou.mean()),'row_center_iou':float(center.mean()),'row_oracle_iou':float(oracle.mean()),'domains':by}
        results['results'].append(rec); print(json.dumps(rec),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
