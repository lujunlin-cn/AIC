#!/usr/bin/env python3
"""Train the shared Temporal U-Net on frozen non-ResNet embeddings.

This is a B0 representation probe. It intentionally does not export a
competition bundle until the raw-video ViT encoder and loader are integrated.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from aic.features import FeatureCacheDataset, collate_feature_batch
from aic.models import TemporalUNet
from aic.train import video_proxy_metrics, TVSUM_TARGET_THRESHOLD

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--val',required=True); ap.add_argument('--output',required=True); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--device',default='cuda'); ap.add_argument('--seed',type=int,default=20260925); a=ap.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 dev=torch.device(a.device); tr=FeatureCacheDataset(a.train); va=FeatureCacheDataset(a.val)
 args={'batch_size':1,'num_workers':2,'collate_fn':collate_feature_batch,'pin_memory':dev.type=='cuda'}
 tl=DataLoader(tr,shuffle=True,**args); vl=DataLoader(va,shuffle=False,**args); model=TemporalUNet(768).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=(-1,None); hist=[]; start=time.time()
 def epoch(loader,train):
  model.train(train); losses=[]; rows=[]
  for b in loader:
   x=b['features'].to(dev); y=b['labels'].to(dev); m=b['mask'].to(dev); lengths=b['lengths'].to(dev); z=model(x,lengths=lengths); loss=nn.functional.binary_cross_entropy_with_logits(z[m],y[m])
   if train: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
   losses.append(float(loss.detach().cpu()));
   for i,v in enumerate(b['video_ids']): rows.append(video_proxy_metrics(z[i],y[i],m[i],.5,str(v)))
  valid=[r for r in rows if r['f1'] is not None]; return {'loss':float(np.mean(losses)),'video_macro_f1':float(np.mean([r['f1'] for r in valid])),'mean_spearman':float(np.mean([r['spearman'] for r in valid])),'empty_prediction_rate':float(np.mean([r['empty_prediction'] for r in valid])),'per_video':valid}
 for e in range(a.epochs):
  tm=epoch(tl,True); vm=epoch(vl,False); hist.append({'epoch':e,'train':tm,'val':vm});
  if vm['video_macro_f1']>best[0]: best=(vm['video_macro_f1'],e); torch.save({'model':model.state_dict(),'input_dim':768,'seed':a.seed},Path(a.output).with_suffix('.pt'))
 out={'run_id':'B0_probe_vit_b16','backbone':'torchvision_vit_b_16_imagenet1k_v1','input_dim':768,'target_protocol':'tvsum_summary_mean_norm_ge_0.5_v1','best_video_macro_f1':best[0],'best_epoch':best[1],'elapsed_seconds':time.time()-start,'history':hist,'submission_candidate':False,'reason':'raw-video ViT loader/export not integrated'}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps({k:out[k] for k in ('best_video_macro_f1','best_epoch','elapsed_seconds')},indent=2))
if __name__=='__main__': main()
