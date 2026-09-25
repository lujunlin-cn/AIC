#!/usr/bin/env python3
"""Evaluate an A0 cached-feature checkpoint (state keys include backbone.*)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from aic.features import FeatureCacheDataset,collate_feature_batch
from aic.models import A0Model
from aic.train import video_proxy_metrics
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--thresholds',default='0.3,0.35,0.4,0.45'); a=ap.parse_args(argv)
 dev=torch.device(a.device); ds=FeatureCacheDataset(a.manifest); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate_feature_batch); model=A0Model().to(dev); state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(state['model']); model.eval(); outputs=[]
 with torch.inference_mode():
  for b in dl: outputs.append((model(b['features'].to(dev),lengths=b['lengths'].to(dev)).cpu(),b['labels'],b['mask'],b['video_ids']))
 report=[]
 for th in [float(x) for x in a.thresholds.split(',')]:
  rows=[]
  for z,y,m,ids in outputs:
   for i,v in enumerate(ids): rows.append(video_proxy_metrics(z[i],y[i],m[i],th,str(v)))
  valid=[r for r in rows if r['f1'] is not None]; report.append({'threshold':th,'video_macro_f1':float(np.mean([r['f1'] for r in valid])),'empty_prediction_rate':float(np.mean([r['empty_prediction'] for r in valid])),'mean_prediction_rate':float(np.mean([r['prediction_rate'] for r in valid])),'per_video':valid})
 out={'checkpoint':a.checkpoint,'manifest':a.manifest,'thresholds':report}; Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps({'thresholds':[(r['threshold'],r['video_macro_f1']) for r in report]},indent=2))
if __name__=='__main__': main()
