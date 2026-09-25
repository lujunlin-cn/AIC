#!/usr/bin/env python3
"""Evaluate a cached TemporalUNet checkpoint over fixed thresholds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from aic.features import FeatureCacheDataset, collate_feature_batch
from aic.models import TemporalUNet
from aic.train import video_proxy_metrics

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--thresholds',default='0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6')
    a=ap.parse_args(argv); dev=torch.device(a.device if a.device!='auto' or torch.cuda.is_available() else 'cpu')
    ds=FeatureCacheDataset(a.manifest); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate_feature_batch,pin_memory=dev.type=='cuda')
    dim=int(ds[0]['features'].shape[1]); model=TemporalUNet(dim).to(dev); state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(state['model']); model.eval(); outputs=[]
    with torch.inference_mode():
      for b in loader:
        z=model(b['features'].to(dev),lengths=b['lengths'].to(dev)); outputs.append((z.detach().cpu(),b['labels'].clone(),b['mask'].clone(),b['video_ids']))
    report=[]
    for threshold in [float(x) for x in a.thresholds.split(',')]:
      rows=[]
      for z,y,m,ids in outputs:
        for i,vid in enumerate(ids): rows.append(video_proxy_metrics(z[i],y[i],m[i],threshold,str(vid)))
      valid=[r for r in rows if r.get('f1') is not None]
      report.append({'threshold':threshold,'video_macro_f1':float(np.mean([r['f1'] for r in valid])),'mean_precision':float(np.mean([r['precision'] for r in valid if r['precision'] is not None])),'mean_recall':float(np.mean([r['recall'] for r in valid if r['recall'] is not None])),'empty_prediction_rate':float(np.mean([r['empty_prediction'] for r in valid])),'mean_prediction_rate':float(np.mean([r['prediction_rate'] for r in valid])),'per_video':valid})
    out={'checkpoint':a.checkpoint,'manifest':a.manifest,'thresholds':report,'best_threshold_by_macro_f1':max(report,key=lambda r:r['video_macro_f1'])['threshold']}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps({'best_threshold_by_macro_f1':out['best_threshold_by_macro_f1'],'best_macro_f1':max(x['video_macro_f1'] for x in report)},indent=2))
if __name__=='__main__': main()
