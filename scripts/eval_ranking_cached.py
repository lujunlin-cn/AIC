#!/usr/bin/env python3
"""Evaluate cached temporal checkpoints with threshold-free ranking metrics."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from aic.features import FeatureCacheDataset, collate_feature_batch
from aic.models import TemporalUNet
from aic.temporal_metrics import ranking_report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda')
    a=ap.parse_args(); dev=torch.device(a.device if a.device!='auto' or torch.cuda.is_available() else 'cpu')
    ds=FeatureCacheDataset(a.manifest); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_feature_batch)
    model=TemporalUNet(int(ds[0]['features'].shape[1])).to(dev); state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(state['model']); model.eval(); rows=[]
    with torch.inference_mode():
      for b in dl:
        z=model(b['features'].to(dev), lengths=b['lengths'].to(dev)).sigmoid().cpu().numpy()
        for i,vid in enumerate(b['video_ids']):
          n=int(b['lengths'][i]); rows.append({'video_id':str(vid), **ranking_report(z[i,:n], b['labels'][i,:n].numpy())})
    keys=['spearman','kendall_tau','ndcg','ndcg_at_15pct','top15_ap']; out={'manifest':a.manifest,'checkpoint':a.checkpoint,'video_count':len(rows),'mean':{k:float(np.mean([r[k] for r in rows])) for k in keys},'std':{k:float(np.std([r[k] for r in rows])) for k in keys},'per_video':rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['mean'],indent=2))
if __name__=='__main__': main()
