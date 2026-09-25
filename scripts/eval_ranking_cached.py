#!/usr/bin/env python3
"""Evaluate cached temporal checkpoints with threshold-free ranking metrics."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from aic.features import FeatureCacheDataset, collate_feature_batch
from aic.models import TemporalUNet
from aic.temporal_metrics import ranking_report
from scripts.benchmark_representations import load_head
from aic.models import temporal_shift

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda')
    a=ap.parse_args(); dev=torch.device(a.device if a.device!='auto' or torch.cuda.is_available() else 'cpu')
    ds=FeatureCacheDataset(a.manifest); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_feature_batch)
    model, shift, state=load_head(a.checkpoint,dev); rows=[]
    with torch.inference_mode():
      for b in dl:
        x=b['features'].to(dev)
        if shift: x=temporal_shift(x)
        z=model(x, aux=b['aux'].to(dev) if model.aux_dim else None, lengths=b['lengths'].to(dev)).sigmoid().cpu().numpy()
        for i,vid in enumerate(b['video_ids']):
          n=int(b['lengths'][i]); m=b['mask'][i,:n].numpy(); rows.append({'video_id':str(vid), **ranking_report(z[i,:n][m], b['labels'][i,:n].numpy()[m])})
    keys=['spearman','kendall_tau_b','ndcg','ndcg_at_15pct','ap_fixed_gt_0p5','top15_mean_relevance']; vals={k:[r[k] for r in rows if r[k] is not None] for k in keys}
    out={'manifest':a.manifest,'checkpoint':a.checkpoint,'video_count':len(rows),'mean':{k:float(np.mean(v)) if v else None for k,v in vals.items()},'std':{k:float(np.std(v)) if v else None for k,v in vals.items()},'per_video':rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['mean'],indent=2))
if __name__=='__main__': main()
