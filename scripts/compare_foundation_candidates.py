"""Compare frozen DEV selections; do not claim test or official scores."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 records={k:json.loads((a.root/k/'metrics.json').read_text()) for k in ['deit','videomae','internvideo']}
 ref={r['video_id']:r for r in records['deit']['best']['per_video']};rng=np.random.default_rng(20260926);out={}
 for k,d in records.items():
  per={r['video_id']:r for r in d['best']['per_video']};assert set(per)==set(ref);diff={}
  for metric in ['f1','spearman','ndcg']:
   delta=np.array([per[v][metric]-ref[v][metric] for v in sorted(ref) if per[v][metric] is not None and ref[v][metric] is not None]);draw=delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1)
   diff[metric]={'mean_delta':float(delta.mean()),'ci95':np.quantile(draw,[.025,.975]).tolist(),'paired_sources':len(delta)}
  out[k]={'best_epoch':d['best']['epoch'],'dev':{m:d['best'][m] for m in ['f1','spearman','ndcg','selected_ratio']},'paired_minus_deit':diff}
 report={'protocol':'NATIVE_FOUNDATION_CANDIDATES_V1','validation_scope':'24-source DEV; checkpoints selected using this DEV; no holdout evaluation; native rated query clips only, not AIC metric','candidates':out,'official_f_video':None}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
