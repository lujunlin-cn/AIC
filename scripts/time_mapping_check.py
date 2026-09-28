"""Separate anchor-resolution limits from a systematic score->time mapping offset.

Facts checked in code (aic/foundation.py, scripts/foundation_candidate_train.py,
aic/video.py expand_scores):
  * anchors every 4 samples at 2 fps -> one score per ~2 s (resolution);
  * the 16-sample clip is centred on the anchor (offsets -8..+7 -> [-4 s,+3.5 s]);
  * training pairs anchor time t with label bin floor(t/2), i.e. the anchor sits
    at the START of its 2 s bin (bin centre = anchor + 1 s);
  * dev eval / official expand_scores place the score AT the anchor time and
    interpolate, so clip i (centre 2i+1) receives mean(anchor i, anchor i+1).
This script scores V0 dev with (a) current placement, (b) scores shifted +1 s so
each anchor's score sits at the centre of the bin it was trained on, and
(c) nearest-anchor lookup (pure resolution, no interpolation).  Dev only.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from aic.models import TemporalUNet
from scripts.foundation_candidate_train import ndcg


def main():
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--v0',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 ck=torch.load(a.v0,map_location='cpu',weights_only=False);head=TemporalUNet(ck['input_dim']);head.load_state_dict(ck['model']);head.eval()
 recs=[r for r in map(json.loads,open(a.records)) if r['split']=='dev'];modes={'current_interp_at_anchor':0.,'shift_plus_1s_to_bin_centre':1.};res={m:[] for m in list(modes)+['nearest_trained_anchor']};offs=[]
 for r in recs:
  with np.load(a.cache/(r['vid']+'.npz')) as z:x=z['features'].astype(np.float32);t=z['timestamps'].astype(np.float64)
  with torch.inference_mode():s=head(torch.from_numpy(x)[None])[0].sigmoid().numpy()
  bins=np.minimum((t/2).astype(int),len(r['labels'])-1);offs+=list((bins*2+1)-t)
  for q in r['queries']:
   ids=np.array(q['relevant_clip_ids']);y=np.mean(q['saliency_scores'],axis=1)/4.
   for m,sh in modes.items():
    pr=np.interp(ids*2.+1,t+sh,s);res[m].append((spearmanr(pr,y).statistic if pr.std()>0 else np.nan,ndcg(pr,y)))
   # anchor whose training label bin is exactly clip i (resolution-only lookup)
   lut={int(b):float(v) for b,v in zip(bins,s)};pr=np.array([lut.get(int(i),np.interp(i*2.+1,t+1,s)) for i in ids])
   res['nearest_trained_anchor'].append((spearmanr(pr,y).statistic if pr.std()>0 else np.nan,ndcg(pr,y)))
 out={'dev_queries':len(res['current_interp_at_anchor']),'anchor_to_bin_centre_offset_s':{'mean':float(np.mean(offs)),'min':float(np.min(offs)),'max':float(np.max(offs))},
  'anchor_step_s':'~2.0 (4 samples at 2 fps)','metrics':{m:{'spearman':float(np.nanmean([v[0] for v in vs])),'ndcg':float(np.mean([v[1] for v in vs]))} for m,vs in res.items()},
  'note':'metrics are query-level means on annotated clips only (same as V0 eval), not AIC F; all current packages are all-select so this offset does not change them','official_f_video':None}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
