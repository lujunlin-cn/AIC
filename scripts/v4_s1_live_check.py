"""V4 S1 report-only transfer check: hold vs linear interpolation on LIVE-YT-VC.

LIVE-YT-VC: x axis, window .316 (the official x-axis geometry), 1 annotator,
~30 sparse human boxes per video (no interpolated GT), 1 s Qwen points (the
T5 point run; DENSE 0.5 s points do not exist for LIVE).  Units come from the
frozen T5 tables; N0 pipeline (NOFACE gate, B0 EMA / resets, max window) with
only the propagation between keyframes changed.  LIVE val 178 is reused
validation (T5 confirm3); LIVE dev fold 156 was the T5 selection fold.  Not a
promotion criterion (the S1 prereg uses RetargetVid only).
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from aic.max_window_path import ema_offsets,qwen_centres,qwen_centres_interp
from scripts.benchmark_spatial import iou
from scripts.teacher_diag_eval import win_boxes,boot


def score(u,fn):
 c,_=fn(u['pts'],u['keys'],u['reset'],u['raw'],u['comp']);c=np.where(u['face'],u['raw'],c)
 off=ema_offsets(c,u['reset'],u['W'],u['H'],u['ratio'])[u['gf']]
 return float(iou(win_boxes(off,u['W'],u['H'],u['ratio'])[None],u['gt']).mean())


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tables',type=Path,default=Path('/data/aic/experiments/T5_CROPHEAD_V3/tables'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out={'protocol':__doc__.strip().splitlines()[0],'official_f_video':None}
 for name in ('live_dev','live_confirm3','rv_confirm2'):
  us=pickle.loads((a.tables/f'{name}.pkl').read_bytes());per={}
  for u in us:
   if not len(u['J']):continue
   per.setdefault((u['ds'],u['vid']),[]).append(score(u,qwen_centres_interp)-score(u,qwen_centres))
  d=np.array([np.mean(v) for v in per.values()])
  out[name]={'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min()),'points':'1 s (D1)'}
  print(name,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[name].items() if k!='ci95'},[round(x,4) for x in out[name]['ci95']],flush=True)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
if __name__=='__main__':main()
