"""V4 S1: linear interpolation of the Qwen point between same-shot keyframes.

Prereg configs/V4_S1_INTERP_PREREG.json.  Protocol of scripts/v4_e1_obs_eval.py
(RetargetVid 6 annotators, NOFACE gate, B0 EMA / resets, max legal window,
per-video mean over both ratios, video macro, bootstrap).  Primary
INTERP_D2 - DENSENF; INTERP_D4 - OBS025 is reported for additivity only.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,ema_offsets,qwen_centres,qwen_centres_interp
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import frame_iou,boot
from scripts.v4_e1_obs_eval import SPLITS,MOTION_THR


def run(fn,pts,keys,z,W,H,ratio,comp,gt):
 reset=z['reset'].astype(bool);raw=z['raw'][:,comp];face=z['chosen']>=0
 c,_=fn(pts,keys,reset,raw,comp);c=np.where(face,raw,c)
 return frame_iou(ema_offsets(c,reset,W,H,ratio),W,H,ratio,gt)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True)
 ap.add_argument('--d2',type=Path,required=True);ap.add_argument('--d4',type=Path,required=True);ap.add_argument('--splits',default='dev2')
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 P=('DENSENF','INTERP_D2','OBS025','INTERP_D4');rows=[]
 for split in a.splits.split(','):
  for vid in SPLITS[split]:
   q2=json.loads((a.d2/f'{vid}.json').read_text());q4=json.loads((a.d4/f'{vid}.json').read_text());per={}
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio);comp=0 if axis==0 else 1;L=W if comp==0 else H
    k2,p2=q2['keyframes'],q2['ratios'][r]['points'];k4,p4=q4['keyframes'],q4['ratios'][r]['points'];reset=z['reset'].astype(bool)
    v={'DENSENF':run(qwen_centres,p2,k2,z,W,H,ratio,comp,gt),'INTERP_D2':run(qwen_centres_interp,p2,k2,z,W,H,ratio,comp,gt),
       'OBS025':run(qwen_centres,p4,k4,z,W,H,ratio,comp,gt),'INTERP_D4':run(qwen_centres_interp,p4,k4,z,W,H,ratio,comp,gt)}
    gc=((gt[...,comp]+gt[...,comp+2])/2).mean(0);motion=float(np.abs(np.diff(gc/L))[~reset[1:]].mean())
    per[r]={'axis':'x' if comp==0 else 'y','motion':'fast' if motion>MOTION_THR else 'slow',**{k:float(x.mean()) for k,x in v.items()}}
   rows.append({'split':split,'vid':vid,'per_ratio':per,**{p:float(np.mean([per[r][p] for r in RATIOS])) for p in P}})
 def paired(sel,p,b,key=None):
  d=np.array([np.mean([v[p]-v[b] for v in x['per_ratio'].values() if key is None or key(v)]) for x in sel if key is None or any(key(v) for v in x['per_ratio'].values())])
  return {'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 out={'prereg':'configs/V4_S1_INTERP_PREREG.json','official_f_video':None}
 for split in a.splits.split(','):
  sel=[x for x in rows if x['split']==split];res={p:float(np.mean([x[p] for x in sel])) for p in P}
  for p,b in (('INTERP_D2','DENSENF'),('INTERP_D4','OBS025'),('INTERP_D4','DENSENF'),('OBS025','DENSENF')):
   res[f'{p}-{b}']={'all':paired(sel,p,b),'axis_x':paired(sel,p,b,lambda v:v['axis']=='x'),'axis_y':paired(sel,p,b,lambda v:v['axis']=='y'),
    'motion_fast':paired(sel,p,b,lambda v:v['motion']=='fast'),'motion_slow':paired(sel,p,b,lambda v:v['motion']=='slow')}
  out[split]=res
 out['per_video']=rows;a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 for split in a.splits.split(','):
  print(split,{p:round(out[split][p],4) for p in P})
  for k,v in out[split].items():
   if isinstance(v,dict):
    for kk,x in v.items():print(' ',k,kk,x['videos'],round(x['mean'],4),[round(y,4) for y in x['ci95']],x['better'],x['worse'],round(x['worst'],4))
if __name__=='__main__':main()
