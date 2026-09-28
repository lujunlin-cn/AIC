"""V4 S2: median-of-3 outlier rejection of same-shot keyframe points.

Prereg configs/V4_S2_MEDIAN3_PREREG.json.  Protocol of scripts/v4_e1_obs_eval.py.
Also a report-only analysis of the B0 EMA on top of S1 interpolation
(alpha .25 = parent, .5, 1 = no EMA); not a promotion criterion.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,ema_offsets,qwen_centres,qwen_centres_interp
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import frame_iou,boot
from scripts.v4_e1_obs_eval import SPLITS


def median3(points,keyframes,reset,axis):
 """Replace each valid point by the median of its valid same-shot neighbours."""
 shot=np.cumsum(np.asarray(reset,bool));ok=lambda p:p is not None and not p[2];out=[None if p is None else list(p) for p in points]
 for j in range(1,len(points)-1):
  a,b,c=points[j-1],points[j],points[j+1]
  if not (ok(a) and ok(b) and ok(c)):continue
  if keyframes[j+1]>=len(reset) or not (shot[keyframes[j-1]]==shot[keyframes[j]]==shot[keyframes[j+1]]):continue
  out[j][axis]=float(np.median([a[axis],b[axis],c[axis]]))
 return out


def run(fn,pts,keys,z,W,H,ratio,comp,gt,alpha=.25):
 reset=z['reset'].astype(bool);raw=z['raw'][:,comp];face=z['chosen']>=0
 c,_=fn(pts,keys,reset,raw,comp);c=np.where(face,raw,c)
 return frame_iou(ema_offsets(c,reset,W,H,ratio,alpha=alpha),W,H,ratio,gt)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True)
 ap.add_argument('--d2',type=Path,required=True);ap.add_argument('--splits',default='dev2');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 P=('DENSENF','MEDIAN3','INTERP','MEDIAN3_INTERP','INTERP_A50','INTERP_A100');rows=[];changed=0;total=0
 for split in a.splits.split(','):
  for vid in SPLITS[split]:
   q2=json.loads((a.d2/f'{vid}.json').read_text());per={}
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio);comp=0 if axis==0 else 1
    k2,p2=q2['keyframes'],q2['ratios'][r]['points'];pm=median3(p2,k2,z['reset'],comp)
    changed+=sum(1 for x,y in zip(p2,pm) if x is not None and y is not None and abs(x[comp]-y[comp])>1e-9);total+=sum(1 for x in p2 if x is not None and not x[2])
    v={'DENSENF':run(qwen_centres,p2,k2,z,W,H,ratio,comp,gt),'MEDIAN3':run(qwen_centres,pm,k2,z,W,H,ratio,comp,gt),
       'INTERP':run(qwen_centres_interp,p2,k2,z,W,H,ratio,comp,gt),'MEDIAN3_INTERP':run(qwen_centres_interp,pm,k2,z,W,H,ratio,comp,gt),
       'INTERP_A50':run(qwen_centres_interp,p2,k2,z,W,H,ratio,comp,gt,.5),'INTERP_A100':run(qwen_centres_interp,p2,k2,z,W,H,ratio,comp,gt,1.)}
    per[r]={'axis':'x' if comp==0 else 'y',**{k:float(x.mean()) for k,x in v.items()}}
   rows.append({'split':split,'vid':vid,'per_ratio':per,**{p:float(np.mean([per[r][p] for r in RATIOS])) for p in P}})
 def paired(sel,p,b,key=None):
  d=np.array([np.mean([v[p]-v[b] for v in x['per_ratio'].values() if key is None or key(v)]) for x in sel])
  return {'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 out={'prereg':'configs/V4_S2_MEDIAN3_PREREG.json','points_changed':changed,'valid_points':total,'official_f_video':None}
 for split in a.splits.split(','):
  sel=[x for x in rows if x['split']==split];res={p:float(np.mean([x[p] for x in sel])) for p in P}
  for p,b in (('MEDIAN3','DENSENF'),('MEDIAN3_INTERP','INTERP'),('INTERP_A50','INTERP'),('INTERP_A100','INTERP')):
   res[f'{p}-{b}']={'all':paired(sel,p,b),'axis_x':paired(sel,p,b,lambda v:v['axis']=='x'),'axis_y':paired(sel,p,b,lambda v:v['axis']=='y')}
  out[split]=res
 out['per_video']=rows;a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 print('points changed',changed,'of',total)
 for split in a.splits.split(','):
  print(split,{p:round(out[split][p],4) for p in P})
  for k,v in out[split].items():
   if isinstance(v,dict):
    for kk,x in v.items():print(' ',k,kk,x['videos'],round(x['mean'],4),[round(y,4) for y in x['ci95']],x['better'],x['worse'],round(x['worst'],4))
if __name__=='__main__':main()
