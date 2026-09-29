"""V5 P0: E4 x-axis rerank points fed through the INTERP pipeline (mother 48.67).

Prereg configs/V5_P0_XRERANK_PREREG.json.  Reuses the E4 order-consistency
rerank points (rv_points_rerank, exposed validation) without new teacher
queries; only x-axis rows take the reranked centres, y rows keep the DENSE
points, so both variants differ from INTERP only on x rows.  Protocol of
scripts/v4_s1_interp_eval.py (NOFACE gate, B0 EMA/resets, 6-annotator mean IoU,
video macro over both ratios, paired bootstrap).  Sanity gate: RERANKNF(hold)
must reproduce the published E4 RERANKNF numbers.
"""
import argparse,hashlib,json
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
 ap.add_argument('--d2',type=Path,required=True);ap.add_argument('--rerank',type=Path,required=True)
 ap.add_argument('--splits',default='dev2,confirm2');ap.add_argument('--prereg',default='configs/V5_P0_XRERANK_PREREG.json')
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 P=('DENSENF','RERANKNF','INTERP','XRERANK_INTERP');rows=[]
 for split in a.splits.split(','):
  for vid in SPLITS[split]:
   q2=json.loads((a.d2/f'{vid}.json').read_text());qr=json.loads((a.rerank/f'{vid}.json').read_text());keys=q2['keyframes']
   if qr['keyframes']!=keys:raise ValueError('rerank keyframes != DENSE '+vid)
   per={}
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio);comp=0 if axis==0 else 1
    p2=q2['ratios'][r]['points'];rr=qr['ratios'][r]
    if rr['points'][0] is not None and len(rr['points'])!=len(p2):raise ValueError('rerank points length '+vid)
    # P0: x rows take the rerank points, y rows the DENSE points (both then interpolated)
    pr=[list(p) if p is not None else None for p in (rr['points'] if comp==0 else p2)]
    xchg=sum(1 for x,y in zip(pr,p2) if x is not None and y is not None and abs(x[comp]-y[comp])>1e-9)
    if comp==1 and xchg:raise ValueError('y row changed by rerank '+vid)
    v={'DENSENF':run(qwen_centres,p2,keys,z,W,H,ratio,comp,gt),
       'RERANKNF':run(qwen_centres,rr['points'],keys,z,W,H,ratio,comp,gt),
       'INTERP':run(qwen_centres_interp,p2,keys,z,W,H,ratio,comp,gt),
       'XRERANK_INTERP':run(qwen_centres_interp,pr,keys,z,W,H,ratio,comp,gt)}
    L=W if comp==0 else H;reset=z['reset'].astype(bool)
    gc=((gt[...,comp]+gt[...,comp+2])/2).mean(0);motion=float(np.abs(np.diff(gc/L))[~reset[1:]].mean())
    per[r]={'axis':'x' if comp==0 else 'y','motion':'fast' if motion>MOTION_THR else 'slow','x_changed_keyframes':xchg,**{k:float(x.mean()) for k,x in v.items()}}
   rows.append({'split':split,'vid':vid,'per_ratio':per,**{p:float(np.mean([per[r][p] for r in RATIOS])) for p in P}})
 def paired(sel,p,b,key=None):
  d=np.array([np.mean([x['per_ratio'][r][p]-x['per_ratio'][r][b] for r in RATIOS if key is None or key(x['per_ratio'][r])]) for x in sel if key is None or any(key(x['per_ratio'][r]) for r in RATIOS)])
  return {'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 out={'prereg':a.prereg,'prereg_sha256':hashlib.sha256(Path(a.prereg).read_bytes()).hexdigest(),'official_f_video':None}
 for split in a.splits.split(','):
  sel=[x for x in rows if x['split']==split];res={p:float(np.mean([x[p] for x in sel])) for p in P}
  for p,b in (('XRERANK_INTERP','INTERP'),('RERANKNF','DENSENF'),('XRERANK_INTERP','DENSENF'),('INTERP','DENSENF')):
   res[f'{p}-{b}']={'all':paired(sel,p,b),'axis_x':paired(sel,p,b,lambda v:v['axis']=='x'),'axis_y':paired(sel,p,b,lambda v:v['axis']=='y'),
    'motion_fast':paired(sel,p,b,lambda v:v['motion']=='fast'),'motion_slow':paired(sel,p,b,lambda v:v['motion']=='slow')}
  out[split]=res
 out['per_video']=rows;a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 gate=True
 for split in a.splits.split(','):
  print(split,{p:round(out[split][p],4) for p in P})
  for k,v in out[split].items():
   if isinstance(v,dict):
    for kk,x in v.items():print(' ',k,kk,x['videos'],round(x['mean'],4),[round(y,4) for y in x['ci95']],x['better'],x['worse'],round(x['worst'],4))
 e4ref={'dev2':.0093,'confirm2':.0072}
 for split,ref in e4ref.items():
  got=out[split]['RERANKNF-DENSENF']['all']['mean'];ok=abs(got-ref)<.004
  out.setdefault('sanity',{})[f'E4_reproduction_{split}']={'expected_mean':ref,'got':got,'ok':ok}
  gate=gate and ok
 out['sanity_gate_pass']=gate
 (a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 print('sanity',json.dumps(out['sanity']),'gate',gate)
if __name__=='__main__':main()
