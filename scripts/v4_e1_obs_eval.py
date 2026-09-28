"""V4 E1: do 0.25 s spatial observations help on top of DENSE (0.5 s)?

Prereg configs/V4_E1_OBS025_PREREG.json.  RetargetVid 6-annotator crops, the
protocol of scripts/teacher_diag_eval.py (rounded inclusive boxes, max legal
window, NOFACE gate, hold within shot, B0 EMA alpha .25 / resets).
  DENSENF  0.5 s replies (parent, = DT_V3 spatial recipe)
  OBS025   all 0.25 s keyframes (0.5 s replies reused byte-identically)
  ADAPT    0.5 s keyframes + 0.25 s keyframes chosen by adapt_keys()
Also: held (0.5 s) vs fresh (0.25 s) IoU at the added keyframes, no EMA.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset,ema_offsets,qwen_centres
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import frame_iou,boot

SPLITS={'dev2':[f'{i:03d}' for i in range(31,101)],'confirm2':[str(i) for i in range(601,701)]}
TAU=.10
MOTION_THR=.0014829153001391375


def adapt_keys(k2,p2,k4,reset,comp,movable):
 """0.25 s keyframes kept by the prereg rule; movable = (L-s)/L (0 -> none)."""
 if movable<=1e-9:return []
 shot=np.cumsum(reset);s2=set(k2);keep=[]
 for t in k4:
  if t in s2:continue
  j=max(i for i,k in enumerate(k2) if k<t);prev=p2[j];nxt=j+1 if j+1<len(k2) else None
  if shot[k2[j]]!=shot[t]:continue
  if prev is None or prev[2]:keep.append(t);continue
  if nxt is None or shot[k2[nxt]]!=shot[t]:continue
  q=p2[nxt]
  if q is None or q[2]:continue
  if abs(prev[comp]-q[comp])/movable>TAU:keep.append(t)
 return keep


def run(pts,keys,z,W,H,ratio,comp,gt):
 reset=z['reset'].astype(bool);raw=z['raw'][:,comp];face=z['chosen']>=0
 c,_=qwen_centres(pts,keys,reset,raw,comp);c=np.where(face,raw,c)
 return frame_iou(ema_offsets(c,reset,W,H,ratio),W,H,ratio,gt)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True)
 ap.add_argument('--d2',type=Path,required=True);ap.add_argument('--d4',type=Path,required=True);ap.add_argument('--splits',default='dev2')
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rows=[];props=[];reuse_bad=0
 for split in a.splits.split(','):
  for vid in SPLITS[split]:
   q2=json.loads((a.d2/f'{vid}.json').read_text());q4=json.loads((a.d4/f'{vid}.json').read_text())
   if not set(q2['keyframes'])<=set(q4['keyframes']):raise ValueError('not superset '+vid)
   per={}
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio)
    if gt.shape[1]!=len(z['b0']):raise ValueError('frames '+vid)
    comp=0 if axis==0 else 1;L,s=(W,w) if comp==0 else (H,h);reset=z['reset'].astype(bool)
    k2,p2=q2['keyframes'],q2['ratios'][r]['points'];k4,p4=q4['keyframes'],q4['ratios'][r]['points']
    raw4=dict(zip(k4,q4['ratios'][r]['raw']));reuse_bad+=sum(raw4[k]!=t for k,t in zip(k2,q2['ratios'][r]['raw']))
    ak=set(adapt_keys(k2,p2,k4,reset,comp,(L-s)/L));ka=[k for k in k4 if k in set(k2) or k in ak];pa=[p for k,p in zip(k4,p4) if k in set(ka)]
    base=run(p2,k2,z,W,H,ratio,comp,gt);full=run(p4,k4,z,W,H,ratio,comp,gt);ad=run(pa,ka,z,W,H,ratio,comp,gt)
    gl,gh=gt[...,comp],gt[...,comp+2];gc=((gl+gh)/2).mean(0);rel=np.clip((gc-s/2)/max(L-s,1e-9),0,1)
    motion=float(np.abs(np.diff(gc/L))[~reset[1:]].mean()) if len(gc)>1 else 0.
    per[r]={'axis':'x' if comp==0 else 'y','DENSENF':float(base.mean()),'OBS025':float(full.mean()),'ADAPT':float(ad.mean()),
     'added_obs025':len(k4)-len(k2),'added_adapt':len(ak),'motion':'fast' if motion>MOTION_THR else 'slow',
     'edge':bool(((rel<.1)|(rel>.9)).mean()>=.5),'face_rate':float((z['chosen']>=0).mean())}
    # held vs fresh at added keyframes (same shot, both valid), window centred on the point, no EMA
    shot=np.cumsum(reset)
    for t,p in zip(k4,p4):
     if t in set(k2):continue
     j=max(i for i,k in enumerate(k2) if k<t);prev=p2[j]
     if shot[k2[j]]!=shot[t] or prev is None or prev[2] or p is None or p[2]:continue
     oh,of=centre_to_offset(np.array([prev[comp],p[comp]]),W,H,ratio);ih,i_f=frame_iou(np.array([oh,of]),W,H,ratio,np.repeat(gt[:,t:t+1],2,1))
     props.append({'split':split,'vid':vid,'r':r,'face':bool(z['chosen'][t]>=0),'held':float(ih),'fresh':float(i_f),'in_adapt':t in ak})
   rows.append({'split':split,'vid':vid,'per_ratio':per,**{p:float(np.mean([per[r][p] for r in RATIOS])) for p in ('DENSENF','OBS025','ADAPT')}})
 def paired(sel,p,key=None):
  if key is None:d=np.array([x[p]-x['DENSENF'] for x in sel])
  else:d=np.array([np.mean([v[p]-v['DENSENF'] for v in x['per_ratio'].values() if key(v)]) for x in sel if any(key(v) for v in x['per_ratio'].values())])
  if not len(d):return None
  return {'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 out={'prereg':'configs/V4_E1_OBS025_PREREG.json','tau':TAU,'reuse_raw_mismatch':reuse_bad,'official_f_video':None}
 for split in a.splits.split(','):
  sel=[x for x in rows if x['split']==split];res={'DENSENF_iou':float(np.mean([x['DENSENF'] for x in sel]))}
  for p in ('OBS025','ADAPT'):
   res[p]={'iou':float(np.mean([x[p] for x in sel])),'vs_DENSENF':paired(sel,p),
    'axis_x':paired(sel,p,lambda v:v['axis']=='x'),'axis_y':paired(sel,p,lambda v:v['axis']=='y'),
    'motion_fast':paired(sel,p,lambda v:v['motion']=='fast'),'motion_slow':paired(sel,p,lambda v:v['motion']=='slow'),
    'edge':paired(sel,p,lambda v:v['edge']),'not_edge':paired(sel,p,lambda v:not v['edge']),
    'face0':paired(sel,p,lambda v:v['face_rate']==0),'face_any':paired(sel,p,lambda v:v['face_rate']>0),
    'added_queries':int(sum(v[f'added_{p.lower()}'] for x in sel for v in x['per_ratio'].values()))}
   res[p]['gain_per_1k_added']=res[p]['vs_DENSENF']['mean']*1000/max(1,res[p]['added_queries']/len(sel))
  ps=[x for x in props if x['split']==split]
  res['held_vs_fresh']={'n':len(ps),'held':float(np.mean([x['held'] for x in ps])),'fresh':float(np.mean([x['fresh'] for x in ps])),
   'noface_n':sum(not x['face'] for x in ps),'noface_fresh_minus_held':float(np.mean([x['fresh']-x['held'] for x in ps if not x['face']])),
   'adapt_n':sum(x['in_adapt'] for x in ps),'adapt_fresh_minus_held':float(np.mean([x['fresh']-x['held'] for x in ps if x['in_adapt']] or [np.nan]))}
  out[split]=res
 out['per_video']=rows
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 def rnd(v):return {k:(round(x,4) if isinstance(x,float) else ([round(y,4) for y in x] if isinstance(x,list) else x)) for k,x in v.items()} if isinstance(v,dict) else v
 print(json.dumps({s:{k:(rnd(v) if not isinstance(v,dict) or 'iou' not in v else {kk:rnd(vv) for kk,vv in v.items()}) for k,v in out[s].items()} for s in a.splits.split(',')},indent=1));print('reuse_raw_mismatch',reuse_bad)
if __name__=='__main__':main()
