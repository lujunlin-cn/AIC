"""V4 E2: context subject point vs single-frame DENSE point on RetargetVid.

Prereg configs/V4_E2_CONTEXT_PREREG.json.  Same protocol as scripts/v4_e1_obs_eval.py:
DENSENF (0.5 s single-frame points) vs CTXNF (same keyframes; replies from
scripts/qwen_context_point.py where it queried, DENSE otherwise), NOFACE gate,
hold within shot, B0 EMA / resets, max legal window, 6 annotators, per-video
mean over both ratios, video macro, bootstrap.
Keyframe decomposition on the queried keyframes (teacher_diag_eval classes):
subject_miss = point outside the GT window of >= half the annotators.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import frame_iou,boot
from scripts.v4_e1_obs_eval import SPLITS,MOTION_THR,run


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True)
 ap.add_argument('--dense',type=Path,required=True);ap.add_argument('--context',type=Path,required=True);ap.add_argument('--splits',default='dev2')
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rows=[];kfs=[];cnt={'queried':0,'context_ok':0,'fallback':0,'status':{}}
 for split in a.splits.split(','):
  for vid in SPLITS[split]:
   q2=json.loads((a.dense/f'{vid}.json').read_text());qc=json.loads((a.context/f'{vid}.json').read_text())
   if qc['keyframes']!=q2['keyframes']:raise ValueError('keyframes differ '+vid)
   keys=q2['keyframes'];per={}
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio)
    comp=0 if axis==0 else 1;L,s=(W,w) if comp==0 else (H,h);reset=z['reset'].astype(bool)
    c2=qc['ratios'][r];p2=q2['ratios'][r]['points'];pc=c2['points']
    for j,src in enumerate(c2['source']):
     if src in ('dense_unconsumed','dense_no_context'):
      if c2['raw'][j]!=q2['ratios'][r]['raw'][j] or pc[j]!=p2[j]:raise ValueError('unqueried keyframe changed '+vid)
    base=run(p2,keys,z,W,H,ratio,comp,gt);ctx=run(pc,keys,z,W,H,ratio,comp,gt)
    gl,gh=gt[...,comp],gt[...,comp+2];gc=((gl+gh)/2).mean(0);motion=float(np.abs(np.diff(gc/L))[~reset[1:]].mean())
    det=z['det'];persons=np.median([int(((det[z['det_off'][t]:z['det_off'][t+1],4]>=.5)&(det[z['det_off'][t]:z['det_off'][t+1],5]==1)).sum()) for t in range(len(reset))])
    per[r]={'axis':'x' if comp==0 else 'y','DENSENF':float(base.mean()),'CTXNF':float(ctx.mean()),'motion':'fast' if motion>MOTION_THR else 'slow',
     'persons':'0' if persons<.5 else ('1' if persons<1.5 else '2+'),'face_rate':float((z['chosen']>=0).mean())}
    for j,(t,src) in enumerate(zip(keys,c2['source'])):
     if src not in ('context','dense_fallback'):continue
     cnt['queried']+=1;cnt['context_ok']+=src=='context';cnt['fallback']+=src=='dense_fallback';cnt['status'][c2['status'][j]]=cnt['status'].get(c2['status'][j],0)+1
     row={'split':split,'vid':vid,'r':r,'frame':t,'face':bool(z['chosen'][t]>=0),'src':src}
     for tag,p in (('dense',p2[j]),('ctx',pc[j])):
      if p is None or p[2]:row[tag]=None;continue
      px=p[comp]*L;inside=float(((gl[:,t]<=px)&(px<=gh[:,t])).mean());o=centre_to_offset(np.array([p[comp]]),W,H,ratio)
      row[tag]={'inside':inside,'miss':inside<.5,'iou':float(frame_iou(o,W,H,ratio,gt[:,t:t+1])[0])}
     if row['dense'] and row['ctx']:row['shift_win']=abs(pc[j][comp]-p2[j][comp])*L/s
     kfs.append(row)
   rows.append({'split':split,'vid':vid,'per_ratio':per,'DENSENF':float(np.mean([v['DENSENF'] for v in per.values()])),'CTXNF':float(np.mean([v['CTXNF'] for v in per.values()]))})
 def paired(sel,key=None):
  if key is None:d=np.array([x['CTXNF']-x['DENSENF'] for x in sel])
  else:d=np.array([np.mean([v['CTXNF']-v['DENSENF'] for v in x['per_ratio'].values() if key(v)]) for x in sel if any(key(v) for v in x['per_ratio'].values())])
  if not len(d):return None
  return {'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 out={'prereg':'configs/V4_E2_CONTEXT_PREREG.json','counts':cnt,'official_f_video':None}
 for split in a.splits.split(','):
  sel=[x for x in rows if x['split']==split];ks=[x for x in kfs if x['split']==split and x['dense'] and x['ctx']]
  both=lambda f:[x for x in ks if f(x)]
  out[split]={'DENSENF':float(np.mean([x['DENSENF'] for x in sel])),'CTXNF':float(np.mean([x['CTXNF'] for x in sel])),'vs_DENSENF':paired(sel),
   **{k:paired(sel,f) for k,f in (('axis_x',lambda v:v['axis']=='x'),('axis_y',lambda v:v['axis']=='y'),('motion_fast',lambda v:v['motion']=='fast'),('motion_slow',lambda v:v['motion']=='slow'),
    ('persons0',lambda v:v['persons']=='0'),('persons1',lambda v:v['persons']=='1'),('persons2p',lambda v:v['persons']=='2+'))},
   'keyframes':{'n':len(ks),'dense_miss':float(np.mean([x['dense']['miss'] for x in ks])),'ctx_miss':float(np.mean([x['ctx']['miss'] for x in ks])),
    'miss_fixed':len(both(lambda x:x['dense']['miss'] and not x['ctx']['miss'])),'miss_new':len(both(lambda x:x['ctx']['miss'] and not x['dense']['miss'])),
    'dense_iou':float(np.mean([x['dense']['iou'] for x in ks])),'ctx_iou':float(np.mean([x['ctx']['iou'] for x in ks])),
    'shift_win_median':float(np.median([x['shift_win'] for x in ks])),'moved_gt_quarter_window':float(np.mean([x['shift_win']>.25 for x in ks])),
    'noface_n':len(both(lambda x:not x['face'])),'noface_iou_gain':float(np.mean([x['ctx']['iou']-x['dense']['iou'] for x in ks if not x['face']]))}}
 out['per_video']=rows
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n');(a.output/'keyframes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in kfs))
 for split in a.splits.split(','):
  o=out[split];print(split,'DENSENF',round(o['DENSENF'],4),'CTXNF',round(o['CTXNF'],4))
  for k,v in o.items():
   if isinstance(v,dict) and 'ci95' in v:print(' ',k,v['videos'],round(v['mean'],4),[round(x,4) for x in v['ci95']],v['better'],v['worse'],round(v['worst'],4))
  print(' ',json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in o['keyframes'].items()}))
 print(json.dumps(cnt))
if __name__=='__main__':main()
