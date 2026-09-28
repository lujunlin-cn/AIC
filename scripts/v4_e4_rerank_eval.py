"""V4 E4 pilot evaluation: visual crop re-ranking vs DENSE on RetargetVid.

Prereg configs/V4_E4_CROP_RERANK_PREREG.json.  Pipeline metric as in
scripts/v4_e1_obs_eval.py (RERANKNF vs DENSENF).  Keyframe level, on the
queried keyframes: IoU (mean over 6 annotators) of the window centred on the
DENSE point, on the selected candidate, and on the best candidate (candidate
oracle: one window per frame, chosen by mean-annotator IoU).
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import frame_iou,boot
from scripts.v4_e1_obs_eval import SPLITS,run


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True)
 ap.add_argument('--dense',type=Path,required=True);ap.add_argument('--rerank',type=Path,required=True);ap.add_argument('--splits',default='dev2')
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 rows=[];kfs=[]
 for split in a.splits.split(','):
  for vid in SPLITS[split]:
   q2=json.loads((a.dense/f'{vid}.json').read_text());qr=json.loads((a.rerank/f'{vid}.json').read_text());keys=q2['keyframes'];per={}
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r);W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio);comp=0 if axis==0 else 1
    rr=qr['ratios'][r];base=run(q2['ratios'][r]['points'],keys,z,W,H,ratio,comp,gt);new=run(rr['points'],keys,z,W,H,ratio,comp,gt)
    per[r]={'axis':'x' if comp==0 else 'y','DENSENF':float(base.mean()),'RERANKNF':float(new.mean())}
    for j,(t,x) in enumerate(zip(keys,rr['rerank'])):
     if x is None:continue
     cs=np.array(x['cands']);ious=frame_iou(centre_to_offset(cs,W,H,ratio),W,H,ratio,np.repeat(gt[:,t:t+1],len(cs),1))
     sel=x['pick0'] if rr['source'][j]=='rerank_agree' else x['dense_index']
     kfs.append({'split':split,'vid':vid,'r':r,'axis':comp,'face':bool(z['chosen'][t]>=0),'n':len(cs),'src':rr['source'][j],'dense':float(ious[x['dense_index']]),'sel':float(ious[sel]),'oracle':float(ious.max()),
      'pick0':x.get('pick0'),'pick1':x.get('pick1'),'dense_index':x['dense_index'],'status':[x.get('status0'),x.get('status1')]})
   rows.append({'split':split,'vid':vid,'per_ratio':per,'DENSENF':float(np.mean([v['DENSENF'] for v in per.values()])),'RERANKNF':float(np.mean([v['RERANKNF'] for v in per.values()]))})
 def paired(sel,key=None):
  d=np.array([np.mean([v['RERANKNF']-v['DENSENF'] for v in x['per_ratio'].values() if key is None or key(v)]) for x in sel])
  return {'videos':len(d),'mean':float(d.mean()),'ci95':boot(d),'better':int((d>1e-4).sum()),'worse':int((d<-1e-4).sum()),'worst':float(d.min())}
 out={'prereg':'configs/V4_E4_CROP_RERANK_PREREG.json','official_f_video':None}
 for split in a.splits.split(','):
  sel=[x for x in rows if x['split']==split];K=[x for x in kfs if x['split']==split];A=[x for x in K if x['src']=='rerank_agree']
  both=[x for x in K if x['pick0'] is not None and x['pick1'] is not None]
  out[split]={'DENSENF':float(np.mean([x['DENSENF'] for x in sel])),'RERANKNF':float(np.mean([x['RERANKNF'] for x in sel])),
   'vs_DENSENF':paired(sel),'axis_x':paired(sel,lambda v:v['axis']=='x'),'axis_y':paired(sel,lambda v:v['axis']=='y'),
   'keyframes':{'queried':len(K),'agree':len(A),'order_agreement':len(A)/max(1,len(both)),'answered_both':len(both),
    'dense_iou':float(np.mean([x['dense'] for x in K])),'selected_iou':float(np.mean([x['sel'] for x in K])),'oracle_iou':float(np.mean([x['oracle'] for x in K])),
    'agree_dense_iou':float(np.mean([x['dense'] for x in A] or [np.nan])),'agree_selected_iou':float(np.mean([x['sel'] for x in A] or [np.nan])),
    'agree_pick_is_dense':float(np.mean([x['pick0']==x['dense_index'] for x in A] or [np.nan])),
    'first_label_bias_order0':float(np.mean([x['pick0']==0 for x in both] or [np.nan])),'first_label_bias_order1':float(np.mean([x['pick1']==x['n']-1 for x in both] or [np.nan])),
    'oracle_is_dense':float(np.mean([abs(x['oracle']-x['dense'])<1e-9 for x in K]))}}
 out['per_video']=rows
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n');(a.output/'keyframes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in kfs))
 for split in a.splits.split(','):
  o=out[split];print(split,'DENSENF',round(o['DENSENF'],4),'RERANKNF',round(o['RERANKNF'],4))
  for k in ('vs_DENSENF','axis_x','axis_y'):v=o[k];print(' ',k,v['videos'],round(v['mean'],4),[round(x,4) for x in v['ci95']],v['better'],v['worse'],round(v['worst'],4))
  print(' ',json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in o['keyframes'].items()}))
if __name__=='__main__':main()
