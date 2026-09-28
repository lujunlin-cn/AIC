"""T5 spatial stage: N0 all-frame predictions with the H1 point->window calibration.

N0 recipe (QWEN_POINT_NOFACE: 1 s Qwen points + shot starts, hold within shot,
face frames keep YuNet raw, B0 EMA alpha .25 / resets, largest window); the
only change is every valid keyframe point p on the free axis becoming
clip(a*p + (1-a)*.5 + b*win, 0, 1) with (a,b) per axis from the frozen T5
probe.  Guard: with (a,b)=(1,0) the output must equal the scored N0
predictions for every frame (checked on all 174 videos before writing).
The temporal mask is applied afterwards by scripts.mask_combo_release.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.contract import load_jsonl,load_index,write_submission
from aic.max_window_path import geometry,to_crops,ema_offsets,qwen_centres
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file


def calibrate(points,comp,win,ab):
 a,b=ab;out=[]
 for p in points:
  if p is None or p[2]:out.append(p);continue
  q=list(p);q[comp]=float(np.clip(a*p[comp]+(1-a)*.5+b*win,0,1));out.append(q)
 return out


def video_crops(z,q,ratio,ab):
 W,H=int(z['W']),int(z['H']);w,h,axis=geometry(W,H,ratio)
 if axis is None:return np.asarray(z['b0']),'full'
 comp=0 if axis==0 else 1;L,s=(W,w) if axis==0 else (H,h);raw=z['raw'][:,comp];reset=z['reset'].astype(bool)
 pts=calibrate(q['ratios']['t']['points'],comp,s/L,ab[str(comp)])
 c,_=qwen_centres(pts,q['keyframes'],reset,raw,comp);c=np.where(z['chosen']>=0,raw,c)
 return np.asarray(to_crops(W,H,ratio,ema_offsets(c,reset,W,H,ratio,alpha=.25))),axis


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 fz=json.loads(a.frozen.read_text())
 if fz['status']!='frozen_exploratory' or sha256_file(a.index)!=fz['index_sha256']:raise ValueError('config not frozen / index changed')
 if sha256_file(fz['probe_metrics'])!=fz['probe_metrics_sha256']:raise ValueError('probe metrics changed')
 if sha256_file(fz['n0_predictions'])!=fz['n0_predictions_sha256']:raise ValueError('N0 changed')
 ab=json.loads(Path(fz['probe_metrics']).read_text())['H1'];ident={'0':(1.,0.),'1':(1.,0.)}
 a.output.mkdir(parents=True,exist_ok=False);index=load_index(a.index);n0={r['video_id']:r for r in load_jsonl(fz['n0_predictions'])}
 cache=Path(fz['obs_cache']);qdir=Path(fz['qwen_points']);rows=[];stats=[];id_bad=0
 for r in load_jsonl(a.index):
  vid=r['video_id'];z=np.load(cache/f'{vid}_t.npz');q=json.loads((qdir/f'{vid}.json').read_text());n=index[vid].frame_count
  ref=np.array([p['bboxes'] for p in n0[vid]['predictions']])
  if [p['frame'] for p in n0[vid]['predictions']]!=list(range(n)):raise ValueError('N0 not all-select '+vid)
  base,axis=video_crops(z,q,r['targetRatioWH'],ident);id_bad+=int((np.abs(base-ref).max(1)>0).sum())
  crops,_=video_crops(z,q,r['targetRatioWH'],ab)
  if np.abs(crops[:,2]-ref[:,2]).max()>0:raise ValueError('width changed '+vid)
  chg=np.abs(crops-ref).max(1)>1e-9;L=int(z['W']) if axis==0 else int(z['H'])
  shift=np.abs((crops[:,0]+crops[:,1])-(ref[:,0]+ref[:,1]))/L if axis!='full' else np.zeros(n)
  row={k:v for k,v in n0[vid].items() if k!='predictions'};row['predictions']=[{'frame':i,'bboxes':crops[i].tolist()} for i in range(n)];rows.append(row)
  stats.append({'video_id':vid,'axis':None if axis=='full' else int(axis),'frames':n,'changed_vs_n0':int(chg.sum()),'mean_shift_frac':float(shift.mean()),'p95_shift_frac':float(np.percentile(shift,95))})
 if id_bad:raise ValueError(f'identity replay != N0 on {id_bad} frames')
 dest=a.output/'predictions.jsonl';validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=None).to_dict();independent=check(a.index,dest,None,require_size=False)
 man={'stage':'spatial_all_frames','h1':ab,'identity_replay_frames_bad':id_bad,'videos':len(rows),'predictions':sum(len(r['predictions']) for r in rows),
  'changed_vs_n0':{'videos':sum(1 for s in stats if s['changed_vs_n0']),'frames':int(sum(s['changed_vs_n0'] for s in stats)),
   'mean_shift_frac_by_axis':{str(k):float(np.mean([s['mean_shift_frac'] for s in stats if s['axis']==k])) for k in (0,1)}},
  'per_video':stats,'validator':validation,'independent':independent,'predictions_sha256':sha256_file(dest),'frozen_sha256':sha256_file(a.frozen)}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1)+'\n');print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator')}))
if __name__=='__main__':main()
