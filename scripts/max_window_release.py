"""Package a max-window spatial candidate on the frozen official index.

Temporal mask = every decoded frame (identical to B0); window width = largest
legal width (identical to B0); only the free-axis position changes.
  policy QWEN_POINT   : Qwen3-VL subject point per 1 s keyframe + shot starts,
                        held within shot, B0 EMA path, B0 fallback on failure.
  policy QWEN_POINT_NOFACE : as QWEN_POINT but frames where YuNet chose a face keep
                        B0's face observation (Qwen only replaces the saliency fallback).
  policy QWEN_REGION_NOFACE : T1, as QWEN_POINT_NOFACE but the Qwen reply is a
                        target-ratio-aware subject/context box mapped to a legal
                        window centre by aic.max_window_path.region_points(mode).
  policy DET_COVERAGE : window maximising prior-weighted coverage of cached COCO
                        detections + YuNet faces (margin m vs B0's window), B0 EMA path.
Checks: cached B0 == frozen B0 package, width diff 0, frame-mask diff 0,
project validator, independent checker, unzip roundtrip.
"""
import argparse,json,zipfile,tempfile
from pathlib import Path
import numpy as np
from aic.contract import load_jsonl,load_index,write_submission
from aic.max_window_path import geometry,to_crops,ema_offsets,qwen_centres,coverage_centre,region_points
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 fz=json.loads(a.frozen.read_text())
 if fz['status']!='frozen_exploratory':raise ValueError('config not frozen')
 if sha256_file(a.index)!=fz['index_sha256']:raise ValueError('index changed')
 a.output.mkdir(parents=True,exist_ok=False);index=load_index(a.index);records=load_jsonl(a.index)
 b0={r['video_id']:r for r in load_jsonl(fz['b0_predictions'])};cache=Path(fz['obs_cache']);policy=fz['policy'];qdir=Path(fz['qwen_points']) if policy.startswith('QWEN') else None
 omit_size=fz['declared_model_size_mb'] is None
 parent={r['video_id']:np.array([p['bboxes'] for p in r['predictions']]) for r in load_jsonl(fz['parent_predictions'])} if fz.get('parent_predictions') else None
 rows=[];stats=[];total_mask_diff=0;total_width_diff=0.
 for r in records:
  vid=r['video_id'];meta=index[vid];z=np.load(cache/f'{vid}_t.npz');W,H=int(z['W']),int(z['H']);ratio=r['targetRatioWH']
  ref=np.array([p['bboxes'] for p in b0[vid]['predictions']]);ref_frames=[p['frame'] for p in b0[vid]['predictions']]
  if ref_frames!=list(range(meta.frame_count)) or np.abs(z['b0']-ref).max()!=0:raise ValueError('cache/B0 mismatch '+vid)
  w,h,axis=geometry(W,H,ratio);reset=z['reset'].astype(bool)
  if axis is None:crops=ref.tolist();src=np.array(['full']*len(ref))
  else:
   comp=0 if axis==0 else 1;raw=z['raw'][:,comp];qq=None
   if policy in ('QWEN_POINT','QWEN_POINT_NOFACE','QWEN_REGION_NOFACE'):
    q=json.loads((qdir/f'{vid}.json').read_text());qq=q['ratios']['t']
    if policy=='QWEN_REGION_NOFACE':
     L,s=(W,w) if axis==0 else (H,h);pts=region_points(qq['regions'],comp,s/L,fz['region_mode'])
    else:pts=qq['points']
    c,src=qwen_centres(pts,q['keyframes'],reset,raw,comp)
    if policy in ('QWEN_POINT_NOFACE','QWEN_REGION_NOFACE'):
     face=z['chosen']>=0;c=np.where(face,raw,c);src=np.where(face,'b0_face',src)
   else:
    cs=[coverage_centre(z,t,raw[t],keep_margin=fz['keep_margin']) for t in range(len(raw))];c=np.array([x[0] for x in cs]);src=np.array([x[1] for x in cs])
   crops=to_crops(W,H,ratio,ema_offsets(c,reset,W,H,ratio,alpha=fz['alpha']))
  crops=np.asarray(crops);total_width_diff=max(total_width_diff,float(np.abs(crops[:,2]-ref[:,2]).max()))
  shift=np.abs((crops[:,0]+crops[:,1])-(ref[:,0]+ref[:,1]))/(W if axis==0 else H)
  preds=[{'frame':i,'bboxes':crops[i].tolist()} for i in range(meta.frame_count)]
  row={'video_id':vid,'targetRatioWH':ratio,'model_size_mb':fz['declared_model_size_mb'],'predictions':preds}
  if omit_size:del row['model_size_mb']
  rows.append(row)
  pchg=int((np.abs(crops-parent[vid]).max(1)>1e-9).sum()) if parent is not None else None
  stats.append({'video_id':vid,'frames':meta.frame_count,'changed_frames':int((shift>1e-9).sum()),'changed_vs_parent':pchg,'mean_shift_frac':float(shift.mean()),'p95_shift_frac':float(np.percentile(shift,95)),
   'source_rates':{k:float((src==k).mean()) for k in set(src.tolist())},'parse_fail':bool(qq is not None and qq['status'].count('ok')!=len(qq['status']))})
 dest=a.output/'predictions.jsonl';size=fz['declared_model_size_mb']
 nb=None if omit_size else int(round(size*1e6))
 validation=write_submission(dest,rows,index,stage='preliminary' if omit_size else 'final',actual_model_size_mb=size).to_dict();independent=check(a.index,dest,nb,require_size=not omit_size)
 zip_path=a.output/f"{fz['submission_id']}.zip"
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:zf.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zip_path) as zf:
   if zf.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   zf.extractall(tmp)
  roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',nb,require_size=not omit_size)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('ZIP roundtrip changed bytes')
 changed=[s for s in stats if s['changed_frames']];sh=np.concatenate([[s['mean_shift_frac']] for s in stats])
 man={'submission_id':fz['submission_id'],'status':fz['release_status'],'uploaded':False,'official_platform_score':None,'parent_baseline':fz.get('parent_baseline','B0_ALL_SELECT_YUNET_V1 (34.42)'),
  'primary_changed_factor':fz['primary_changed_factor'],'frozen_sha256':sha256_file(a.frozen),'components':fz['components'],'total_parameters':sum(c['parameters'] for c in fz['components']),
  'total_weight_bytes':sum(c['bytes'] for c in fz['components']),'declared_model_size_mb':size,'declared_size_note':fz['declared_size_note'],
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'empty_videos':[r['video_id'] for r in rows if not r['predictions']],
  'diff_vs_B0':{'frame_mask_diff':total_mask_diff,'max_width_diff_px':total_width_diff,'videos_changed':len(changed),'frames_changed':int(sum(s['changed_frames'] for s in stats)),
   'mean_shift_frac_video_mean':float(sh.mean()),'mean_shift_frac_video_p90':float(np.percentile(sh,90)),'source_rate_mean':{k:float(np.mean([s['source_rates'].get(k,0.) for s in stats])) for k in sorted({k for s in stats for k in s['source_rates']})},
   'videos_with_parse_fail':sum(s['parse_fail'] for s in stats)},
  'diff_vs_parent':None if parent is None else {'parent':fz['parent_predictions'],'videos_changed':sum(1 for s in stats if s['changed_vs_parent']),'frames_changed':int(sum(s['changed_vs_parent'] for s in stats))},'per_video':stats,
  'validator':validation,'independent':independent,'unzip_independent':roundtrip,'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zip_path),'zip_bytes':zip_path.stat().st_size}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1,allow_nan=False)+'\n');(a.output/f"{fz['submission_id']}.zip.sha256").write_text(man['zip_sha256']+f"  {fz['submission_id']}.zip\n")
 print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator','components')}))
if __name__=='__main__':main()
