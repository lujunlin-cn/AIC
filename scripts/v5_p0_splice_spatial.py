"""V5 P0 official spatial release: x-axis rows take the rerank points through the
INTERP pipeline; everything else stays byte-identical to the INTERP mother.

Identity guards (hard fails):
  G1  the DENSE points pushed through this exact code path reproduce the INTERP
      spatial parent byte-for-byte on every video (proves the reimplementation);
  G2  rerank points may only differ from DENSE on the x component of x-axis rows;
  G3  y-axis and non-spatial rows equal the parent exactly (rerank is x-only).
Output is the all-frame spatial candidate (never uploaded); the TEMP-masked combo
is produced afterwards by scripts.mask_combo_release.py from the generated frozen
config (--combo-config).
"""
import argparse,json,zipfile,tempfile
from pathlib import Path
import numpy as np
from aic.contract import load_jsonl,load_index,write_submission
from aic.max_window_path import geometry,to_crops,ema_offsets,qwen_centres_interp
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file


def pipeline(z,W,H,ratio,comp,pts,keys,alpha=.25):
 reset=z['reset'].astype(bool);raw=z['raw'][:,comp]
 c,_=qwen_centres_interp(pts,keys,reset,raw,comp)
 c=np.where(z['chosen']>=0,raw,c)
 return np.asarray(to_crops(W,H,ratio,ema_offsets(c,reset,W,H,ratio,alpha=alpha)))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True)
 ap.add_argument('--mother-config',type=Path,default=Path('configs/QWEN32B_NOFACE_INTERP_V4_SPATIAL.json'))
 ap.add_argument('--parent',type=Path,default=Path('/data/aic/official_test_20260926/submissions/QWEN32B_NOFACE_INTERP_V4_SPATIAL/predictions.jsonl'))
 ap.add_argument('--rerank',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--submission-id',default='QWEN32B_INTERP_XRERANK_V5_SPATIAL');ap.add_argument('--combo-config',type=Path,default=None)
 ap.add_argument('--combo-submission-id',default='QWEN32B_INTERP_XRERANK_V5_FINAL');a=ap.parse_args()
 fz=json.loads(a.mother_config.read_text())
 if fz['status']!='frozen_exploratory':raise ValueError('mother config not frozen')
 if fz.get('centre_mode','hold')!='interp' or fz['policy']!='QWEN_POINT_NOFACE':raise ValueError('mother is not the INTERP/NOFACE protocol')
 if sha256_file(a.index)!=fz['index_sha256']:raise ValueError('index changed')
 index=load_index(a.index);records=load_jsonl(a.index)
 b0={r['video_id']:r for r in load_jsonl(fz['b0_predictions'])};cache=Path(fz['obs_cache'])
 dense=Path(fz['qwen_points']);parent={r['video_id']:np.array([p['bboxes'] for p in r['predictions']]) for r in load_jsonl(a.parent)}
 alpha=fz['alpha'];rows=[];stats=[];width_diff=0.
 for r in records:
  vid=r['video_id'];meta=index[vid];z=np.load(cache/f'{vid}_t.npz');W,H=int(z['W']),int(z['H']);ratio=r['targetRatioWH']
  ref=np.array([p['bboxes'] for p in b0[vid]['predictions']])
  if [p['frame'] for p in b0[vid]['predictions']]!=list(range(meta.frame_count)) or np.abs(z['b0']-ref).max()!=0:raise ValueError('cache/B0 mismatch '+vid)
  w,h,axis=geometry(W,H,ratio);q2=json.loads((dense/f'{vid}.json').read_text());keys=q2['keyframes']
  if axis is None:
   crops=ref
  else:
   comp=0 if axis==0 else 1
   pts=[None if p is None else list(p) for p in q2['ratios']['t']['points']]
   crops=pipeline(z,W,H,ratio,comp,pts,keys,alpha)
   if parent is not None and (crops!=parent[vid]).any():raise ValueError('G1 dense-recompute != INTERP parent '+vid)
   if comp==0:
    qr=json.loads((a.rerank/f'{vid}.json').read_text())
    if qr['keyframes']!=keys:raise ValueError('rerank keyframes != DENSE '+vid)
    pr=qr['ratios']['t']['points']
    for pd,pp in zip(q2['ratios']['t']['points'],pr):
     if (pd is None)!=(pp is None) or (pp is not None and (abs(pp[1]-pd[1])>1e-9 or pp[2]!=pd[2])):raise ValueError('G2 rerank moved non-x component '+vid)
    crops=pipeline(z,W,H,ratio,comp,pr,keys,alpha)
  crops=np.asarray(crops);width_diff=max(width_diff,float(np.abs(crops[:,2]-ref[:,2]).max()))
  pchg=int((np.abs(crops-parent[vid]).max(1)>1e-9).sum()) if parent is not None else None
  preds=[{'frame':i,'bboxes':crops[i].tolist()} for i in range(meta.frame_count)]
  rows.append({'video_id':vid,'targetRatioWH':ratio,'predictions':preds})
  stats.append({'video_id':vid,'axis':'none' if axis is None else ('x' if axis==0 else 'y'),'frames':meta.frame_count,'changed_vs_parent':pchg})
 if parent is not None:
  bad=sum(1 for s in stats if s['axis']!='x' and s['changed_vs_parent']!=0)
  if bad:raise ValueError(f'G3 non-x rows changed vs parent: {bad} videos')
 dest=a.output/'predictions.jsonl'
 validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=None).to_dict();independent=check(a.index,dest,None,require_size=False)
 with tempfile.TemporaryDirectory() as tmp:
  zp=a.output/f'{a.submission_id}.zip'
  with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as zf:zf.write(dest,'predictions.jsonl')
  with zipfile.ZipFile(zp) as zf:
   if zf.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   zf.extractall(tmp)
  rt=check(a.index,Path(tmp)/'predictions.jsonl',None,require_size=False)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('roundtrip')
 out={r['video_id']:np.array([p['bboxes'] for p in r['predictions']]) for r in load_jsonl(dest)}
 for r in rows:
  if (out[r['video_id']]!=np.array([p['bboxes'] for p in r['predictions']])).any():raise ValueError('rewrite mismatch '+r['video_id'])
 chg=sum(s['changed_vs_parent'] for s in stats if s['changed_vs_parent'])
 man={'submission_id':a.submission_id,'status':'INTERMEDIATE_SPATIAL_ALL_FRAMES_NOT_FOR_UPLOAD','uploaded':False,'official_platform_score':None,
  'mother':'QWEN32B_NOFACE_INTERP_V4_SPATIAL (official 48.67 as QWEN32B_DT_INTERP_V4_FINAL after TEMP mask)','mother_config_sha256':sha256_file(a.mother_config),
  'parent_predictions':str(a.parent),'parent_predictions_sha256':sha256_file(a.parent),
  'rerank_dir':str(a.rerank),'primary_changed_factor':'P0: x-axis rows replace the DENSE free-axis point with the E4 order-consistency rerank choice before INTERP propagation; y rows and non-spatial rows byte-identical to the mother',
  'index_sha256':fz['index_sha256'],'components':fz['components'],'total_parameters':sum(c['parameters'] for c in fz['components']),
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),
  'guards':{'G1_dense_recompute_equals_parent':True,'G2_rerank_x_only':True,'G3_nonx_rows_unchanged':True,'max_width_diff_px':width_diff},
  'diff_vs_parent':{'videos_changed':sum(1 for s in stats if s['changed_vs_parent']),'frames_changed':int(chg)},
  'per_video':stats,'validator':validation,'independent':independent,'unzip_independent':rt,
  'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zp),'zip_bytes':zp.stat().st_size}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1,allow_nan=False)+'\n')
 (a.output/f'{a.submission_id}.zip.sha256').write_text(man['zip_sha256']+f'  {a.submission_id}.zip\n')
 if a.combo_config:
  mc=json.loads(Path('configs/QWEN32B_DT_INTERP_V4.json').read_text())
  mc['submission_id']=a.combo_submission_id
  mc['release_status']='READY_TO_UPLOAD_EXPLORATORY_INTERNAL_TEACHER_COMBO'
  mc['status']='frozen_exploratory'
  mc['primary_changed_factor']='P0 vs INTERP_V4: x-axis rows take the rerank points through the same INTERP pipeline (dev2 x +1.87 CI[+0.73,+3.23], confirm2 x +1.88 CI[+0.79,+3.26], all-video +0.93/+0.94, y exactly unchanged); TEMP mask, keys and everything else identical to INTERP'
  mc['parent_baseline']='INTERP = QWEN32B_DT_INTERP_V4_FINAL (official 48.67, zip 501cebce...)'
  mc['spatial_parent']={'name':a.submission_id,'platform_name':None,'official_platform_score':None,'zip_sha256':man['zip_sha256'],
   'predictions':str(dest),'predictions_sha256':man['predictions_sha256']}
  a.combo_config.parent.mkdir(parents=True,exist_ok=True)
  a.combo_config.write_text(json.dumps(mc,indent=1)+'\n')
 print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator','components')}))
if __name__=='__main__':main()
