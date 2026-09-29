"""V6 E0: axis-selectable H3 splice contract (generalises the x-only P0 splice
to an H3/V6 head released on either or both free axes).

Guards (hard fails), generalised from v5_p0_splice_spatial.py:
  G1  DENSE points through this exact code path reproduce the INTERP mother
      byte-for-byte on every video (proves the reimplementation);
  G2  rerank points differ from DENSE only on the free-axis component of rows
      whose axis is in --axes (x rows: x component; y rows: y component);
      every other component, including the invalid flag, must be identical;
  G3  rows whose axis is NOT in --axes, and non-spatial rows, are byte-identical
      to the mother.
Output is the all-frame spatial candidate (never uploaded); the TEMP-masked
combo is produced afterwards by scripts.mask_combo_release.py.
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
 ap.add_argument('--rerank',type=Path,required=True);ap.add_argument('--axes',required=True,help='comma list of free axes the rerank may touch, e.g. 0 / 1 / 0,1')
 ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--submission-id',required=True);ap.add_argument('--combo-config',type=Path,default=None)
 ap.add_argument('--combo-submission-id',default=None);ap.add_argument('--changed-factor',default='')
 ap.add_argument('--only',default=None,help='comma list of video_ids; diagnostic E2E slices only, never a release')
 a=ap.parse_args();axes={int(x) for x in a.axes.split(',')}
 only=set(a.only.split(',')) if a.only else None
 if not axes<= {0,1} or not axes:raise ValueError('--axes must be a nonempty subset of {0,1}')
 fz=json.loads(a.mother_config.read_text())
 if fz['status']!='frozen_exploratory':raise ValueError('mother config not frozen')
 if fz.get('centre_mode','hold')!='interp' or fz['policy']!='QWEN_POINT_NOFACE':raise ValueError('mother is not the INTERP/NOFACE protocol')
 if sha256_file(a.index)!=fz['index_sha256']:raise ValueError('index changed')
 index=load_index(a.index);records=load_jsonl(a.index)
 if only is not None:
  records=[r for r in records if r['video_id'] in only]
  # diagnostic slice: validate against the sliced index, never against the full 174
  idx_diag=a.output/'index_diag.jsonl';idx_diag.parent.mkdir(parents=True,exist_ok=True)
  idx_diag.write_text(''.join(json.dumps(r)+'\n' for r in records))
  a=argparse.Namespace(**{**vars(a),'index':idx_diag})
 audit_path=a.rerank/'_domain_audit.json';head_label=json.loads(audit_path.read_text())['kind'] if audit_path.exists() else 'unknown'
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
   if comp in axes:
    qr=json.loads((a.rerank/f'{vid}.json').read_text())
    if qr['keyframes']!=keys:raise ValueError('rerank keyframes != DENSE '+vid)
    if qr.get('meta',{}).get('comp',comp)!=comp:raise ValueError('rerank meta comp mismatch '+vid)
    pr=qr['ratios']['t']['points']
    for pd,pp in zip(q2['ratios']['t']['points'],pr):
     if (pd is None)!=(pp is None) or (pp is not None and (abs(pp[1-comp]-pd[1-comp])>1e-9 or pp[2]!=pd[2])):raise ValueError(f'G2 rerank moved component outside axis {comp} '+vid)
    crops=pipeline(z,W,H,ratio,comp,pr,keys,alpha)
  crops=np.asarray(crops);width_diff=max(width_diff,float(np.abs(crops[:,2]-ref[:,2]).max()))
  pchg=int((np.abs(crops-parent[vid]).max(1)>1e-9).sum()) if parent is not None else None
  preds=[{'frame':i,'bboxes':crops[i].tolist()} for i in range(meta.frame_count)]
  rows.append({'video_id':vid,'targetRatioWH':ratio,'predictions':preds})
  stats.append({'video_id':vid,'axis':'none' if axis is None else ('x' if axis==0 else 'y'),'in_scope':(axis is not None and (0 if axis==0 else 1) in axes),'frames':meta.frame_count,'changed_vs_parent':pchg})
 if parent is not None:
  bad=sum(1 for s in stats if not s['in_scope'] and s['changed_vs_parent']!=0)
  if bad:raise ValueError(f'G3 rows outside axes {sorted(axes)} changed vs parent: {bad} videos')
 dest=a.output/'predictions.jsonl'
 validation=write_submission(dest,rows,load_index(a.index),stage='preliminary',actual_model_size_mb=None).to_dict();independent=check(a.index,dest,None,require_size=False)
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
  'diagnostic_only':only is not None,'diagnostic_videos':sorted(only) if only else None,
  'mother':'QWEN32B_NOFACE_INTERP_V4_SPATIAL (official 48.67 as QWEN32B_DT_INTERP_V4_FINAL after TEMP mask)','mother_config_sha256':sha256_file(a.mother_config),
  'parent_predictions':str(a.parent),'parent_predictions_sha256':sha256_file(a.parent),
  'rerank_dir':str(a.rerank),'rerank_head':head_label,'axes':sorted(axes),
  'primary_changed_factor':a.changed_factor or f'H3 head {head_label} rerank applied to axes {sorted(axes)} through the INTERP pipeline; all other rows byte-identical to the mother',
  'index_sha256':fz['index_sha256'],'components':fz['components'],'total_parameters':sum(c['parameters'] for c in fz['components']),
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),
  'guards':{'G1_dense_recompute_equals_parent':True,'G2_rerank_axis_components_only':True,'G3_non_target_rows_unchanged':True,'axes':sorted(axes),'max_width_diff_px':width_diff},
  'diff_vs_parent':{'videos_changed':sum(1 for s in stats if s['changed_vs_parent']),'frames_changed':int(chg)},
  'per_video':stats,'validator':validation,'independent':independent,'unzip_independent':rt,
  'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zp),'zip_bytes':zp.stat().st_size}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1,allow_nan=False)+'\n')
 (a.output/f'{a.submission_id}.zip.sha256').write_text(man['zip_sha256']+f'  {a.submission_id}.zip\n')
 if a.combo_config:
  mc=json.loads(Path('configs/QWEN32B_DT_INTERP_V4.json').read_text())
  mc['submission_id']=a.combo_submission_id or a.submission_id
  mc['release_status']='READY_TO_UPLOAD_EXPLORATORY_INTERNAL_TEACHER_COMBO'
  mc['status']='frozen_exploratory'
  mc['primary_changed_factor']=man['primary_changed_factor']
  mc['parent_baseline']='INTERP = QWEN32B_DT_INTERP_V4_FINAL (official 48.67, zip 501cebce...)'
  mc['spatial_parent']={'name':a.submission_id,'platform_name':None,'official_platform_score':None,'zip_sha256':man['zip_sha256'],
   'predictions':str(dest),'predictions_sha256':man['predictions_sha256']}
  a.combo_config.parent.mkdir(parents=True,exist_ok=True)
  a.combo_config.write_text(json.dumps(mc,indent=1)+'\n')
 print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator','components')}))
if __name__=='__main__':main()
