"""V3 combo package: spatial parent bboxes filtered by a scored temporal mask.

The mask is recomputed from the frozen removal replies with
scripts.qwen_temporal_removal.keep_frames and must equal the frame set of the
scored mask parent (TEMP).  Every kept frame carries the spatial parent's
bbox byte-for-byte; no re-smoothing, interpolation or shot logic runs on the
shorter sequence.  Checks: parent SHAs, keys(out)==keys(mask parent),
bbox(out)==bbox(spatial parent), 174 rows, validator, independent checker,
unzip roundtrip.
"""
import argparse,json,zipfile,tempfile
from pathlib import Path
import numpy as np
from aic.contract import load_jsonl,load_index,write_submission
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file
from scripts.qwen_temporal_removal import keep_frames


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 fz=json.loads(a.frozen.read_text())
 if fz['status']!='frozen_exploratory' or sha256_file(a.index)!=fz['index_sha256']:raise ValueError('config not frozen / index changed')
 for k in ('spatial_parent','mask_parent','mask_parent_spatial_source'):
  if sha256_file(fz[k]['predictions'])!=fz[k]['predictions_sha256']:raise ValueError(k+' changed')
 a.output.mkdir(parents=True,exist_ok=False);index=load_index(a.index)
 sp,mp,ms=({r['video_id']:r for r in load_jsonl(fz[k]['predictions'])} for k in ('spatial_parent','mask_parent','mask_parent_spatial_source'))
 rem=Path(fz['removal']);anc=Path(fz['anchors']);rows=[];stats=[]
 for r in load_jsonl(a.index):
  vid=r['video_id'];n=index[vid].frame_count;s=sp[vid]
  if [x['frame'] for x in s['predictions']]!=list(range(n)):raise ValueError('spatial parent not all-select '+vid)
  meta=json.loads((anc/vid/'meta.json').read_text())
  if meta['n_frames']!=n:raise ValueError('frame count '+vid)
  kept,guard=keep_frames(meta['frame_cell'],json.loads((rem/f'{vid}.json').read_text())['removed_cells'])
  mask=[x['frame'] for x in mp[vid]['predictions']]
  if kept.tolist()!=mask:raise ValueError('recomputed mask != mask parent '+vid)
  preds=[s['predictions'][i] for i in mask]
  row={k:v for k,v in s.items() if k!='predictions'};row['predictions']=preds;rows.append(row)
  mb={x['frame']:x['bboxes'] for x in mp[vid]['predictions']};nb=[x['bboxes'] for x in ms[vid]['predictions']]
  if any(nb[f]!=b for f,b in mb.items()):raise ValueError('mask parent bbox != its spatial source '+vid)
  chg=np.array([s['predictions'][i]['bboxes']!=nb[i] for i in range(n)]);keep=np.zeros(n,bool);keep[mask]=True
  stats.append({'video_id':vid,'frames':n,'kept':len(preds),'guard_tripped':guard,
   'spatial_changed_all':int(chg.sum()),'spatial_changed_kept':int((chg&keep).sum()),'spatial_changed_removed':int((chg&~keep).sum())})
 dest=a.output/'predictions.jsonl';validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=None).to_dict();independent=check(a.index,dest,None,require_size=False)
 # re-read the written file: frame keys == mask parent, bboxes == spatial parent
 out={r['video_id']:r for r in load_jsonl(dest)};key_bad=bbox_bad=0
 if len(out)!=len(index) or set(out)!=set(index):raise ValueError('video coverage')
 for vid,o in out.items():
  if [x['frame'] for x in o['predictions']]!=[x['frame'] for x in mp[vid]['predictions']]:key_bad+=1
  pb=[x['bboxes'] for x in sp[vid]['predictions']];bbox_bad+=sum(1 for x in o['predictions'] if x['bboxes']!=pb[x['frame']])
  if o['targetRatioWH']!=sp[vid]['targetRatioWH']:raise ValueError('ratio '+vid)
 if key_bad or bbox_bad:raise ValueError(f'key diff {key_bad} videos / bbox diff {bbox_bad} frames')
 zp=a.output/f"{fz['submission_id']}.zip"
 with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as zf:zf.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zp) as zf:
   if zf.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   zf.extractall(tmp)
  roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',None,require_size=False)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('roundtrip')
 tot={k:int(sum(s[k] for s in stats)) for k in ('spatial_changed_all','spatial_changed_kept','spatial_changed_removed')}
 man={'submission_id':fz['submission_id'],'status':fz['release_status'],'uploaded':False,'official_platform_score':None,'parents':{k:fz[k] for k in ('spatial_parent','mask_parent','mask_parent_spatial_source')},
  'primary_changed_factor':fz['primary_changed_factor'],'additive_reference_not_prediction':fz.get('additive_reference_not_prediction'),'frozen_sha256':sha256_file(a.frozen),'components':fz['components'],
  'total_parameters':sum(c['parameters'] for c in fz['components']),'total_weight_bytes':sum(c['bytes'] for c in fz['components']),
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'empty_videos':[r['video_id'] for r in rows if not r['predictions']],
  'contract':{'keys_equal_mask_parent_videos_bad':key_bad,'bbox_equal_spatial_parent_frames_bad':bbox_bad,'mask_recomputed_from_removal':True,'guard_tripped':sum(s['guard_tripped'] for s in stats)},
  'diff_vs_mask_parent':{'kept_frames_bbox_changed':tot['spatial_changed_kept'],'videos_changed':sum(1 for s in stats if s['spatial_changed_kept']),
   'spatial_vs_mask_source_changed_all_frames':tot['spatial_changed_all'],'spatial_changes_on_removed_frames':tot['spatial_changed_removed'],
   'spatial_change_retained_frac':tot['spatial_changed_kept']/tot['spatial_changed_all'] if tot['spatial_changed_all'] else None},
  'per_video':stats,'validator':validation,'independent':independent,'unzip_independent':roundtrip,'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zp),'zip_bytes':zp.stat().st_size}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1)+'\n');(a.output/f"{fz['submission_id']}.zip.sha256").write_text(man['zip_sha256']+f"  {fz['submission_id']}.zip\n")
 print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator','components','parents')}))
if __name__=='__main__':main()
