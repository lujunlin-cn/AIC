"""T3 package: N0 bboxes unchanged, only the temporal keep mask changes.

For every video the kept frames are a subset of N0's frames and each kept
frame carries N0's bbox byte-for-byte (no re-smoothing on the shorter
sequence).  Coverage contract (pre-declared): >=1 frame per video, frames
sorted/unique, removal guard as in scripts/qwen_temporal_removal.py.
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
 if sha256_file(fz['parent_predictions'])!=fz['parent_predictions_sha256']:raise ValueError('parent changed')
 a.output.mkdir(parents=True,exist_ok=False);index=load_index(a.index);parent={r['video_id']:r for r in load_jsonl(fz['parent_predictions'])}
 rem=Path(fz['removal']);anc=Path(fz['anchors']);rows=[];stats=[]
 for r in load_jsonl(a.index):
  vid=r['video_id'];p=parent[vid];n=index[vid].frame_count;meta=json.loads((anc/vid/'meta.json').read_text())
  if meta['n_frames']!=n or [x['frame'] for x in p['predictions']]!=list(range(n)):raise ValueError('frame count '+vid)
  d=json.loads((rem/f'{vid}.json').read_text());kept,guard=keep_frames(meta['frame_cell'],d['removed_cells'])
  preds=[p['predictions'][i] for i in kept.tolist()]
  if not preds:raise ValueError('empty '+vid)
  row={k:v for k,v in p.items() if k!='predictions'};row['predictions']=preds;rows.append(row)
  stats.append({'video_id':vid,'frames':n,'kept':len(preds),'keep_rate':len(preds)/n,'removed_cells':d['removed_cells'],'guard_tripped':guard})
 dest=a.output/'predictions.jsonl';validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=None).to_dict();independent=check(a.index,dest,None,require_size=False)
 # bbox identity on kept frames
 out={r['video_id']:r for r in load_jsonl(dest)};bad=0
 for vid,r in out.items():
  pp={x['frame']:x['bboxes'] for x in parent[vid]['predictions']};bad+=sum(1 for x in r['predictions'] if x['bboxes']!=pp[x['frame']])
 if bad:raise ValueError(f'{bad} kept frames changed bbox')
 zp=a.output/f"{fz['submission_id']}.zip"
 with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as zf:zf.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zp) as zf:
   if zf.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   zf.extractall(tmp)
  roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',None,require_size=False)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('roundtrip')
 kr=np.array([s['keep_rate'] for s in stats])
 man={'submission_id':fz['submission_id'],'status':fz['release_status'],'uploaded':False,'official_platform_score':None,'parent_baseline':fz['parent_baseline'],'primary_changed_factor':fz['primary_changed_factor'],
  'hypothesis':fz['hypothesis'],'frozen_sha256':sha256_file(a.frozen),'components':fz['components'],'total_parameters':sum(c['parameters'] for c in fz['components']),'total_weight_bytes':sum(c['bytes'] for c in fz['components']),
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'parent_prediction_count':sum(s['frames'] for s in stats),
  'diff_vs_parent':{'bbox_changed_on_kept_frames':bad,'videos_with_removal':int((kr<1).sum()),'frames_removed':int(sum(s['frames']-s['kept'] for s in stats)),'keep_rate_mean':float(kr.mean()),
   'keep_rate_p10_p50':np.percentile(kr,[10,50]).round(4).tolist(),'guard_tripped':sum(s['guard_tripped'] for s in stats)},'per_video':stats,
  'validator':validation,'independent':independent,'unzip_independent':roundtrip,'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zp),'zip_bytes':zp.stat().st_size}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1)+'\n');(a.output/f"{fz['submission_id']}.zip.sha256").write_text(man['zip_sha256']+f"  {fz['submission_id']}.zip\n")
 print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator','components')}))
if __name__=='__main__':main()
