"""R0 diagnostic: take the override package only on videos whose free axis is `--axis`.

Base and override must share (video_id, frame) keys exactly.  The free axis of
each video comes from its source size and target ratio (aic.max_window_path
.geometry: 0 = window slides along x, 1 = along y); no video IDs are listed.
Output rows are copied from the override on the chosen axis and from the base
elsewhere, byte-identical bboxes.  Checks: parent SHAs, keys == base, rows ==
override (chosen axis) / base (other axis), validator, independent checker,
unzip roundtrip.
"""
import argparse,json,zipfile,tempfile
from pathlib import Path
from aic.contract import load_jsonl,load_index,write_submission
from aic.max_window_path import geometry
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 fz=json.loads(a.frozen.read_text())
 if fz['status']!='frozen_exploratory' or sha256_file(a.index)!=fz['index_sha256']:raise ValueError('config not frozen / index changed')
 for k in ('base','override'):
  if sha256_file(fz[k]['predictions'])!=fz[k]['predictions_sha256']:raise ValueError(k+' changed')
 a.output.mkdir(parents=True,exist_ok=False);index=load_index(a.index);axis=int(fz['override_axis'])
 base,over=({r['video_id']:r for r in load_jsonl(fz[k]['predictions'])} for k in ('base','override'))
 rows=[];stats=[]
 for r in load_jsonl(a.index):
  vid=r['video_id'];b,o=base[vid],over[vid]
  if [x['frame'] for x in b['predictions']]!=[x['frame'] for x in o['predictions']]:raise ValueError('keys differ '+vid)
  _,_,ax=geometry(r['width'],r['height'],r['targetRatioWH']);src=o if ax==axis else b
  rows.append(src);stats.append({'video_id':vid,'axis':ax,'source':'override' if ax==axis else 'base','frames':len(src['predictions']),
   'override_differs_from_base':sum(1 for x,y in zip(b['predictions'],o['predictions']) if x['bboxes']!=y['bboxes'])})
 dest=a.output/'predictions.jsonl';validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=None).to_dict();independent=check(a.index,dest,None,require_size=False)
 out={r['video_id']:r for r in load_jsonl(dest)};bad=0
 for s in stats:
  ref=(over if s['source']=='override' else base)[s['video_id']]
  if out[s['video_id']]['predictions']!=ref['predictions'] or out[s['video_id']]['targetRatioWH']!=ref['targetRatioWH']:bad+=1
 if bad:raise ValueError(f'{bad} videos differ from their declared source')
 zp=a.output/f"{fz['submission_id']}.zip"
 with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as zf:zf.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zp) as zf:
   if zf.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   zf.extractall(tmp)
  roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',None,require_size=False)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('roundtrip')
 grp=lambda src:[s for s in stats if s['source']==src]
 man={'submission_id':fz['submission_id'],'status':fz['release_status'],'uploaded':False,'official_platform_score':None,'parents':{k:fz[k] for k in ('base','override')},'override_axis':axis,
  'primary_changed_factor':fz['primary_changed_factor'],'score_decomposition_rule':fz['score_decomposition_rule'],'frozen_sha256':sha256_file(a.frozen),'components':fz['components'],
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'empty_videos':[r['video_id'] for r in rows if not r['predictions']],
  'groups':{src:{'videos':len(grp(src)),'frames':sum(s['frames'] for s in grp(src)),'videos_where_override_differs':sum(1 for s in grp(src) if s['override_differs_from_base']),
   'frames_where_override_differs':sum(s['override_differs_from_base'] for s in grp(src))} for src in ('override','base')},'rows_not_equal_declared_source':bad,
  'per_video':stats,'validator':validation,'independent':independent,'unzip_independent':roundtrip,'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zp),'zip_bytes':zp.stat().st_size}
 (a.output/'manifest.json').write_text(json.dumps(man,indent=1)+'\n');(a.output/f"{fz['submission_id']}.zip.sha256").write_text(man['zip_sha256']+f"  {fz['submission_id']}.zip\n")
 print(json.dumps({k:v for k,v in man.items() if k not in ('per_video','validator','components','parents')}))
if __name__=='__main__':main()
