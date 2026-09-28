"""B0: all decoded frames + the V0 stateful YuNet crop path, no encoder / temporal head.

`run` exports one fixed index shard; `finalize` merges shards, validates twice,
compares prediction content with a reference submission, zips and writes the manifest.
Does not tune on or inspect test content beyond format/frame-mapping checks.
"""
import argparse,json,sys,time,zipfile,tempfile,resource,hashlib
from pathlib import Path
from aic.inference import _row,_record_path
from aic.contract import load_jsonl,load_index,write_submission
from aic.video import probe_video,_decoded
from aic.spatial_pipeline import SpatialPath
from scripts.independent_submission_check import check


def sha256_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def load_frozen(path,index):
 frozen=json.loads(Path(path).read_text())
 if frozen['status']!='frozen_baseline':raise ValueError('config not frozen')
 if sha256_file(index)!=frozen['index_sha256']:raise ValueError('index changed')
 det=Path(frozen['detector'])
 if det.stat().st_size!=frozen['detector_bytes'] or sha256_file(det)!=frozen['detector_sha256']:raise ValueError('detector changed')
 return frozen


def run(a):
 frozen=load_frozen(a.frozen,a.index);a.output.mkdir(parents=True,exist_ok=False)
 index=load_index(a.index);records=load_jsonl(a.index)[a.part::a.parts]
 size_mb=frozen['detector_bytes']/1e6;rows=[];diagnostics=[];start=time.time()
 for r in records:
  t0=time.time();path=_record_path(r,None);info=probe_video(path);meta=index[r['video_id']]
  if (info.width,info.height,info.frame_count)!=(meta.width,meta.height,meta.frame_count):raise ValueError('metadata mismatch')
  # Same stateful update as V0: every decoded frame is stepped in PTS order.
  spatial=SpatialPath(frozen['spatial_mode'],r['targetRatioWH'],frozen['detector'],alpha=frozen['alpha']);crops={}
  for stamp,frame in _decoded(path):
   crop,_=spatial.step(frame.to_ndarray(format='rgb24'));crops[stamp.index]=crop
  selected=sorted(crops)
  if selected!=list(range(meta.frame_count)):raise ValueError(f"decoded frames != index frame_count for {r['video_id']}")
  rows.append(_row(r['video_id'],r['targetRatioWH'],selected,info.width,info.height,size_mb,frozen['spatial_mode'],crops))
  diagnostics.append({'video_id':r['video_id'],'predictions':len(selected),'frames':info.frame_count,'seconds':time.time()-t0,'resets':spatial.resets,'detections':spatial.detections})
  print(json.dumps({'part':a.part,'completed':len(rows),'total':len(records),'seconds':time.time()-start}),flush=True)
 with open(a.output/'rows.jsonl','w') as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
 heavy=[m for m in ('torch','transformers','timm') if m in sys.modules]
 (a.output/'shard_report.json').write_text(json.dumps({'part':a.part,'parts':a.parts,'videos':len(rows),'wall_seconds':time.time()-start,'cpu_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'heavy_modules_loaded':heavy,'diagnostics':diagnostics},indent=2)+'\n')


def compare(rows,reference):
 ref={r['video_id']:r for r in load_jsonl(reference)};diffs=[];same_frames=same_crops=0;total=0
 for row in rows:
  vid=row['video_id'];other=ref.get(vid);item={'video_id':vid}
  if other is None:diffs.append({**item,'issue':'missing_in_reference'});continue
  if row['targetRatioWH']!=other['targetRatioWH']:item['ratio']=[row['targetRatioWH'],other['targetRatioWH']]
  fa=[p['frame'] for p in row['predictions']];fb=[p['frame'] for p in other['predictions']]
  if fa!=fb:item['frames']={'b0':len(fa),'reference':len(fb),'only_b0':len(set(fa)-set(fb)),'only_reference':len(set(fb)-set(fa))}
  cb={p['frame']:p['bboxes'] for p in other['predictions']};mismatch=0;max_abs=0.0
  for p in row['predictions']:
   q=cb.get(p['frame'])
   if q is None:continue
   d=max(abs(float(x)-float(y)) for x,y in zip(p['bboxes'],q));max_abs=max(max_abs,d);mismatch+=d>0
  total+=len(fa);same_frames+=len(set(fa)&set(fb));same_crops+=len(set(fa)&set(fb))-mismatch
  if mismatch:item['crop_mismatch_frames']=mismatch;item['crop_max_abs_diff']=max_abs
  if len(item)>1:diffs.append(item)
 missing=sorted(set(ref)-{r['video_id'] for r in rows})
 return {'reference':str(reference),'reference_sha256':sha256_file(reference),'videos':len(rows),'reference_only_videos':missing,'b0_frames':total,'shared_frames':same_frames,'identical_crop_frames':same_crops,'videos_with_differences':len(diffs),'differences':diffs,'prediction_content_identical':not diffs and not missing}


def finalize(a):
 frozen=load_frozen(a.frozen,a.index);a.output.mkdir(parents=True,exist_ok=False)
 index=load_index(a.index);order=list(index);by_id={};shards=[]
 for shard in a.shard:
  for row in load_jsonl(Path(shard)/'rows.jsonl'):
   if row['video_id'] in by_id:raise ValueError('duplicate video_id '+row['video_id'])
   by_id[row['video_id']]=row
  shards.append(json.loads((Path(shard)/'shard_report.json').read_text()))
 if set(by_id)!=set(order):raise ValueError(f'coverage mismatch: {len(by_id)} vs {len(order)}')
 rows=[by_id[v] for v in order];total=frozen['detector_bytes'];dest=a.output/'predictions.jsonl'
 validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=total/1e6).to_dict()
 independent=check(a.index,dest,total);expected=sum(m.frame_count for m in index.values())
 zip_path=a.output/a.zip_name
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:z.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zip_path) as z:
   if z.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   z.extractall(tmp)
  roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',total)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('ZIP roundtrip changed bytes')
 comparison=compare(rows,frozen['reference_predictions']);(a.output/'v0_equivalence.json').write_text(json.dumps(comparison,indent=2)+'\n')
 inventory=[{'path':frozen['detector'],'bytes':total,'sha256':frozen['detector_sha256'],'parameters':'YuNet ONNX (OpenCV DNN, CPU)'}]
 report={'submission_id':frozen['submission_id'],'status':'READY_TO_SUBMIT_BASELINE','uploaded':False,'official_submission':'NOT_SUBMITTED','official_f_video':None,
  'frozen_sha256':sha256_file(a.frozen),'weight_bytes':total,'model_size_mb':total/1e6,'size_coefficient':1.0,'weights':inventory,
  'encoder_loaded':False,'heavy_modules_loaded':sorted({m for s in shards for m in s['heavy_modules_loaded']}),'gpu_used':False,'peak_vram_bytes':0,
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'index_frame_count':expected,'empty_videos':sum(not r['predictions'] for r in rows),
  'shard_wall_seconds':[s['wall_seconds'] for s in shards],'sum_video_seconds':sum(d['seconds'] for s in shards for d in s['diagnostics']),'cpu_maxrss_kib':max(s['cpu_maxrss_kib'] for s in shards),
  'validator':validation,'independent':independent,'unzip_independent':roundtrip,'predictions_sha256':sha256_file(dest),'zip_name':a.zip_name,'zip_sha256':sha256_file(zip_path),'zip_bytes':zip_path.stat().st_size,
  'v0_prediction_content_identical':comparison['prediction_content_identical'],'v0_videos_with_differences':comparison['videos_with_differences'],
  'score_note':'equivalence inference only: identical prediction content implies same raw F_video as V0 (34.42 feedback at k=0.90); not a new official score'}
 if report['prediction_count']!=expected:raise ValueError('prediction count != index frame total')
 (a.output/'run_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.output/'weight_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
 (a.output/(a.zip_name+'.sha256')).write_text(report['zip_sha256']+'  '+a.zip_name+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('validator','weights')}),flush=True)


def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 r=sub.add_parser('run');r.add_argument('--part',type=int,default=0);r.add_argument('--parts',type=int,default=1)
 f=sub.add_parser('finalize');f.add_argument('--shard',action='append',required=True);f.add_argument('--zip-name',default='B0_ALL_SELECT_YUNET_V1.zip')
 for s in (r,f):s.add_argument('--frozen',type=Path,required=True);s.add_argument('--index',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
 a=p.parse_args();run(a) if a.cmd=='run' else finalize(a)
if __name__=='__main__':main()
