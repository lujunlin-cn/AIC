"""Spatial-factor packages on the frozen official index (no test tuning).

`run`: one shard; decodes every frame in PTS order, steps the B0 face path
(true_face_smooth, max window) and a ScalePath nested inside it statefully over
all frames, and saves per-frame base/scaled crops per video.
`finalize --variant`: builds one package from saved crops.
  S1_CENTER_MAX      all frames, largest centered legal window (no weights)
  S2_SUBJECT_SCALE   all frames, variable-scale sub-window inside the B0 face window
  Q2_QWEN_SUBJECT    Qwen SUB_Q frame selection, crops identical to S2
"""
import argparse,json,time,zipfile,tempfile,resource,hashlib
from pathlib import Path
import numpy as np
from aic.inference import _record_path
from aic.contract import load_jsonl,load_index,write_submission,center_crop
from aic.video import probe_video,_decoded
from aic.subject_crop import ScalePath,BatchDetector
from aic.spatial_pipeline import SpatialPath
from scripts.independent_submission_check import check
from scripts.all_select_yunet_release import sha256_file

VARIANTS={'S1_CENTER_MAX':('all','center'),'S2_SUBJECT_SCALE':('all','scaled'),'Q2_QWEN_SUBJECT':('qwen','scaled')}


def load_frozen(path,index):
 frozen=json.loads(Path(path).read_text())
 if frozen['status']!='frozen_exploratory':raise ValueError('config not frozen')
 if sha256_file(index)!=frozen['index_sha256']:raise ValueError('index changed')
 for w in frozen['weights']:
  if Path(w['path']).stat().st_size!=w['bytes'] or sha256_file(w['path'])!=w['sha256']:raise ValueError('weight changed '+w['path'])
 return frozen


def run(a):
 import torch
 frozen=load_frozen(a.frozen,a.index);a.output.mkdir(parents=True,exist_ok=False)
 index=load_index(a.index);records=load_jsonl(a.index)[a.part::a.parts]
 det=BatchDetector(frozen['detector'],'cuda',score=frozen['detector_score']);start=time.time();diag=[]
 for r in records:
  t0=time.time();path=_record_path(r,None);info=probe_video(path);meta=index[r['video_id']]
  if (info.width,info.height,info.frame_count)!=(meta.width,meta.height,meta.frame_count):raise ValueError('metadata mismatch')
  base_path=SpatialPath(frozen['spatial_mode'],r['targetRatioWH'],frozen['face'],alpha=frozen['base_alpha'])
  sub=ScalePath(r['targetRatioWH'],frozen['face'],min_scale=frozen['min_scale'],margin=frozen['margin'],alpha=frozen['alpha'],scale_alpha=frozen['scale_alpha'])
  scaled=[];full=[];order=[];buf=[]
  def flush():
   for (i,f),d in zip(buf,det([f for _,f in buf])):
    b,_=base_path.step(f);s,_=sub.step(f,d,b);scaled.append(s);full.append(b);order.append(i)
   buf.clear()
  for stamp,frame in _decoded(path):
   buf.append((stamp.index,frame.to_ndarray(format='rgb24')))
   if len(buf)==frozen['batch']:flush()
  if buf:flush()
  if order!=list(range(meta.frame_count)):raise ValueError('decoded frames != index frame_count '+r['video_id'])
  center=center_crop(info.width,info.height,r['targetRatioWH'])
  np.savez(a.output/f"{r['video_id']}.npz",scaled=np.asarray(scaled,np.float64),full=np.asarray(full,np.float64),center=np.asarray(center,np.float64))
  sc=np.asarray(scaled)[:,2]/np.asarray(full)[:,2]
  diag.append({'video_id':r['video_id'],'frames':len(order),'seconds':time.time()-t0,'resets':sub.resets,'subject_frame_rate':sub.subject_frames/len(order),
               'scale_mean':float(sc.mean()),'scale_p10':float(np.percentile(sc,10)),'scale_min':float(sc.min())})
  print(json.dumps({'part':a.part,'completed':len(diag),'total':len(records),'seconds':time.time()-start}),flush=True)
 (a.output/'shard_report.json').write_text(json.dumps({'part':a.part,'parts':a.parts,'wall_seconds':time.time()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
  'cpu_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'detector_parameters':det.parameters,'diagnostics':diag},indent=2)+'\n')


def finalize(a):
 frozen=load_frozen(a.frozen,a.index);a.output.mkdir(parents=True,exist_ok=False)
 frames_src,crop_key=VARIANTS[a.variant];index=load_index(a.index);records=load_jsonl(a.index)
 crops={};reports=[]
 for shard in a.shard:
  reports.append(json.loads((Path(shard)/'shard_report.json').read_text()))
  for f in Path(shard).glob('*.npz'):
   if f.stem in crops:raise ValueError('duplicate '+f.stem)
   crops[f.stem]=f
 if set(crops)!=set(index):raise ValueError(f'coverage mismatch {len(crops)} vs {len(index)}')
 qwen={r['video_id']:[p['frame'] for p in r['predictions']] for r in load_jsonl(frozen['qwen_predictions'])} if frames_src=='qwen' else None
 # S1 loads no weights. Q2 truly loads Qwen3-VL-32B (> 9 GB / 9 B limit), so no
 # placeholder size is written: preliminary-format internal probe without the field.
 if crop_key=='center':weights=[];size_mb=frozen['no_weight_model_size_mb']
 else:weights=frozen['weights'];size_mb=sum(w['bytes'] for w in weights)/1e6
 internal=frames_src=='qwen'
 if internal:weights=weights+frozen['qwen_weights'];size_mb=None
 b0={r['video_id']:{p['frame']:p['bboxes'] for p in r['predictions']} for r in load_jsonl(frozen['b0_predictions'])}
 rows=[];scales=[];base_diff=0.
 for r in records:
  vid=r['video_id'];meta=index[vid];z=np.load(crops[vid])
  # the saved base window must reproduce the B0 face path frame by frame
  base_diff=max(base_diff,float(np.abs(z['full']-np.asarray([b0[vid][i] for i in range(meta.frame_count)])).max()))
  frames=list(range(meta.frame_count)) if qwen is None else qwen[vid]
  if crop_key=='center':box=lambda i:z['center'].tolist()
  else:
   arr=z[crop_key]
   if len(arr)!=meta.frame_count:raise ValueError('crop count '+vid)
   box=lambda i:arr[i].tolist()
  preds=[{'frame':int(i),'bboxes':box(i)} for i in frames]
  scales+=[z[crop_key][p['frame']][2]/z['full'][p['frame']][2] if crop_key!='center' else 1. for p in preds]
  row={'video_id':vid,'targetRatioWH':r['targetRatioWH'],'model_size_mb':size_mb,'predictions':preds}
  if internal:del row['model_size_mb']
  rows.append(row)
 if base_diff>1e-6:raise ValueError(f'base window differs from B0 by {base_diff}')
 dest=a.output/'predictions.jsonl';validation=write_submission(dest,rows,index,stage='preliminary' if internal else 'final',actual_model_size_mb=size_mb).to_dict()
 total_bytes=None if internal else int(round(size_mb*1e6));size_is_placeholder=crop_key=='center';independent=check(a.index,dest,total_bytes,require_size=not internal)
 zip_path=a.output/f'{a.variant}.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:z.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zip_path) as z:
   if z.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   z.extractall(tmp)
  roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',total_bytes,require_size=not internal)
  if sha256_file(Path(tmp)/'predictions.jsonl')!=sha256_file(dest):raise ValueError('ZIP roundtrip changed bytes')
 report={'submission_id':a.variant,'status':'INTERNAL_TEACHER_PROBE_NOT_FOR_UPLOAD' if internal else 'READY_TO_SUBMIT_EXPLORATORY','uploaded':False,'official_submission':'NOT_SUBMITTED','official_f_video':None,
  'frozen_sha256':sha256_file(a.frozen),'temporal':frames_src,'crop':crop_key,'weights':weights,'model_size_mb':size_mb,'model_size_is_placeholder':size_is_placeholder,'true_parameters':sum(w.get('parameters',0) for w in weights),'true_weight_bytes':sum(w['bytes'] for w in weights),
  'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'empty_videos':[r['video_id'] for r in rows if not r['predictions']],
  'base_vs_b0_max_abs':base_diff,'scale_mean':float(np.mean(scales)),'scaled_frame_rate':float(np.mean(np.asarray(scales)<.995)),'scale_p10':float(np.percentile(scales,10)),'scale_min':float(np.min(scales)),
  'shard_wall_seconds':[s['wall_seconds'] for s in reports],'peak_vram_bytes':max(s['peak_vram_bytes'] for s in reports),'detector_parameters':reports[0]['detector_parameters'],
  'validator':validation,'independent':independent,'unzip_independent':roundtrip,'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zip_path),'zip_bytes':zip_path.stat().st_size}
 (a.output/'run_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.output/f'{a.variant}.zip.sha256').write_text(report['zip_sha256']+f'  {a.variant}.zip\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('validator','weights')}),flush=True)


def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 r=sub.add_parser('run');r.add_argument('--part',type=int,default=0);r.add_argument('--parts',type=int,default=1)
 f=sub.add_parser('finalize');f.add_argument('--shard',action='append',required=True);f.add_argument('--variant',choices=sorted(VARIANTS),required=True)
 for s in (r,f):s.add_argument('--frozen',type=Path,required=True);s.add_argument('--index',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
 a=p.parse_args();run(a) if a.cmd=='run' else finalize(a)
if __name__=='__main__':main()
