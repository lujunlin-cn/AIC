"""Frozen foundation candidate -> raw video -> shared YuNet -> verified artifacts.

Run only after train/dev promotion gate. Does not tune or inspect test content.
"""
import argparse,json,time,zipfile,tempfile,resource
from pathlib import Path
import numpy as np
import torch
from aic.foundation import load_encoder,encode_video
from aic.models import TemporalUNet,sha256_file
from aic.inference import _row,_record_path
from aic.contract import load_jsonl,load_index,write_submission
from aic.video import probe_video,frame_timeline,expand_scores,_decoded
from aic.spatial_pipeline import SpatialPath
from scripts.independent_submission_check import check


def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 frozen=json.loads(a.frozen.read_text());a.output.mkdir(parents=True,exist_ok=False)
 if frozen['status']!='frozen_after_dev_gate':raise ValueError('no dev promotion gate')
 for path,digest in frozen['code_files'].items():
  if sha256_file(path)!=digest:raise ValueError('frozen inference implementation changed')
 inventory=[]
 for w in frozen['weights']:
  path=Path(w['path']);actual={'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file(path)}
  if actual['sha256']!=w['sha256'] or actual['bytes']!=w['bytes']:raise ValueError('weight changed')
  inventory.append(actual)
 total=sum(w['bytes'] for w in inventory)
 torch.set_num_threads(4);start=time.time();encoder,_=load_encoder(frozen['kind'],frozen['pretrained_root'],'cuda')
 ck=torch.load(frozen['head'],map_location='cpu',weights_only=True);head=TemporalUNet(ck['input_dim']);head.load_state_dict(ck['model']);head=head.cuda().eval()
 rows=[];diagnostics=[];index=load_index(a.index);records=load_jsonl(a.index)
 if sha256_file(a.index)!=frozen['index_sha256']:raise ValueError('test index changed')
 with torch.inference_mode():
  for r in records:
   t0=time.time();path=_record_path(r,None);info=probe_video(path);meta=index[r['video_id']]
   if (info.width,info.height,info.frame_count)!=(meta.width,meta.height,meta.frame_count):raise ValueError('metadata mismatch')
   feat,times,idx=encode_video(encoder,frozen['kind'],path,'cuda',batch=2 if frozen['kind']=='internvideo' else 8)
   scores=head(torch.tensor(feat,device='cuda')[None])[0].sigmoid().float().cpu().numpy()
   if not np.isfinite(scores).all():raise FloatingPointError('scores')
   selected=expand_scores(frame_timeline(path),times,scores,frozen['threshold']);wanted=set(selected)
   spatial=SpatialPath('true_face_smooth',r['targetRatioWH'],frozen['detector'],alpha=.25);crops={}
   for stamp,frame in _decoded(path):
    crop,_=spatial.step(frame.to_ndarray(format='rgb24'))
    if stamp.index in wanted:crops[stamp.index]=crop
   if set(crops)!=wanted:raise ValueError('missing selected frames')
   rows.append(_row(r['video_id'],r['targetRatioWH'],selected,info.width,info.height,total/1e6,'true_face_smooth',crops))
   diagnostics.append({'video_id':r['video_id'],'predictions':len(selected),'frames':info.frame_count,'seconds':time.time()-t0})
   # Checkpoint technical progress only; no automatic model decisions.
   (a.output/'progress.json').write_text(json.dumps(diagnostics))
   print(json.dumps({'completed':len(rows),'total':len(records),'seconds':time.time()-start}),flush=True)
 dest=a.output/'predictions.jsonl';validation=write_submission(dest,rows,index,stage='preliminary',actual_model_size_mb=total/1e6).to_dict()
 independent=check(a.index,dest,total)
 zip_path=a.output/'upload.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:z.write(dest,'predictions.jsonl')
 with tempfile.TemporaryDirectory() as tmp:
  with zipfile.ZipFile(zip_path) as z:
   if z.namelist()!=['predictions.jsonl']:raise ValueError('ZIP structure')
   z.extractall(tmp)
  independent_roundtrip=check(a.index,Path(tmp)/'predictions.jsonl',total)
  reread=load_jsonl(Path(tmp)/'predictions.jsonl')
  write_submission(Path(tmp)/'verified.jsonl',reread,index,stage='preliminary',actual_model_size_mb=total/1e6)
 report={'submission_id':frozen['submission_id'],'status':'READY_TO_SUBMIT','uploaded':False,'frozen_sha256':sha256_file(a.frozen),'weight_bytes':total,'parameter_count':sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in head.parameters()),'weights':inventory,'video_count':len(rows),'prediction_count':sum(len(r['predictions']) for r in rows),'empty_videos':sum(not r['predictions'] for r in rows),'wall_seconds':time.time()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'cpu_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'validator':validation,'independent':independent,'unzip_independent':independent_roundtrip,'predictions_sha256':sha256_file(dest),'zip_sha256':sha256_file(zip_path),'zip_bytes':zip_path.stat().st_size,'official_score':None}
 (a.output/'run_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.output/'weight_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');(a.output/'upload.zip.sha256').write_text(report['zip_sha256']+'  upload.zip\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
