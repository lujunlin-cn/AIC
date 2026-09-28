"""Build per-frame observation caches (B0 YuNet path + FasterRCNN detections).

  official: --index <enriched index> --part/--parts   (frozen inference only)
  retarget: --videos <dir of *.AVI> --ratios 1-3,3-1   (RetargetVid dev/confirm)
Detections are kept at score >= --det-score (default .3) in original pixels so
later selectors can raise the threshold without re-running the detector.
"""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from aic.obs_cache import ObservedFacePath,save_cache
from aic.subject_crop import BatchDetector
from aic.video import _decoded


def cache_video(path,ratios,face,det,batch):
 paths={k:ObservedFacePath(r,face) for k,r in ratios.items()};crops={k:[] for k in ratios};obs={k:[] for k in ratios};dets=[];buf=[];n=0
 def flush():
  for f,d in zip(buf,det(buf)):
   dets.append(d)
   for k,p in paths.items():c,o=p.step(f);crops[k].append(c);obs[k].append(o)
  buf.clear()
 for stamp,frame in _decoded(path):
  if stamp.index!=n:raise ValueError('non-contiguous frame index')
  buf.append(frame.to_ndarray(format='rgb24'));n+=1
  if len(buf)==batch:flush()
 if buf:flush()
 return n,crops,obs,dets


def main():
 p=argparse.ArgumentParser();p.add_argument('--index');p.add_argument('--videos');p.add_argument('--ratios',default='1-3,3-1')
 p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1);p.add_argument('--face',required=True);p.add_argument('--detector',required=True)
 p.add_argument('--det-score',type=float,default=.3);p.add_argument('--batch',type=int,default=16);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cv2,torch;cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=True);det=BatchDetector(a.detector,'cuda',score=a.det_score)
 if a.index:
  from aic.inference import _record_path
  jobs=[(r['video_id'],_record_path(r,None),{'t':r['targetRatioWH']},(r['width'],r['height'],r['frame_count'])) for r in map(json.loads,open(a.index))]
 else:
  rs={k:[int(x) for x in k.split('-')] for k in a.ratios.split(',')};jobs=[(v.stem,str(v),rs,None) for v in sorted(Path(a.videos).glob('*.AVI'))]
 jobs=jobs[a.part::a.parts];start=time.time();done=[]
 for vid,path,ratios,meta in jobs:
  if all((a.output/f'{vid}_{k}.npz').exists() for k in ratios):continue
  t0=time.time();n,crops,obs,dets=cache_video(path,ratios,a.face,det,a.batch)
  import av
  with av.open(str(path)) as c:s=c.streams.video[0];W,H=s.codec_context.width,s.codec_context.height
  if meta and (W,H,n)!=meta:raise ValueError(f'{vid}: metadata mismatch {(W,H,n)} vs {meta}')
  for k,r in ratios.items():save_cache(a.output/f'{vid}_{k}.npz',W,H,r,crops[k],obs[k],dets)
  done.append({'video_id':vid,'frames':n,'seconds':time.time()-t0});print(json.dumps({'part':a.part,'done':len(done),'total':len(jobs),'vid':vid,'wall':time.time()-start}),flush=True)
 (a.output/f'part{a.part}_of{a.parts}.json').write_text(json.dumps({'part':a.part,'parts':a.parts,'videos':done,'wall_seconds':time.time()-start,'det_score':a.det_score,
  'detector_parameters':det.parameters,'peak_vram':torch.cuda.max_memory_allocated(),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=1)+'\n')
if __name__=='__main__':main()
