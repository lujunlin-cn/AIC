#!/usr/bin/env python3
"""RetargetVid dense crop check for SubjectPath vs center / true_face_smooth.

Same native protocol as rescore_retargetvid_native: rounded [x0,y0,x1,y1],
max(coord,0) clamp, inclusive-pixel IoU, mean over 6 annotators and frames.
RetargetVid GT windows are fixed max-size, so only the crop *center* is tested
here; the variable-scale output is reported but penalised by construction.
"""
import argparse,json,time,hashlib
from pathlib import Path
import av,cv2,numpy as np
from aic.spatial_pipeline import SpatialPath
from aic.subject_crop import SubjectPath,ScalePath,BatchDetector
from scripts.benchmark_spatial import iou

RATIOS={'1-3':[1,3],'3-1':[3,1]}

def box(c,r):
 x,y,w=c;return np.rint([x,y,x+w,y+w*r[1]/r[0]]).astype(int)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--videos',required=True);ap.add_argument('--annotations',required=True)
 ap.add_argument('--detector',required=True);ap.add_argument('--face',required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--min-scale',type=float,default=.6);ap.add_argument('--batch',type=int,default=16);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);det=BatchDetector(a.detector);cv2.setNumThreads(2);rows=[];start=time.time()
 for p in sorted(Path(a.videos).glob('*.AVI')):
  vid=p.stem;gt={r:np.maximum(np.stack([np.loadtxt(Path(a.annotations)/f'annotator_{i}'/f'{vid}_{r}.txt',delimiter=',') for i in range(1,7)]),0) for r in RATIOS}
  with av.open(str(p)) as c:frames=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
  dets=[]
  for i in range(0,len(frames),a.batch):dets+=det(frames[i:i+a.batch])
  for r,ratio in RATIOS.items():
   if gt[r].shape[1]!=len(frames):raise ValueError(f'{vid} frame mismatch')
   face=SpatialPath('true_face_smooth',ratio,a.face,alpha=.25);center=SpatialPath('center',ratio)
   sub=SubjectPath(ratio,a.face,min_scale=a.min_scale);nest=ScalePath(ratio,a.face,min_scale=a.min_scale)
   pred={k:[] for k in ('center','true_face_smooth','subject_full','subject_scaled','face_scaled')};raw={k:[] for k in pred};scales=[];nscales=[];resets=[]
   for f,d in zip(frames,dets):
    fc,rs=face.step(f);cc=center.step(f)[0];scaled,full,_=sub.step(f,d);nc,_=nest.step(f,d,fc)
    for k,c in (('center',cc),('true_face_smooth',fc),('subject_full',full),('subject_scaled',scaled),('face_scaled',nc)):pred[k].append(box(c,ratio));raw[k].append(c)
    scales.append(sub.scale);nscales.append(nc[2]/fc[2]);resets.append(rs)
   H,W=frames[0].shape[:2]
   for m,b in pred.items():
    b=np.maximum(np.asarray(b),0);ov=iou(b[None],gt[r]);c=np.asarray(raw[m],float);ch=c[:,2]*ratio[1]/ratio[0]
    legal=bool(((c[:,0]>=0)&(c[:,1]>=0)&(c[:,2]>0)&(c[:,0]+c[:,2]<=W+1e-6)&(c[:,1]+ch<=H+1e-6)).all())
    ctr=np.stack([(c[:,0]+c[:,2]/2)/W,(c[:,1]+ch/2)/H],1);ok=~np.asarray(resets[1:])
    jit=float(np.linalg.norm(np.diff(ctr,axis=0),axis=1)[ok].mean()) if ok.any() else 0.;sj=float(np.abs(np.diff(c[:,2]/c[:,2].max()))[ok].mean()) if ok.any() else 0.
    ms={'subject_scaled':float(np.mean(scales)),'face_scaled':float(np.mean(nscales))}.get(m,1.0)
    rows.append({'video_id':vid,'ratio':r,'mode':m,'iou':float(ov.mean()),'frames':len(frames),'legal':legal,'center_jitter_per_frame':jit,'scale_jitter_per_frame':sj,
                 'mean_scale':ms,'scaled_frame_rate':float(np.mean(np.asarray(nscales)<.995)) if m=='face_scaled' else None,'subject_frame_rate':sub.subject_frames/len(frames)})
  print('DONE',vid,len(frames),round(time.time()-start,1),flush=True)
 modes=sorted({r['mode'] for r in rows});summary={m:{r:float(np.mean([x['iou'] for x in rows if x['mode']==m and x['ratio']==r])) for r in RATIOS} for m in modes}
 paired={}
 for m in ('subject_full','true_face_smooth','face_scaled'):
  for r in RATIOS:
   d=np.array([x['iou']-y['iou'] for x in rows for y in rows if x['mode']==m and y['mode']=='center' and x['ratio']==r==y['ratio'] and x['video_id']==y['video_id']])
   rng=np.random.default_rng(20260927);boot=[rng.choice(d,len(d)).mean() for _ in range(2000)]
   paired[f'{m}-center@{r}']={'mean':float(d.mean()),'ci95':[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],'positive_videos':int((d>0).sum()),'videos':len(d)}
 d=np.array([x['iou']-y['iou'] for x in rows for y in rows if x['mode']=='subject_full' and y['mode']=='true_face_smooth' and x['ratio']==y['ratio'] and x['video_id']==y['video_id']])
 paired['subject_full-true_face_smooth@both']={'mean':float(d.mean()),'positive':int((d>0).sum()),'n':len(d)}
 d=np.array([x['iou']-y['iou'] for x in rows for y in rows if x['mode']=='face_scaled' and y['mode']=='true_face_smooth' and x['ratio']==y['ratio'] and x['video_id']==y['video_id']])
 paired['face_scaled-true_face_smooth@both']={'mean':float(d.mean()),'positive':int((d>0).sum()),'n':len(d),'note':'GT max-size: this is the downside bound if AIC GT were max-size'}
 agg={m:{'legal_all':all(x['legal'] for x in rows if x['mode']==m),'center_jitter':float(np.mean([x['center_jitter_per_frame'] for x in rows if x['mode']==m])),'scale_jitter':float(np.mean([x['scale_jitter_per_frame'] for x in rows if x['mode']==m])),'mean_scale':float(np.mean([x['mean_scale'] for x in rows if x['mode']==m]))} for m in modes}
 report={'protocol':'RETARGETVID_DENSE_NATIVE_IOU_V2 (subject crop)','tuning':'none; fixed parameters before measurement','min_scale':a.min_scale,
  'gt_scale_note':'all RetargetVid GT windows are max legal size; scale cannot be validated here','methods':summary,'paired':paired,'geometry':agg,'per_video':rows,
  'detector_sha256':hashlib.sha256(Path(a.detector).read_bytes()).hexdigest(),'detector_parameters':det.parameters,'elapsed_seconds':time.time()-start,'official_f_video':None}
 (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('methods','paired','geometry')},indent=2))
if __name__=='__main__':main()
