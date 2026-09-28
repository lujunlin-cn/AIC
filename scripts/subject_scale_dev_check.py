"""Engineering check of the frozen S2 crop on QVH train/dev sources (no GT crop).

For each video and both AIC ratios (sources are 534x300 landscape, so the
landscape->portrait 9:16 case and a milder 3:4 case are exercised): legality of every window, nesting inside the B0 face window, scale
distribution, per-frame center/scale jitter excluding resets, and a contact
sheet (base window = green, S2 window = red) for visual review.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from aic.subject_crop import ScalePath,BatchDetector
from aic.spatial_pipeline import SpatialPath
from aic.video import _decoded


def main():
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--frozen',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--split',default='dev');p.add_argument('--ratios',default='9,16;3,4');a=p.parse_args()
 fz=json.loads(Path(a.frozen).read_text());a.output.mkdir(parents=True,exist_ok=False);det=BatchDetector(fz['detector'],'cuda',score=fz['detector_score'])
 recs=[r for r in map(json.loads,open(a.records)) if r['split']==a.split];rows=[]
 for ratio in [[int(x) for x in s.split(',')] for s in a.ratios.split(';')]:
  for r in recs:
   base=SpatialPath(fz['spatial_mode'],ratio,fz['face'],alpha=fz['base_alpha']);sp=ScalePath(ratio,fz['face'],min_scale=fz['min_scale'],margin=fz['margin'],alpha=fz['alpha'],scale_alpha=fz['scale_alpha'])
   B=[];S=[];R=[];thumbs=[];buf=[]
   def flush():
    for f,d in zip(buf,det(buf)):
     b,_=base.step(f);s,rs=sp.step(f,d,b);B.append(b);S.append(s);R.append(rs)
     if len(B)%375==1:thumbs.append((f.copy(),b,s))
    buf.clear()
   for st,fr in _decoded(r['video_path']):
    img=fr.to_ndarray(format='rgb24');H,W=img.shape[:2];buf.append(img)
    if len(buf)==fz['batch']:flush()
   if buf:flush()
   B=np.asarray(B);S=np.asarray(S);rh=ratio[1]/ratio[0];eps=1e-6
   legal=bool(np.all((S[:,0]>=-eps)&(S[:,1]>=-eps)&(S[:,0]+S[:,2]<=W+eps)&(S[:,1]+S[:,2]*rh<=H+eps)&(S[:,2]>0)))
   nested=bool(np.all((S[:,0]>=B[:,0]-eps)&(S[:,1]>=B[:,1]-eps)&(S[:,0]+S[:,2]<=B[:,0]+B[:,2]+eps)&(S[:,1]+S[:,2]*rh<=B[:,1]+B[:,2]*rh+eps)))
   sc=S[:,2]/B[:,2];keep=~np.asarray(R[1:])
   cx=(S[:,0]+S[:,2]/2)/W;cy=(S[:,1]+S[:,2]*rh/2)/H
   jit=float(np.median(np.hypot(np.diff(cx),np.diff(cy))[keep])) if keep.any() else 0.
   sj=float(np.median(np.abs(np.diff(sc))[keep])) if keep.any() else 0.
   rows.append({'vid':r['vid'],'ratio':ratio,'frames':len(S),'legal':legal,'nested_in_base':nested,'resets':int(sum(R)),'scale_mean':float(sc.mean()),'scale_p10':float(np.percentile(sc,10)),
    'scale_min':float(sc.min()),'scaled_frame_rate':float((sc<.995).mean()),'center_jitter_median':jit,'scale_jitter_median':sj,'p95_center_jump':float(np.percentile(np.hypot(np.diff(cx),np.diff(cy))[keep],95)) if keep.any() else 0.})
   if len(rows)<=8 and ratio==[9,16]:
    tiles=[]
    for img,b,s in thumbs[:6]:
     t=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
     for (x,y,w),col in ((b,(0,200,0)),(s,(0,0,255))):cv2.rectangle(t,(int(x),int(y)),(int(x+w),int(y+w*rh)),col,2)
     tiles.append(t)
    while len(tiles)<6:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(a.output/f"sheet_{r['vid']}.jpg"),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
   print(json.dumps(rows[-1]),flush=True)
 summ={}
 for rr in sorted({tuple(x['ratio']) for x in rows}):
  v=[x for x in rows if tuple(x['ratio'])==rr]
  summ[f'{rr[0]}:{rr[1]}']={'videos':len(v),'legal_all':all(x['legal'] for x in v),'nested_all':all(x['nested_in_base'] for x in v),'scale_mean':float(np.mean([x['scale_mean'] for x in v])),
   'scaled_frame_rate':float(np.mean([x['scaled_frame_rate'] for x in v])),'center_jitter_median':float(np.median([x['center_jitter_median'] for x in v])),'p95_center_jump_max':float(max(x['p95_center_jump'] for x in v))}
 (a.output/'metrics.json').write_text(json.dumps({'split':a.split,'frozen':a.frozen,'summary':summ,'per_video':rows,'gt_crop_available':False},indent=2)+'\n');print(json.dumps(summ))
if __name__=='__main__':main()
