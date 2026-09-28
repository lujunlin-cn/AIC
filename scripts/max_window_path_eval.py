"""P1a: same YuNet observations, only the max-window path changes (RetargetVid).

Dev = DHF1K 001-020 (already exposed), confirm = 021-030 (never used before).
All parameter choices (lam, saliency weight) are made on dev only; confirm is
evaluated once with the frozen choice.  IoU protocol matches
scripts/benchmark_subject_crop.py (rounded boxes, inclusive pixels, mean over
6 annotators and frames).  Also checks that the cached B0 crop reproduces a
fresh SpatialPath('true_face_smooth') within 1e-9 px on sampled videos.
"""
import argparse,json
from pathlib import Path
import numpy as np
from aic.max_window_path import cache_arrays,centre_to_offset,dp_path,shot_static,to_crops,geometry
from scripts.benchmark_spatial import iou

RATIOS={'1-3':[1,3],'3-1':[3,1]}


def boxes(c,r):
 c=np.asarray(c,float);return np.maximum(np.rint(np.column_stack([c[:,0],c[:,1],c[:,0]+c[:,2],c[:,1]+c[:,2]*r[1]/r[0]])).astype(int),0)


def load_gt(ann,vid,r):
 return np.maximum(np.stack([np.loadtxt(Path(ann)/f'annotator_{i}'/f'{vid}_{r}.txt',delimiter=',') for i in range(1,7)]),0)


def policies(z,lams,sal_ws):
 W,H,ratio,raw,reset,face=cache_arrays(z);w,h,axis=geometry(W,H,ratio);span=(W-w) if axis==0 else ((H-h) if axis==1 else 0.)
 comp=0 if axis==0 else 1;obs=centre_to_offset(raw[:,comp],W,H,ratio)
 out={'B0':z['b0'].tolist(),'CENTER':to_crops(W,H,ratio,np.full(len(raw),span/2)),'RAW':to_crops(W,H,ratio,obs)}
 for sw in sal_ws:
  wt=np.where(face,1.,sw)
  out[f'STATIC_sw{sw}']=to_crops(W,H,ratio,shot_static(obs,wt,reset))
  for lam in lams:out[f'DP_l{lam}_sw{sw}']=to_crops(W,H,ratio,dp_path(obs,wt,reset,span,lam=lam))
 return out,{'face_rate':float(face.mean()),'shots':int(reset.sum()),'frames':len(raw)}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--annotations',required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--dev',default='001-020');ap.add_argument('--confirm',default='021-030');ap.add_argument('--lams',default='0.5,1,2,4,8,16');ap.add_argument('--sal-ws',default='0.25,1')
 ap.add_argument('--frozen',help='JSON {"policy": name} to evaluate on confirm; if absent choose on dev');a=ap.parse_args()
 rng=lambda s:[f'{i:03d}' for i in range(int(s.split('-')[0]),int(s.split('-')[1])+1)]
 lams=[float(x) for x in a.lams.split(',')];sws=[float(x) for x in a.sal_ws.split(',')];rows=[];meta={}
 for split,vids in (('dev',rng(a.dev)),('confirm',rng(a.confirm))):
  for vid in vids:
   for r,ratio in RATIOS.items():
    z=np.load(a.cache/f'{vid}_{r}.npz');gt=load_gt(a.annotations,vid,r)
    pol,m=policies(z,lams,sws);meta[f'{vid}_{r}']=m
    if gt.shape[1]!=len(z['b0']):raise ValueError(f'{vid} frame mismatch')
    for name,c in pol.items():
     c=np.asarray(c);ov=iou(boxes(c,ratio)[None],gt).mean()
     ctr=c[:,0]+c[:,1];mv=np.abs(np.diff(ctr))[~z['reset'][1:].astype(bool)]
     rows.append({'split':split,'video_id':vid,'ratio':r,'policy':name,'iou':float(ov),'motion_px_per_frame':float(mv.mean()) if len(mv) else 0.,'static_frame_rate':float((mv<.5).mean()) if len(mv) else 1.})
 names=sorted({x['policy'] for x in rows});summ={}
 for split in ('dev','confirm'):
  for n in names:
   v=[x for x in rows if x['split']==split and x['policy']==n];summ[f'{split}:{n}']={'iou':float(np.mean([x['iou'] for x in v])),**{r:float(np.mean([x['iou'] for x in v if x['ratio']==r])) for r in RATIOS},'motion':float(np.mean([x['motion_px_per_frame'] for x in v]))}
 def paired(split,n,base='B0'):
  d=np.array([x['iou']-y['iou'] for x in rows for y in rows if x['split']==y['split']==split and x['policy']==n and y['policy']==base and x['video_id']==y['video_id'] and x['ratio']==y['ratio']])
  # resample videos (both ratios of a video move together)
  dv=d.reshape(-1,2).mean(1);b=np.random.default_rng(20260928);bs=[b.choice(dv,len(dv)).mean() for _ in range(5000)]
  return {'mean':float(d.mean()),'ci95_video':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'improved_pairs':int((d>1e-4).sum()),'worse_pairs':int((d<-1e-4).sum()),'pairs':len(d),'worst':float(d.min())}
 cand=[n for n in names if n.startswith(('DP','STATIC'))]
 chosen=json.loads(Path(a.frozen).read_text())['policy'] if a.frozen else max(cand,key=lambda n:summ[f'dev:{n}']['iou'])
 out={'protocol':'RetargetVid native IoU, max window, same B0 YuNet observations','dev':a.dev,'confirm':a.confirm,'chosen_on_dev':chosen,'summary':summ,
  'paired_vs_B0':{f'{s}:{n}':paired(s,n) for s in ('dev','confirm') for n in names if n!='B0'},'meta':meta,'per_video':rows,'official_f_video':None}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'metrics.json').write_text(json.dumps(out,indent=1)+'\n')
 for s in ('dev','confirm'):
  for n in ['B0','CENTER','RAW',chosen]+[x for x in cand if x!=chosen][:0]:print(s,n,json.dumps(summ[f'{s}:{n}']),json.dumps(out['paired_vs_B0'].get(f'{s}:{n}')))
 print('chosen',chosen)
if __name__=='__main__':main()
