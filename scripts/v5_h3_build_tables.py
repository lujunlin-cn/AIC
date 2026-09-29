"""V5 H3 step 1: build unit tables (geometry features, candidates, IoU targets).

T5_CROPHEAD_V3 protocol with three changes: DENSE d2 points (the INTERP mother)
instead of 1 s D1 on RetargetVid, 33 grid candidates (the mother centre is
snapped to the nearest grid candidate, gap recorded as mo_gap_max, so the
candidate count is constant per unit) instead of 65+1, and the exact keyframe
list kept for the visual cache step.  Targets
are the native human-IoU conventions of each dataset (RV rounded-inclusive
6-annotator max-window GT; LIVE clip_exp_v1 rounded-inclusive 1-annotator
variable-size GT).  Splits: rv 001-080/081-100/601-700 (old H2, source-disjoint;
dev2 031-100 overlaps train 001-080 => never quoted as clean), live T5
train/fold-dev/val178 (exposed, report-only).
"""
import argparse,pickle,hashlib,json
from pathlib import Path
import numpy as np
from aic.max_window_path import geometry,centre_to_offset
from scripts.benchmark_spatial import iou
from scripts.max_window_path_eval import load_gt,RATIOS
from scripts.teacher_diag_eval import win_boxes
from scripts.t5_crophead_probe import kf_features,FEATS

NC=33
E=Path('/data/aic/experiments')
T5=E/'T5_CROPHEAD_V3'
RV_ANN=Path('/data/aic/datasets/RetargetVid/annotations_all')
RV_CACHE=E/'OBS_CACHE_RETARGET_ALL200';RV_PTS=E/'QWEN_RV200_POINT_D2/points';RV_FRAMES=E/'QWEN_RV200_KEYFRAMES/d2'


def candidate_offsets(p,W,H,ratio,comp):
 w,h,_=geometry(W,H,ratio);L,s=(W,w) if comp==0 else (H,h)
 grid=np.linspace(0.,L-s,NC);m=float(centre_to_offset(np.array([p[comp]]),W,H,ratio)[0])
 j=int(np.argmin(np.abs(grid-m)))
 return grid.copy(),L,s,m,j,abs(grid[j]-m)


def live_rows(split):
 rows=[json.loads(l) for l in open(T5/f'live_{"val" if split=="val" else "train"}_index.jsonl')]
 fold=lambda v:int(hashlib.sha1(v.encode()).hexdigest(),16)%10==0
 return [r for r in rows if split=='val' or (fold(r['video_id'])==(split=='dev'))]


def live_dirs(sub):
 d='val' if sub=='val' else 'train'  # fold-dev videos live in the train cache/points/frames dirs
 return T5/f'obs_cache_live_{d}',T5/f'points_{d}',T5/f'keyframes_{d}'


def load_rv(vid,r):
 z=np.load(RV_CACHE/f'{vid}_{r}.npz');qq=json.loads((RV_PTS/f'{vid}.json').read_text());gt=load_gt(RV_ANN,vid,r);gf=np.arange(gt.shape[1])
 if gt.shape[1]!=len(z['b0']):raise ValueError('frames '+vid)
 return z,qq,gt,gf


def load_live(row,sub):
 cdir,pdir,fdir=live_dirs(sub)
 z=np.load(cdir/f'{row["video_id"]}_t.npz');qq=json.loads((pdir/f'{row["video_id"]}.json').read_text())
 a=json.loads(Path(row['annotation']).read_text());W,H=int(z['W']),int(z['H'])
 b=np.array([[x['left'],x['top'],x['right'],x['bottom']] for x in a['raw_ltrb']],float);b[:,[0,2]]=np.clip(b[:,[0,2]],0,W-1);b[:,[1,3]]=np.clip(b[:,[1,3]],0,H-1)
 gf=np.array(a['frames']);ok=gf<len(z['b0']);gf,gt=gf[ok],b[ok][None]
 return z,qq,gt,gf


def load_unit(ds,vid,r,row):
 z,qq,gt,gf=(load_rv(vid,r) if ds=='rv' else load_live(row,row.get('_sub','train')))
 W,H=int(z['W']),int(z['H']);ratio=z['ratio'].tolist();w,h,axis=geometry(W,H,ratio)
 if axis is None:return None
 comp=0 if axis==0 else 1;reset=z['reset'].astype(bool);keys=qq['keyframes'];pts=qq['ratios'][r]['points'] if ds=='rv' else qq['ratios']['t']['points']
 shot=np.cumsum(reset);valid=[j for j,p in enumerate(pts) if p is not None and not p[2]];gpos={int(f):i for i,f in enumerate(gf)}
 X=[];Y=[];OF=[];J=[];MO=[];GAPS=[]
 for j in valid:
  t=keys[j];p=pts[j]
  pv=[k for k in valid if k<j and shot[keys[k]]==shot[t]];nx=[k for k in valid if k>j and shot[keys[k]]==shot[t]]
  nb=(pts[pv[-1]][comp] if pv else None,pts[nx[0]][comp] if nx else None)
  offs,L,s,mother_off,mo,gap=candidate_offsets(p,W,H,ratio,comp)
  X.append(kf_features(z,t,p,nb,comp,W,H,ratio,offs));OF.append(offs);J.append(j);MO.append(np.arange(len(offs))==mo);GAPS.append(gap/s)
  Y.append(iou(win_boxes(offs,W,H,ratio)[None],gt[:,gpos[t]][:,None]).mean(0) if t in gpos else np.full(len(offs),np.nan))
 if not J:return None
 C=len(OF[0])
 for o in OF:
  if len(o)!=C:raise ValueError('candidate count varies '+vid)
 fr=E/'QWEN_RV200_KEYFRAMES/d2' if ds=='rv' else live_dirs(row.get('_sub','train'))[2]
 return {'ds':ds,'vid':vid,'r':r,'W':W,'H':H,'ratio':ratio,'comp':comp,'L':L,'s':s,'keys':keys,'pts':pts,'reset':reset,
  'raw':z['raw'][:,comp].copy(),'face':z['chosen']>=0,'gt':gt,'gf':gf,'X':np.array(X,np.float32).reshape(-1,C,len(FEATS)),
  'Y':np.array(Y,np.float32).reshape(-1,C),'offs':np.array(OF).reshape(-1,C),'is_mother':np.array(MO,bool).reshape(-1,C),
  'mo_gap_max':float(max(GAPS)),'J':np.array(J,int),'pa':np.array([pts[j][comp] for j in J]),'frames_dir':str(fr),
  'has_gt':~np.isnan(np.array(Y,np.float32).reshape(-1,C)[:,0])}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=E/'V5_H3_VISUAL/tables');ap.add_argument('--only',nargs='*');a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 sets={}
 sets['rv_train']=[('rv',f'{i:03d}',r,None) for i in range(1,81) for r in RATIOS]
 sets['rv_dev']=[('rv',f'{i:03d}',r,None) for i in range(81,101) for r in RATIOS]
 sets['rv_confirm2']=[('rv',str(i),r,None) for i in range(601,701) for r in RATIOS]
 for sub,name in (('train','live_train'),('dev','live_dev')):
  sets[name]=[('live',r['video_id'],'t',{**r,'_sub':sub}) for r in live_rows(sub)]
 sets['live_val']=[('live',r['video_id'],'t',{**r,'_sub':'val'}) for r in live_rows('val')]
 for name,units in sets.items():
  if a.only and name not in a.only:continue
  out=[u for u in (load_unit(*x) for x in units) if u is not None]
  with open(a.output/f'{name}.pkl','wb') as f:pickle.dump(out,f)
  nk=sum(len(u['J']) for u in out);ng=int(sum(u['has_gt'].sum() for u in out))
  print(json.dumps({'table':name,'units':len(out),'keyframes':nk,'keyframes_with_gt':ng,
   'axis0':int(sum(u['comp']==0 for u in out)),'axis1':int(sum(u['comp']==1 for u in out))}),flush=True)
if __name__=='__main__':main()
