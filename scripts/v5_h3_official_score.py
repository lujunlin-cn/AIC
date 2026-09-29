"""V5 H3 official scoring: apply a trained H3 head to the 174 official videos.

Reuses the INTERP frozen config (obs cache, d2 points, alpha) and the H3 table
conventions (33 constant-grid candidates, mother snap recorded).  Produces a
rerank-point directory in the same format v5_p0_splice_spatial.py consumes, so
the P0 identity guards (G1 dense recompute, G2 x-only, G3 non-x unchanged) all
carry over unchanged.  This round packages the V head only when kind=V: G has
no visual input and VQ needs the official 26-feature table, which is not built.
Stages: tables -> features -> score.
"""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np

NC=33


def candidate_offsets(p,W,H,ratio,comp):
 from aic.max_window_path import geometry,centre_to_offset
 w,h,_=geometry(W,H,ratio);L,s=(W,w) if comp==0 else (H,h)
 grid=np.linspace(0.,L-s,NC);m=float(centre_to_offset(np.array([p[comp]]),W,H,ratio)[0])
 j=int(np.argmin(np.abs(grid-m)))
 return grid.copy(),L,s,m,j,abs(grid[j]-m)


def build_units(cfg,index_path):
 from aic.contract import load_index,load_jsonl
 from aic.max_window_path import geometry
 fz=json.loads(Path(cfg).read_text());cache=Path(fz['obs_cache']);dense=Path(fz['qwen_points'])
 index=load_index(index_path);records=load_jsonl(index_path)
 units=[]
 for r in records:
  vid=r['video_id'];meta=index[vid];z=np.load(cache/f'{vid}_t.npz')
  W,H=int(z['W']),int(z['H']);ratio=r['targetRatioWH'];w,h,axis=geometry(W,H,ratio)
  if axis is None:continue
  comp=0 if axis==0 else 1
  q=json.loads((dense/f'{vid}.json').read_text());keys=q['keyframes']
  pts=[None if p is None else list(p) for p in q['ratios']['t']['points']]
  J=[k for k,p in enumerate(pts) if p is not None and not p[2]]
  if not J:continue
  offs=np.zeros((len(J),NC),np.float32);moids=[];gaps=[]
  for i,k in enumerate(J):
   g,L,s,m,j,gap=candidate_offsets(pts[k],W,H,ratio,comp);offs[i]=g;moids.append(j);gaps.append(gap/s)
  units.append({'vid':vid,'ds':'off','comp':comp,'W':W,'H':H,'ratio':ratio,'keys':keys,'J':J,'pts':pts,
   'offs':offs,'s':s,'L':L,'frames_dir':'/data/aic/experiments/QWEN_SUBJECT_POINT_OFFICIAL_V1/keyframes',
   'reset':z['reset'].astype(bool),'raw':z['raw'][:,comp],'face':z['chosen']>=0,'gf':np.arange(int(z['b0'].shape[0])),
   'gt':None,'Y':np.full((len(J),NC),np.nan,np.float32),'has_gt':np.zeros((len(J),NC),bool),
   'is_mother':(np.arange(NC)[None,:]==np.array(moids)[:,None]),'mo_gap_max':float(np.max(gaps)),
   'frame_count':meta.frame_count,'alpha':fz['alpha']})
 return units


def stage_features(units,out,cuda,batch=24,workers=12):
 import torch
 from multiprocessing import Pool
 from scripts.v5_h3_dinov2_cache import letterbox_meta,valid_patch_mask,pool_unit,_prep
 from transformers import Dinov2Model
 from PIL import Image
 dev=cuda if torch.cuda.is_available() else 'cpu';out.mkdir(parents=True,exist_ok=True)
 model=Dinov2Model.from_pretrained('/data/aic/pretrained/dinov2_vitb14',torch_dtype=torch.float16).to(dev).eval()
 mean=torch.tensor([.485,.456,.406],device=dev).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=dev).view(1,3,1,1)
 GRID=37;D=518;PATCH=14;pool=Pool(workers);t0=time.time()
 for ui,u in enumerate(units):
  dest=out/f'off_{ui:04d}.npz'
  if dest.exists():continue
  frdir=Path(u['frames_dir']);png_wh=Image.open(frdir/u['vid']/f'{u["keys"][0]}.png').size
  meta=letterbox_meta(png_wh[0],png_wh[1]);valid=valid_patch_mask(meta)
  patches=np.zeros((len(u['J']),GRID,GRID,768),np.float16)
  with torch.no_grad():
   for b0 in range(0,len(u['J']),batch):
    part=u['J'][b0:b0+batch]
    imgs=pool.map(_prep,[(str(frdir/u['vid']/f'{u["keys"][k]}.png'),meta['resized'],meta['ox'],meta['oy']) for k in part])
    xb=torch.from_numpy(np.stack(imgs)).to(dev).float().permute(0,3,1,2)/255.
    xb=((xb-mean)/std).half()
    o=model(xb,interpolate_pos_encoding=True).last_hidden_state[:,1:,:]
    patches[b0:b0+len(o)]=o.reshape(-1,GRID,GRID,768).float().cpu().numpy().astype(np.float16)
  pooled=pool_unit(patches,valid,u,png_wh)
  np.savez(dest,pooled=pooled,rows=np.arange(len(u['J'])),meta=json.dumps({'png_wh':png_wh}))
 print(json.dumps({'features':len(units),'s':round(time.time()-t0,1)}),flush=True)
 pool.close();pool.join()


def stage_score(units,feat,heads_pt,kind,rerank_out,cuda):
 import torch
 from scripts.v5_h3_visual_scorer import unit_tensors,make_head
 dev=cuda if torch.cuda.is_available() else 'cpu'
 H_=torch.load(heads_pt,map_location=dev)
 net,proj=make_head(kind,dev);net.load_state_dict(H_[kind]['net']);net.eval()
 if H_[kind]['proj'] is not None:proj.load_state_dict(H_[kind]['proj']);proj.eval()
 mu=H_[kind]['mu'].to(dev);sd=H_[kind]['sd'].to(dev)
 rerank_out.mkdir(parents=True,exist_ok=True);n=0
 for ui,u in enumerate(units):
  z=np.load(feat/f'off_{ui:04d}.npz');K=len(u['J']);V=np.zeros((K,u['offs'].shape[1],12,768),np.float16)
  rows=z['rows'];V[rows]=z['pooled'];u['V']=V.reshape(K,u['offs'].shape[1],-1)
  x,_=unit_tensors(u,dev,mu,sd,proj,kind)
  with torch.no_grad():s=net(x).squeeze(-1).cpu().numpy()
  pts=[list(p) if p is not None else None for p in u['pts']]
  for i,k in enumerate(u['J']):pts[k][u['comp']]=float((u['offs'][i,int(s[i].argmax())]+u['s']/2)/u['L'])
  (rerank_out/f'{u["vid"]}.json').write_text(json.dumps({'keyframes':u['keys'],'ratios':{'t':{'points':pts}}})+'\n')
  n+=1
 print(json.dumps({'scored_videos':n,'kind':kind}),flush=True)


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--index',default='/data/aic/official_test_20260926/index.enriched.jsonl')
 ap.add_argument('--combo-config',default='configs/QWEN32B_DT_INTERP_V4.json')
 ap.add_argument('--heads',required=True);ap.add_argument('--kind',default='V',choices=('V',))
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cuda',default='cuda:2')
 ap.add_argument('--stage',required=True,choices=('tables','features','score'))
 a=ap.parse_args();(a.out).mkdir(parents=True,exist_ok=True)
 if a.stage=='tables':
  u=build_units(a.combo_config,a.index);pickle.dump(u,open(a.out/'off_units.pkl','wb'))
  print(json.dumps({'units':len(u),'kf':sum(len(x['J']) for x in u)}))
 elif a.stage=='features':
  stage_features(pickle.load(open(a.out/'off_units.pkl','rb')),a.out/'features',a.cuda)
 else:
  stage_score(pickle.load(open(a.out/'off_units.pkl','rb')),a.out/'features',a.heads,a.kind,a.out/'rerank',a.cuda)
if __name__=='__main__':main()
