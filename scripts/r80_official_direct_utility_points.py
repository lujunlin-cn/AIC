#!/usr/bin/env python3
"""Official keyframe inference for direct V9 deployed-box utility.

Frozen V9/B3 candidate head selects the deployed legal crop. A small direct
utility head predicts the native IoU of that selected crop. Official videos
are used only for inference; no labels are read.
"""
from __future__ import annotations
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(); ap.add_argument('--vision-dir',type=Path,required=True); ap.add_argument('--processor-dir',type=Path,required=True); ap.add_argument('--v9-head',type=Path,required=True); ap.add_argument('--utility-head',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--keyframe-src',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--prefetch-workers',type=int,default=1); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parent)); from r70_spatial_utility_pilot import NC,candidate_offsets,load_vision,Head
import torch
from PIL import Image

torch_mod,vision,proc=load_vision(args)
v9ck=torch.load(args.v9_head,map_location='cpu',weights_only=False); v9=Head.make(torch_mod).to(args.device).float().eval(); v9.load_state_dict(v9ck['state_dict'])
class UtilityHead(torch_mod.nn.Module):
 def __init__(self,d):
  super().__init__(); self.net=torch_mod.nn.Sequential(torch_mod.nn.Linear(d,256),torch_mod.nn.GELU(),torch_mod.nn.Linear(256,64),torch_mod.nn.GELU(),torch_mod.nn.Linear(64,1))
 def forward(self,x): return self.net(x).squeeze(-1)
uck=torch.load(args.utility_head,map_location='cpu',weights_only=False); utility=UtilityHead(int(uck['config']['input_dim'])).to(args.device).float().eval(); utility.load_state_dict(uck['state_dict'])
mu=torch.as_tensor(np.asarray(uck['mu'],np.float32),device=args.device).view(1,-1); sd=torch.as_tensor(np.asarray(uck['sd'],np.float32),device=args.device).view(1,-1)
rows=[json.loads(x) for x in args.index.read_text().splitlines() if x.strip()][args.shard::args.nshards];
if args.limit: rows=rows[:args.limit]
out=args.output_dir/'points';out.mkdir(parents=True,exist_ok=True); src_points=args.keyframe_src/'points';src_frames=args.keyframe_src/'keyframes'

def mask_for(W,H,ratio,fh,fw,device):
 offsets,w,h,axis=candidate_offsets(W,H,ratio); ax=0 if axis is None else axis; px=np.array([W/fw,H/fh],np.float32); gy,gx=np.mgrid[0:fh,0:fw]; m=np.zeros((NC,fh*fw),np.float32)
 for j,o in enumerate(offsets):
  if ax==0:x1,y1,x2,y2=o,0.,o+w,h
  else:x1,y1,x2,y2=0.,o,w,o+h
  cx1=int(np.floor(x1/px[0]));cx2=max(int(np.ceil(x2/px[0])),cx1+1);cy1=int(np.floor(y1/px[1]));cy2=max(int(np.ceil(y2/px[1])),cy1+1);m[j]=((gx>=cx1)&(gx<cx2)&(gy>=cy1)&(gy<cy2)).reshape(-1)
 return torch.from_numpy(m).to(device),offsets,w,h,axis
cache={}
def prep(vid,kbatch):
 ims=[]
 for k in kbatch:
  with Image.open(src_frames/vid/f'{k}.png') as im: ims.append(im.convert('RGB'))
 x=proc(images=ims,return_tensors='pt');
 if torch_mod.is_tensor(x.get('pixel_values')): x['pixel_values']=x['pixel_values'].half()
 return x
def infer(kbatch,xcpu,W,H,ratio):
 x={k:v.to(args.device) for k,v in xcpu.items() if torch_mod.is_tensor(v)}
 with torch_mod.inference_mode():
  o=vision(pixel_values=x['pixel_values'],spatial_shapes=x['spatial_shapes'],pixel_attention_mask=x['pixel_attention_mask'],return_dict=True); counts=x['pixel_attention_mask'].sum(1).tolist(); shapes=[tuple(int(v) for v in sh) for sh in x['spatial_shapes'].tolist()]; groups={}
  for bi,(sh,valid) in enumerate(zip(shapes,counts)):groups.setdefault((sh[0],sh[1],int(valid)),[]).append(bi)
  pts=[None]*len(kbatch); util=[None]*len(kbatch)
  for (fh,fw,valid),bis in groups.items():
   ids=torch_mod.as_tensor(bis,device=o.last_hidden_state.device,dtype=torch_mod.long); grid=o.last_hidden_state.index_select(0,ids)[:,:valid].float().reshape(len(bis),fh,fw,-1);grid=grid[:,: (fh//2)*2,: (fw//2)*2]
   m,offs,w,h,axis=cache.setdefault((W,H,tuple(ratio),fh,fw),mask_for(W,H,ratio,fh,fw,o.last_hidden_state.device)); flat=grid.reshape(len(bis),-1,grid.shape[-1]); win=torch_mod.einsum('nk,bkd->bnd',m,flat)/m.sum(1,keepdim=True).clamp_min(1).view(1,NC,1); inv=1-m;outm=torch_mod.einsum('nk,bkd->bnd',inv,flat)/inv.sum(1,keepdim=True).clamp_min(1).view(1,NC,1);pos=torch_mod.from_numpy((offs/(offs[-1] if offs[-1]>0 else 1)).astype(np.float32)).to(grid.device).view(1,NC,1).expand(len(bis),-1,-1);feat=torch_mod.cat([win,outm,win-outm,pos],-1).half();scores=v9(feat.float());j=scores.argmax(1);srt=torch_mod.sort(scores,dim=1).values;chosen=feat[torch_mod.arange(len(bis),device=feat.device),j];side=torch_mod.stack([scores[torch_mod.arange(len(bis),device=feat.device),j],srt[:,-1]-srt[:,-2],scores[:,NC//2],j.float()/(NC-1),scores.mean(1)],1);direct=utility((torch_mod.cat([chosen.float(),side],1)-mu)/sd)
   for li,bi in enumerate(bis):
    jj=int(j[li]);
    if axis==0:cx,cy=(float(offs[jj]+w/2)/W),.5
    elif axis==1:cx,cy=.5,float(offs[jj]+h/2)/H
    else:cx=cy=.5
    pts[bi]=[round(cx,6),round(cy,6),False];util[bi]={'direct_score':float(direct[li]),'v9_selected_score':float(scores[li,jj]),'v9_top2_gap':float((srt[li,-1]-srt[li,-2])),'selected_index':jj}
 return pts,util

t0=time.time();done=0;fallback=0; executor=ThreadPoolExecutor(max_workers=args.prefetch_workers) if args.prefetch_workers else None
try:
 for rec in rows:
  vid=str(rec['video_id']);dst=out/f'{vid}.json'
  if dst.exists():continue
  meta=json.loads((src_points/f'{vid}.json').read_text());W,H=int(meta['W']),int(meta['H']);ratio=list(map(float,rec['targetRatioWH'])); kfs=meta['keyframes'];pts=[];st=[];us=[];starts=list(range(0,len(kfs),args.batch));future=None
  for si,start in enumerate(starts):
   kb=kfs[start:start+args.batch]
   if executor:
    if future is None:future=executor.submit(prep,vid,kb)
    nf=None
    if si+1<len(starts):nf=executor.submit(prep,vid,kfs[starts[si+1]:starts[si+1]+args.batch])
    try:p,u=infer(kb,future.result(),W,H,ratio);pts.extend(p);us.extend(u);st.extend(['ok_r80']*len(kb))
    except Exception as e:pts.extend([[.5,.5,True]]*len(kb));us.extend([None]*len(kb));st.extend(['fallback:'+type(e).__name__]*len(kb));fallback+=len(kb)
    future=nf
   else:
    try:p,u=infer(kb,prep(vid,kb),W,H,ratio);pts.extend(p);us.extend(u);st.extend(['ok_r80']*len(kb))
    except Exception as e:pts.extend([[.5,.5,True]]*len(kb));us.extend([None]*len(kb));st.extend(['fallback:'+type(e).__name__]*len(kb));fallback+=len(kb)
  dst.write_text(json.dumps({'video_id':vid,'W':W,'H':H,'fps':meta['fps'],'step':meta['step'],'keyframes':kfs,'model':'V9 B3 frozen candidate + direct native-IoU utility head','prompt':'NONE_HEAD','ratios':{'t':{'ratio':rec['targetRatioWH'],'points':pts,'status':st,'raw':[],'boxes':[],'utility':us}}},ensure_ascii=False)+'\n');done+=1
  if done%10==0:print(json.dumps({'done':done,'total':len(rows),'fallback':fallback,'elapsed_s':round(time.time()-t0,1)}),flush=True)
finally:
 if executor:executor.shutdown(wait=True)
print(json.dumps({'done':done,'total':len(rows),'fallback':fallback,'elapsed_s':round(time.time()-t0,1)}),flush=True)
