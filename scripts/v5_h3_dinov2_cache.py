"""V5 H3 step 2: frozen DINOv2 ViT-B/14 patch features + pooled candidate features.

Letterbox to 518x518 preserving aspect ratio (explicit valid-pixel patch mask;
no centre crop).  Per unit keyframe the 37x37x768 patch grid is computed once
(fp16, frozen encoder); per candidate window the pooled features are 3x3
in-window block means + in-window mean + out-of-window mean + boundary ring mean
(12x768 fp16).  Candidates/keyframe order come from the step-1 tables.  Table
offsets are in source pixels; keyframe PNGs may be downscaled (LIVE 640 long
side), so source rects are mapped to PNG pixels first and the mapping recorded.
"""
import argparse,pickle,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation

D=518;PATCH=14;GRID=D//PATCH


def letterbox_meta(pw,ph):
 sc=min(D/pw,D/ph);w2,h2=int(round(pw*sc)),int(round(ph*sc));ox,oy=(D-w2)//2,(D-h2)//2
 return {'scale':sc,'ox':ox,'oy':oy,'resized':[w2,h2]}


def valid_patch_mask(meta):
 w2,h2,ox,oy=meta['resized'][0],meta['resized'][1],meta['ox'],meta['oy']
 xs=(np.arange(GRID)+.5)*PATCH;ys=(np.arange(GRID)+.5)*PATCH
 cx=(xs[None,:]>ox)&(xs[None,:]<ox+w2);cy=(ys[:,None]>oy)&(ys[:,None]<oy+h2)
 return cx&cy


def block_means(P,m,i0,i1,j0,j1):
 js=np.array_split(np.arange(j0,j1),3);is_=np.array_split(np.arange(i0,i1),3)
 out=np.zeros((3,3,768),np.float32)
 for r,jr in enumerate(js):
  for c,ir in enumerate(is_):
   sel=np.zeros_like(m);sel[np.ix_(jr,ir)]=True;sel&=m
   out[r,c]=P[sel].mean(0) if sel.sum() else 0.
 return out


def pool_unit(patches,valid,u,png_wh):
 K,C=patches.shape[0],u['offs'].shape[1];out=np.zeros((K,C,12,768),np.float16)
 sx,sy=png_wh[0]/u['W'],png_wh[1]/u['H']
 meta=letterbox_meta(png_wh[0],png_wh[1]);sc,ox,oy=meta['scale'],meta['ox'],meta['oy']
 cx=(np.arange(GRID)+.5)*PATCH;cy=(np.arange(GRID)+.5)*PATCH
 s=u['s'];comp=u['comp'];Lpx=(png_wh[0] if comp==0 else png_wh[1])
 for k in range(K):
  P=patches[k].astype(np.float32)
  for c in range(C):
   off=u['offs'][k,c]
   if comp==0:wx0,wy0,wx1,wy1=off*sx,0.,(off+s)*sx,float(png_wh[1])
   else:wx0,wy0,wx1,wy1=0.,off*sy,float(png_wh[0]),(off+s)*sy
   lx0,ly0,lx1,ly1=wx0*sc+ox,wy0*sc+oy,wx1*sc+ox,wy1*sc+oy
   i0,i1=int(np.floor(lx0/PATCH)),int(min(GRID,max(np.ceil(lx1/PATCH),np.floor(lx0/PATCH)+1)))
   j0,j1=int(np.floor(ly0/PATCH)),int(min(GRID,max(np.ceil(ly1/PATCH),np.floor(ly0/PATCH)+1)))
   m=np.zeros((GRID,GRID),bool);m[j0:j1,i0:i1]=True;m&=valid
   if m.sum()==0:  # tiny window at a border: nearest valid patch to the window centre
    ctr=np.array([(lx0+lx1)/2,(ly0+ly1)/2]);d=(cx[None,:]-ctr[0])**2+(cy[:,None]-ctr[1])**2
    m=np.zeros_like(valid);m[np.unravel_index(int(np.argmin(np.where(valid,d,1e18))),m.shape)]=True
   win=P[m].mean(0)
   out[k,c,:9]=block_means(P,m,i0,i1,j0,j1).reshape(9,768)
   out[k,c,9]=win
   om=valid&~m
   out[k,c,10]=P[om].mean(0) if om.sum() else 0.
   ring=binary_dilation(m,iterations=2)&~m&valid
   out[k,c,11]=P[ring].mean(0) if ring.sum() else 0.
 return out


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tables',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch',type=int,default=24);ap.add_argument('--only',nargs='*')
 ap.add_argument('--cuda',default='cuda:2');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 import torch
 from transformers import Dinov2Model
 from PIL import Image
 dev=a.cuda if torch.cuda.is_available() else 'cpu'
 model=Dinov2Model.from_pretrained(a.weights,torch_dtype=torch.float16).to(dev).eval()
 for p in model.parameters():p.requires_grad_(False)
 mean=torch.tensor([.485,.456,.406],device=dev).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=dev).view(1,3,1,1)
 names=a.only or ['rv_train','rv_dev','rv_confirm2','live_train','live_dev','live_val']
 t0=time.time()
 for name in names:
  units=pickle.loads((a.tables/f'{name}.pkl').read_bytes());done=0
  for ui,u in enumerate(units):
   dest=a.output/f'{name}_{ui:04d}.npz'
   if dest.exists():continue
   frdir=Path(u['frames_dir']);rows=[j for j in range(len(u['J'])) if u['has_gt'][j]] if name.endswith('train') else list(range(len(u['J'])))
   if not rows:
    np.savez_compressed(dest,pooled=np.zeros((0,u['offs'].shape[1],12,768),np.float16),rows=np.array([],int),meta=json.dumps({}));continue
   im0=Image.open(frdir/u['vid']/f'{u["keys"][rows[0]]}.png');png_wh=im0.size
   meta=letterbox_meta(png_wh[0],png_wh[1]);valid=valid_patch_mask(meta)
   patches=np.zeros((len(rows),GRID,GRID,768),np.float16)
   with torch.no_grad():
    for b0 in range(0,len(rows),a.batch):
     part=rows[b0:b0+a.batch];imgs=[]
     for j in part:
      im=Image.open(frdir/u['vid']/f'{u["keys"][j]}.png').convert('RGB').resize(tuple(meta['resized']),Image.BILINEAR)
      canvas=Image.new('RGB',(D,D),(0,0,0));canvas.paste(im,(meta['ox'],meta['oy']))
      imgs.append(np.asarray(canvas,np.uint8))
     xb=torch.from_numpy(np.stack(imgs)).to(dev).float().permute(0,3,1,2)/255.
     xb=((xb-mean)/std).half()
     o=model(xb,interpolate_pos_encoding=True).last_hidden_state[:,1:,:]
     patches[b0:b0+len(o)]=o.reshape(-1,GRID,GRID,768).float().cpu().numpy().astype(np.float16)
   pooled=pool_unit(patches,valid,u,png_wh)
   np.savez_compressed(dest,pooled=pooled,rows=np.array(rows,int),
    meta=json.dumps({'letterbox':meta,'png_wh':png_wh,'valid':valid.tolist(),'encoder':'hf:dinov2_vitb14(facebook/dinov2-base)','sha256':'d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841','grid':GRID}))
   done+=1
  print(json.dumps({'table':name,'units':len(units),'cached_now':done,'s':round(time.time()-t0,1)}),flush=True)
if __name__=='__main__':main()
