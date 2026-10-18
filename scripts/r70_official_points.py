#!/usr/bin/env python3
"""Run the R70 native-crop utility head on the 426 official videos.

Official videos are used only for legal inference.  The head was trained on
public native crop sources and is a new R70 readout; this script never reads
official labels.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--vision-dir',type=Path,required=True); ap.add_argument('--processor-dir',type=Path,required=True)
ap.add_argument('--head',type=Path,required=True); ap.add_argument('--index',type=Path,required=True)
ap.add_argument('--keyframe-src',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--device',default='cuda'); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
ap.add_argument('--batch',type=int,default=8)
args=ap.parse_args()

sys.path.insert(0,str(Path(__file__).resolve().parent))
from r70_spatial_utility_pilot import geometry, feature_from_grid, load_vision, Head
import torch
from PIL import Image

torch_mod, vision, proc = load_vision(args)
ck=torch.load(args.head,map_location='cpu',weights_only=False)
head=Head.make(torch_mod).to(args.device).float().eval(); head.load_state_dict(ck['state_dict'])
mu=np.asarray(ck['mu'],np.float32); sd=np.asarray(ck['sd'],np.float32)
rows=[json.loads(x) for x in args.index.read_text().splitlines() if x.strip()]
rows=rows[args.shard::args.nshards]
out=args.output_dir/'points'; out.mkdir(parents=True,exist_ok=True)
src_points=args.keyframe_src/'points'; src_frames=args.keyframe_src/'keyframes'
t0=time.time(); done=0; fallback=0
for rec in rows:
    vid=str(rec['video_id']); dst=out/f'{vid}.json'
    if dst.exists(): continue
    meta=json.loads((src_points/f'{vid}.json').read_text())
    W,H=int(meta['W']),int(meta['H']); ratio=list(map(float,rec['targetRatioWH']))
    pts=[]; status=[]
    for start in range(0,len(meta['keyframes']),args.batch):
        kbatch=meta['keyframes'][start:start+args.batch]
        try:
            imgs=[Image.open(src_frames/vid/f'{kf}.png').convert('RGB') for kf in kbatch]
            x=proc(images=imgs,return_tensors='pt')
            x={k:v.to(args.device) for k,v in x.items() if torch.is_tensor(v)}
            with torch.no_grad():
                o=vision(pixel_values=x['pixel_values'].half(),spatial_shapes=x['spatial_shapes'],pixel_attention_mask=x['pixel_attention_mask'],return_dict=True)
            for bi in range(len(kbatch)):
                valid=int(x['pixel_attention_mask'][bi].sum()); fh,fw=[int(v) for v in x['spatial_shapes'][bi]]
                grid=o.last_hidden_state[bi,:valid].float().detach().cpu().numpy().reshape(fh,fw,-1)
                grid=grid[:(fh//2)*2,:(fw//2)*2]
                feat,offs=feature_from_grid(grid,W,H,ratio)
                z=torch.from_numpy((feat.astype(np.float32)-mu)/sd).unsqueeze(0).to(args.device)
                with torch.no_grad(): j=int(head(z).squeeze(0).argmax().item())
                w,h,axis=geometry(W,H,ratio)
                if axis==0: cx,cy=(float(offs[j]+w/2)/W),.5
                elif axis==1: cx,cy=.5,float(offs[j]+h/2)/H
                else: cx=cy=.5
                pts.append([round(cx,6),round(cy,6),False]); status.append('ok_r70')
        except Exception as e:
            for _ in kbatch: pts.append([.5,.5,True]); status.append('fallback:'+type(e).__name__); fallback+=len(kbatch)
    dst.write_text(json.dumps({'video_id':vid,'W':W,'H':H,'fps':meta['fps'],'step':meta['step'],'keyframes':meta['keyframes'],'model':'R70 V9 vision-only tower + public native-crop utility head','prompt':'NONE_HEAD','ratios':{'t':{'ratio':rec['targetRatioWH'],'points':pts,'status':status,'raw':[],'boxes':[]}}},ensure_ascii=False)+'\n')
    done+=1
    if done%10==0: print(json.dumps({'shard':args.shard,'done':done,'total':len(rows),'fallback':fallback,'elapsed_s':round(time.time()-t0,1)}),flush=True)
print(json.dumps({'shard':args.shard,'done':done,'fallback':fallback,'elapsed_s':round(time.time()-t0,1)}),flush=True)
