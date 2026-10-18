#!/usr/bin/env python3
"""Run the R70 native-crop utility head on the 426 official videos.

Official videos are used only for legal inference.  The head was trained on
public native crop sources and is a new R70 readout; this script never reads
official labels.
"""
import argparse, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--vision-dir',type=Path,required=True); ap.add_argument('--processor-dir',type=Path,required=True)
ap.add_argument('--head',type=Path,required=True); ap.add_argument('--index',type=Path,required=True)
ap.add_argument('--keyframe-src',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--device',default='cuda'); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
ap.add_argument('--batch',type=int,default=8)
ap.add_argument('--prefetch-workers',type=int,default=1,
                help='CPU image/processor workers; one worker overlaps the next batch with CUDA')
ap.add_argument('--emit-utility',action='store_true',
                help='保存每个关键帧的候选分数、中心分数、top1-top2 margin和候选索引')
args=ap.parse_args()

sys.path.insert(0,str(Path(__file__).resolve().parent))
from r70_spatial_utility_pilot import NC, candidate_offsets, geometry, load_vision, Head
import torch
from PIL import Image

torch_mod, vision, proc = load_vision(args)
ck=torch.load(args.head,map_location='cpu',weights_only=False)
head=Head.make(torch_mod).to(args.device).float().eval(); head.load_state_dict(ck['state_dict'])
mu=np.asarray(ck['mu'],np.float32); sd=np.asarray(ck['sd'],np.float32)
mu_t=torch.as_tensor(mu,device=args.device,dtype=torch.float32).view(1,1,-1)
sd_t=torch.as_tensor(sd,device=args.device,dtype=torch.float32).view(1,1,-1)
rows=[json.loads(x) for x in args.index.read_text().splitlines() if x.strip()]
rows=rows[args.shard::args.nshards]
out=args.output_dir/'points'; out.mkdir(parents=True,exist_ok=True)
src_points=args.keyframe_src/'points'; src_frames=args.keyframe_src/'keyframes'


def _candidate_mask(W, H, ratio, fh, fw, device):
    """Build the exact V9 pooling mask once, on the inference device.

    ``feature_from_grid`` was originally written for the public training
    extractor and constructs this 129 x (fh*fw) mask in a Python loop for
    every frame.  Official inference has ~200k frames, so doing that work on
    CPU also starves the vision tower.  The arithmetic below intentionally
    mirrors that function (including floor/ceil boundary handling) and only
    moves the resulting constant mask to CUDA once per geometry/grid shape.
    """
    offsets, w, h, axis = candidate_offsets(W, H, ratio)
    effective_axis = 0 if axis is None else axis
    px_per_x, px_per_y = W / fw, H / fh
    m = np.zeros((NC, fh * fw), np.float32)
    gy, gx = np.mgrid[0:fh, 0:fw]
    for j, o in enumerate(offsets):
        if effective_axis == 0:
            x1, y1, x2, y2 = o, 0.0, o + w, h
        else:
            x1, y1, x2, y2 = 0.0, o, w, o + h
        cx1 = int(math.floor(x1 / px_per_x))
        cx2 = max(int(math.ceil(x2 / px_per_x)), cx1 + 1)
        cy1 = int(math.floor(y1 / px_per_y))
        cy2 = max(int(math.ceil(y2 / px_per_y)), cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1)
    mt = torch.from_numpy(m).to(device=device)
    return mt, offsets, w, h, axis


def _features_from_grids(grids, W, H, ratio, mask_cache):
    """GPU equivalent of ``feature_from_grid`` for a batch of grids.

    Grids are [B, fh, fw, D].  Returning a single [B, 129, 2305] tensor lets
    the utility head run one batched attention call instead of 129 CPU/Python
    operations and one CUDA launch per frame.
    """
    b, fh, fw, d = grids.shape
    key = (int(W), int(H), tuple(float(v) for v in ratio), int(fh), int(fw), str(grids.device))
    if key not in mask_cache:
        mask_cache[key] = _candidate_mask(W, H, ratio, fh, fw, grids.device)
    m, offsets, w, h, axis = mask_cache[key]
    flat = grids.float().reshape(b, fh * fw, d)
    den = m.sum(1, keepdim=True).clamp_min(1.0)
    win = torch.einsum('nk,bkd->bnd', m, flat) / den.view(1, NC, 1)
    inv = 1.0 - m
    outm = torch.einsum('nk,bkd->bnd', inv, flat) / inv.sum(1, keepdim=True).clamp_min(1.0).view(1, NC, 1)
    pos = np.asarray(offsets, np.float32)
    pos = pos / (pos[-1] if pos[-1] > 0 else 1.0)
    post = torch.from_numpy(pos).to(device=grids.device, dtype=torch.float32).view(1, NC, 1).expand(b, -1, -1)
    # The public extractor stores features as float16 before normalisation.
    # Keep this cast so the new batched path has the same head input contract.
    return torch.cat((win, outm, win - outm, post), dim=-1).half(), offsets, w, h, axis


mask_cache = {}


def _prepare_images(vid, kbatch):
    """Decode and preprocess one batch away from the CUDA thread.

    A single producer is enough to hide most PIL/processor latency while the
    main thread executes the vision tower.  Keeping this bounded to one batch
    avoids the RAM blow-up seen with several full inference processes.
    """
    imgs = []
    for kf in kbatch:
        with Image.open(src_frames / vid / f'{kf}.png') as im:
            imgs.append(im.convert('RGB'))
    x = proc(images=imgs, return_tensors='pt')
    # The tower is loaded in FP16.  Convert before the host-to-device copy so
    # PCIe traffic and the synchronous cast in the CUDA thread are halved.
    if torch.is_tensor(x.get('pixel_values')):
        x['pixel_values'] = x['pixel_values'].half()
    return x


def _infer_batch(kbatch, xcpu, W, H, ratio):
    """Run vision, utility pooling and head for one prepared image batch."""
    x={k:v.to(args.device) for k,v in xcpu.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        o=vision(pixel_values=x['pixel_values'],spatial_shapes=x['spatial_shapes'],pixel_attention_mask=x['pixel_attention_mask'],return_dict=True)
        # A processor batch normally has one spatial shape.  Group by
        # shape/valid length for robustness to an unusual dynamic processor
        # output while keeping every group on the GPU.
        counts=x['pixel_attention_mask'].sum(1).tolist()
        shapes=[tuple(int(v) for v in sh) for sh in x['spatial_shapes'].tolist()]
        groups={}
        for bi,(sh,valid) in enumerate(zip(shapes,counts)):
            groups.setdefault((sh[0],sh[1],int(valid)),[]).append(bi)
        batch_pts=[None]*len(kbatch)
        batch_utility=[None]*len(kbatch)
        for (fh0,fw0,valid), bis in groups.items():
            idx=torch.as_tensor(bis,device=o.last_hidden_state.device,dtype=torch.long)
            grid=o.last_hidden_state.index_select(0,idx)[:,:valid].float().reshape(len(bis),fh0,fw0,-1)
            # V9 contract discards odd token rows/columns before pooling.
            grid=grid[:,: (fh0//2)*2, : (fw0//2)*2]
            feat,offs,w,h,axis=_features_from_grids(grid,W,H,ratio,mask_cache)
            z=(feat.float()-mu_t) / sd_t
            scores=head(z)
            j=scores.argmax(1).detach().cpu().tolist()
            for local,bi in enumerate(bis):
                jj=j[local]
                if axis==0: cx,cy=(float(offs[jj]+w/2)/W),.5
                elif axis==1: cx,cy=.5,float(offs[jj]+h/2)/H
                else: cx=cy=.5
                batch_pts[bi]=[round(cx,6),round(cy,6),False]
                if args.emit_utility:
                    row=scores[local]
                    top2=torch.topk(row,k=2,dim=0).values
                    batch_utility[bi]={
                        'top_score':float(row[jj].detach().cpu()),
                        'center_score':float(row[NC//2].detach().cpu()),
                        'margin':float((top2[0]-top2[1]).detach().cpu()),
                        'chosen_index':int(jj),
                    }
    return (batch_pts,batch_utility) if args.emit_utility else batch_pts


t0=time.time(); done=0; fallback=0
executor = ThreadPoolExecutor(max_workers=args.prefetch_workers) if args.prefetch_workers > 0 else None
try:
  for rec in rows:
    vid=str(rec['video_id']); dst=out/f'{vid}.json'
    if dst.exists(): continue
    meta=json.loads((src_points/f'{vid}.json').read_text())
    W,H=int(meta['W']),int(meta['H']); ratio=list(map(float,rec['targetRatioWH']))
    pts=[]; status=[]; utility=[]
    starts=list(range(0,len(meta['keyframes']),args.batch))
    future=None
    for si,start in enumerate(starts):
        kbatch=meta['keyframes'][start:start+args.batch]
        # Submit one look-ahead batch.  This is deliberately bounded: several
        # inference processes each running processor workers exhaust CPU/RAM.
        if executor is not None:
            if future is None:
                future=executor.submit(_prepare_images,vid,kbatch)
            next_future=None
            if si+1 < len(starts):
                nk=meta['keyframes'][starts[si+1]:starts[si+1]+args.batch]
                next_future=executor.submit(_prepare_images,vid,nk)
            try:
                xcpu=future.result()
                inferred=_infer_batch(kbatch,xcpu,W,H,ratio)
                if args.emit_utility:
                    batch_pts,batch_utility=inferred
                    utility.extend(batch_utility)
                else:
                    batch_pts=inferred
                pts.extend(batch_pts); status.extend(['ok_r70']*len(kbatch))
            except Exception as e:
                for _ in kbatch:
                    pts.append([.5,.5,True]); status.append('fallback:'+type(e).__name__)
                    if args.emit_utility: utility.append(None)
                fallback+=len(kbatch)
            future=next_future
        else:
            try:
                xcpu=_prepare_images(vid,kbatch)
                inferred=_infer_batch(kbatch,xcpu,W,H,ratio)
                if args.emit_utility:
                    batch_pts,batch_utility=inferred
                    utility.extend(batch_utility)
                else:
                    batch_pts=inferred
                pts.extend(batch_pts); status.extend(['ok_r70']*len(kbatch))
            except Exception as e:
                for _ in kbatch:
                    pts.append([.5,.5,True]); status.append('fallback:'+type(e).__name__)
                    if args.emit_utility: utility.append(None)
                fallback+=len(kbatch)
    t_payload={'ratio':rec['targetRatioWH'],'points':pts,'status':status,'raw':[],'boxes':[]}
    if args.emit_utility:
        t_payload['utility']=utility
    dst.write_text(json.dumps({'video_id':vid,'W':W,'H':H,'fps':meta['fps'],'step':meta['step'],'keyframes':meta['keyframes'],'model':'R70 V9 vision-only tower + public native-crop utility head','prompt':'NONE_HEAD','ratios':{'t':t_payload}},ensure_ascii=False)+'\n')
    done+=1
    if done%10==0: print(json.dumps({'shard':args.shard,'done':done,'total':len(rows),'fallback':fallback,'elapsed_s':round(time.time()-t0,1)}),flush=True)
finally:
  if executor is not None: executor.shutdown(wait=True)
print(json.dumps({'shard':args.shard,'done':done,'fallback':fallback,'elapsed_s':round(time.time()-t0,1)}),flush=True)
