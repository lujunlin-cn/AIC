#!/usr/bin/env python3
"""R91: <=100M frozen ResNet temporal ranker + audited B3 spatial boxes.

The temporal ranker is fit only on the public TVSum train feature caches.  On
the official drop it reads keyframe RGB, scores with a frozen headless
ResNet18 plus a linear continuous-importance readout, interpolates scores onto
the original frame axis, and selects exactly the per-video count of R54.  The
spatial B3 full-frame artifact is copied byte-for-byte.  No TimeLens or
official labels are used in this candidate.
"""
from __future__ import annotations
import argparse, hashlib, json, tempfile, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def sha(p):
    h=hashlib.sha256();
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def fit_static(manifest):
    xs=[]; ys=[]; ns=[]
    for line in Path(manifest).read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); z=np.load(r['path'],allow_pickle=False); m=z['mask'].astype(bool); xs.append(z['features'][m].astype(np.float32)); ys.append(z['labels'][m].astype(np.float32)); ns.append(str(r.get('video_id',Path(r['path']).stem)))
    X=np.concatenate(xs); y=np.concatenate(ys); mu=X.mean(0); sd=X.std(0); sd[sd<1e-5]=1.; Z=(X-mu)/sd; A=np.concatenate([Z,np.ones((len(Z),1),np.float32)],1)
    counts=np.concatenate([np.full(len(x),len(x),np.float32) for x in xs]); w=1./np.maximum(counts,1.)
    G=(A.T*w)@A; G[:-1,:-1]+=10*np.eye(G.shape[1]-1); beta=np.linalg.solve(G,(A.T*w)@y).astype(np.float32)
    return beta,mu.astype(np.float32),sd.astype(np.float32),sum(len(x) for x in xs),len(xs)


def letterbox(path,size=224):
    from PIL import Image
    im=Image.open(path).convert('RGB'); W,H=im.size; scale=min(size/W,size/H); rw=max(1,round(W*scale)); rh=max(1,round(H*scale)); im=im.resize((rw,rh),Image.Resampling.BILINEAR); canvas=Image.new('RGB',(size,size),(124,116,104)); canvas.paste(im,((size-rw)//2,(size-rh)//2)); return np.asarray(canvas)


def infer_resnet(frame_paths, backbone, device, batch=64):
    import torch
    from torchvision.models import resnet18
    out=[]; mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1); std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    # Decode in parallel batches so the GPU sees a steady stream.  Keeping
    # only one batch in memory avoids the 8k-image CPU RAM spike.
    with ThreadPoolExecutor(max_workers=8) as pool:
      for i in range(0,len(frame_paths),batch):
        arr=list(pool.map(letterbox, frame_paths[i:i+batch]))
        a=torch.from_numpy(np.stack(arr)).permute(0,3,1,2).contiguous().float().to(device,non_blocking=True)/255.; a=(a-mean)/std
        with torch.inference_mode(): out.append(backbone(a).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tvsum-manifest',type=Path,required=True); ap.add_argument('--backbone-state',type=Path,required=True); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--parent',type=Path,required=True); ap.add_argument('--all-b3',type=Path,required=True); ap.add_argument('--points',type=Path,required=True); ap.add_argument('--frames',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch',type=int,default=64); args=ap.parse_args()
    import torch
    from torchvision.models import resnet18
    beta,mu,sd,train_rows,train_videos=fit_static(args.tvsum_manifest)
    m=resnet18(weights=None); m.fc=torch.nn.Identity(); state=torch.load(args.backbone_state,map_location='cpu',weights_only=True); state=state.get('state_dict',state); state={k.removeprefix('backbone.'):v for k,v in state.items()}; m.load_state_dict(state,strict=False); m.to(args.device).eval()
    index={str(json.loads(x)['video_id']):json.loads(x) for x in args.index.read_text().splitlines() if x.strip()}; parent={str(x['video_id']):x for x in map(json.loads,args.parent.read_text().splitlines())}; full={str(x['video_id']):x for x in map(json.loads,args.all_b3.read_text().splitlines())}; out=[]; detail=[]
    for vid in sorted(index,key=lambda x:int(x) if x.isdigit() else x):
        p=json.loads((args.points/f'{vid}.json').read_text()); kfs=[int(x) for x in p['keyframes']]; paths=[args.frames/vid/f'{k}.png' for k in kfs]; feat=infer_resnet(paths,m,args.device,args.batch); scores=((feat-mu)/sd)@beta[:-1]+beta[-1]; frames=np.asarray([int(q['frame']) for q in full[vid]['predictions']],np.int64); vals=np.interp(frames,np.asarray(kfs),scores); k=len(parent[vid]['predictions']); idx=np.argsort(-vals,kind='mergesort')[:k]; chosen=np.sort(frames[idx]); by={int(q['frame']):q for q in full[vid]['predictions']}; preds=[by[int(f)] for f in chosen]; out.append({'video_id':vid,'targetRatioWH':index[vid]['targetRatioWH'],'model_size_mb':98.584,'model_params_b':0.098584754,'predictions':preds}); detail.append({'video_id':vid,'all_frames':len(frames),'kept':len(preds),'keyframes':len(kfs),'score_min':float(scores.min()),'score_max':float(scores.max())})
    args.output.mkdir(parents=True,exist_ok=False); dest=args.output/'predictions.jsonl'; dest.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in out)); zp=args.output/(args.output.name+'.zip');
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(dest,'predictions.jsonl')
    man={'submission_id':args.output.name,'status':'PACKAGED_NOT_UPLOADED','official_platform_score':None,'mechanism':'<=100M route: frozen headless ResNet18 static semantic temporal ranker fit on public TVSum continuous importance; exact R54 per-video counts; audited V9/B3 full-frame boxes copied unchanged','temporal_training_manifest':str(args.tvsum_manifest),'train_rows':train_rows,'train_videos':train_videos,'spatial_parent':str(args.all_b3),'official_test_labels_used':False,'video_count':len(out),'prediction_count':sum(len(x['predictions']) for x in out),'parameters':[{'name':'ResNet18 headless','parameters':11176512},{'name':'linear temporal readout','parameters':513},{'name':'V9 B3 vision + head + YuNet','parameters':87407729}],'total_parameters':98584754,'model_params_b':0.098584754,'prediction_sha256':sha(dest),'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'per_video':detail}
    (args.output/'manifest.json').write_text(json.dumps(man,indent=2)+'\n'); (args.output/(zp.name+'.sha256')).write_text(sha(zp)+'  '+zp.name+'\n'); print(json.dumps({k:v for k,v in man.items() if k!='per_video'},indent=2))

if __name__=='__main__': main()
