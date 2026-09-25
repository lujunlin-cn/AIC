#!/usr/bin/env python3
"""Extract frozen torchvision ViT-B/16 frame embeddings for B0 pilot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from torchvision.models import vit_b_16
from aic.features import extract_video_cache

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--sample-fps',type=float,default=2.0); a=ap.parse_args()
    net=vit_b_16(weights=None); state=torch.load(a.weights,map_location='cpu',weights_only=True); net.load_state_dict(state,strict=True); net.heads=nn.Identity(); net.eval().to(a.device)
    records=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); rows=[]
    for r in records:
        p=out/f"{r['video_id']}.npz"; source=r.get('video_path',r.get('path')); res=extract_video_cache(net,source,p,r.get('labels_path'),device=a.device,sample_fps=a.sample_fps,batch_size=16,metadata={'video_id':r['video_id'],'split':r.get('split'),'source_group':r.get('source_group'),'backbone':'torchvision_vit_b_16_imagenet1k_v1','feature_dim':768,'feature_protocol':'letterbox224_imagenet_norm'})
        q=dict(r); q.update({'path':str(p),'feature_dim':768,'backbone':'vit_b_16'}); rows.append(q); print(r['video_id'],res,flush=True)
    Path(str(out.parent)+f'/{Path(a.manifest).stem}_features.jsonl').write_text(''.join(json.dumps(q,sort_keys=True)+'\n' for q in rows))
if __name__=='__main__': main()
