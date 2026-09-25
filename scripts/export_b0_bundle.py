#!/usr/bin/env python3
"""Combine the frozen ViT backbone and B0 temporal probe into an inference bundle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torchvision.models import vit_b_16
from torch import nn

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--vit',required=True); ap.add_argument('--temporal',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 net=vit_b_16(weights=None); net.load_state_dict(torch.load(a.vit,map_location='cpu',weights_only=True),strict=True); net.heads=nn.Identity()
 st=torch.load(a.temporal,map_location='cpu',weights_only=True)['model']; out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 state={k:v.detach().cpu() for k,v in net.state_dict().items()}; tstate={k:v.detach().cpu() for k,v in st.items()}
 # Persistent FP16 is the candidate format; runtime loader converts to FP32
 # because Volta FP16/CPU kernels are not uniformly safe for every op.
 b={'format_version':1,'architecture':'B0_vit_b16_temporal_probe','persistent_dtype':'fp16','backbone_name':'torchvision_vit_b_16_imagenet1k_v1','backbone_state':{k:v.half() if v.is_floating_point() else v for k,v in state.items()},'temporal_state':{k:v.half() if v.is_floating_point() else v for k,v in tstate.items()},'parameter_count':sum(v.numel() for v in state.values())+sum(v.numel() for v in tstate.values()),'source_temporal_checkpoint':str(a.temporal),'submission_candidate':True,'size_note':'FP16 ViT B/16 + TemporalUNet; M tier under current decimal-size assumption'}
 torch.save(b,out); print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'MB_decimal':out.stat().st_size/1e6,'parameter_count':b['parameter_count']},indent=2))
if __name__=='__main__': main()
