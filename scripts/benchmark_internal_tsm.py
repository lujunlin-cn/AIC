#!/usr/bin/env python3
"""Small correctness/throughput probe for true ResNet intermediate TSM."""
from __future__ import annotations
import argparse, json, time
import torch
from torchvision.models import resnet18
from aic.models import temporal_shift_feature_map

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cpu'); ap.add_argument('--steps',type=int,default=5); a=ap.parse_args()
    device=torch.device(a.device); net=resnet18(weights=None).to(device).eval()
    x=torch.randn(1,8,3,224,224,device=device)
    def stem(z):
        b,t=z.shape[:2]; z=z.reshape(b*t,*z.shape[2:]); z=net.maxpool(net.relu(net.bn1(net.conv1(z)))); return net.layer1(z).reshape(b,t,64,z.shape[-2],z.shape[-1])
    with torch.inference_mode():
        fmap=stem(x); shifted=temporal_shift_feature_map(fmap); diff=float((fmap-shifted).abs().mean().cpu());
        def run(use_shift):
            z=temporal_shift_feature_map(stem(x)) if use_shift else stem(x); b,t=z.shape[:2]; z=z.reshape(b*t,*z.shape[2:]); z=net.layer2(z); z=net.layer3(z); z=net.layer4(z); return net.avgpool(z).flatten(1).reshape(b,t,-1)
        run(False); run(True)
        t0=time.perf_counter()
        for _ in range(a.steps): run(False)
        base=(time.perf_counter()-t0)/a.steps
        t0=time.perf_counter()
        for _ in range(a.steps): run(True)
        internal=(time.perf_counter()-t0)/a.steps
    out={'device':str(device),'shape':list(fmap.shape),'mean_shift_difference':diff,'baseline_seconds_per_batch':base,'internal_tsm_seconds_per_batch':internal,'relative_overhead':internal/base-1 if base else None,'parameter_free':True,'feature_level_a1_is_not_this':True}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
