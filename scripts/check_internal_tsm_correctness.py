#!/usr/bin/env python3
"""Check chunked intermediate TSM against a full-sequence reference."""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch
from torchvision.models import resnet18
from aic.video import iter_sampled_frames
from aic.features import _letterbox
from aic.models import temporal_shift_feature_map
from scripts.extract_internal_tsm_features import load_backbone, encode_internal_tsm

@torch.inference_mode()
def baseline(net, images, dev, chunk=128):
    outs=[]
    for start in range(0,len(images),chunk):
        x=_letterbox(images[start:start+chunk]).to(dev); z=net.maxpool(net.relu(net.bn1(net.conv1(x)))); z=net.layer1(z); z=net.layer2(z); z=net.layer3(z); z=net.layer4(z); outs.append(net.avgpool(z).flatten(1).cpu().numpy())
    return np.concatenate(outs)

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--video',required=True); ap.add_argument('--backbone-state',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--sample-fps',type=float,default=2.); ap.add_argument('--chunk-size',type=int,default=128); ap.add_argument('--output',required=True)
    a=ap.parse_args(argv); dev=torch.device(a.device); net=load_backbone(a.backbone_state,dev); sampled=list(iter_sampled_frames(a.video,a.sample_fps,224)); images=np.stack([x[2] for x in sampled])
    t=time.perf_counter(); chunked=encode_internal_tsm(net,images,dev,a.chunk_size); tchunk=time.perf_counter()-t
    t=time.perf_counter(); full=encode_internal_tsm(net,images,dev,len(images)+1); tfull=time.perf_counter()-t
    t=time.perf_counter(); base=baseline(net,images,dev,a.chunk_size); tbase=time.perf_counter()-t
    out={'video':a.video,'frames':len(images),'chunk_size':a.chunk_size,'chunk_vs_full_max_abs':float(np.max(np.abs(chunked-full))),'chunk_vs_full_mean_abs':float(np.mean(np.abs(chunked-full))),'baseline_vs_internal_max_abs':float(np.max(np.abs(base-chunked))),'baseline_vs_internal_mean_abs':float(np.mean(np.abs(base-chunked))),'chunk_seconds':tchunk,'full_seconds':tfull,'baseline_seconds':tbase,'parameter_free':True,'encoder_variant':'canonical_internal_tsm_layer1_fold8_v1'}
    open(a.output,'w').write(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
