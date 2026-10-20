#!/usr/bin/env python3
"""Real-image equivalence and coordinate regression checks for matched study."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dinov3_runtime import preprocess_rgb
from dinov3_spatial_matched import prepare_image, pooling, summarize
from dinov3_spatial_patch_probe import candidate_features, grid_masks


def main():
    torch.set_num_threads(4)
    path=Path(sys.argv[1])
    with Image.open(path) as image: rgb=np.asarray(image.convert("RGB"))
    legacy,tr=preprocess_rgb(rgb)
    fast,tf=prepare_image(path)
    assert tr==tf
    np.testing.assert_allclose(legacy.numpy(),fast,rtol=0,atol=5e-7)
    rng=np.random.default_rng(12); grid=rng.normal(size=(14,14,768)).astype(np.float32); cls=rng.normal(size=768)
    errors=[]
    for ratio in ([9,16],[16,9],[3,1]):
        W,H=rgb.shape[1],rgb.shape[0]
        mask,valid,offsets=grid_masks(grid,tr,W,H,ratio)
        a=candidate_features(grid,cls,tr,W,H,ratio,"patch",rng).astype(np.float32)
        g=torch.from_numpy(grid.reshape(1,196,768)).cuda()
        pos=offsets/(offsets[-1] if offsets[-1]>0 else 1.)
        b=pooling(g,torch.from_numpy(mask[None]).cuda(),torch.from_numpy(valid[None]).cuda(),torch.from_numpy(pos[None]).cuda()).cpu().numpy()[0]
        # Final float16 storage causes up to one half-precision ULP of change.
        np.testing.assert_allclose(a,b,atol=.002,rtol=.002)
        errors.append(float(np.max(abs(a-b))))
        assert np.all(mask[:,valid==0]==0), "padding must be excluded"
    # Expose source-vs-frame weighting bug: two sources have unequal row counts.
    y=np.zeros((4,129),np.float32); y[:3,64]=1.; y[3,64]=0.
    p=np.zeros_like(y);p[:,64]=1.;src=np.array(["rv:a"]*3+["rv:b"]);names=np.array(["rv_dev"]*4)
    # summarize includes both domains; add a neutral LIVE source.
    y=np.concatenate([y,np.zeros((1,129),np.float32)]); p=np.concatenate([p,np.zeros((1,129),np.float32)])
    src=np.append(src,"live:c"); names=np.append(names,"live_dev")
    report,_=summarize(p,y,src,names,np.ones(5,bool),4)
    assert report["rv"]["center"]==.5,report
    assert report["rv"]["delta"]==0.,report
    t=time.perf_counter()
    for _ in range(32):prepare_image(path)
    print(json.dumps({"status":"passed","preprocess_max_error":float(np.max(abs(legacy.numpy()-fast))),"pooling_max_errors":errors,"source_macro_weighting":"passed","32_preprocess_s":time.perf_counter()-t}))


if __name__=="__main__": main()
