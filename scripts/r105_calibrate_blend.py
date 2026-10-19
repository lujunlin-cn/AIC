#!/usr/bin/env python3
"""Calibrate a simple AViMoS saliency blend against the R70 utility readout."""
import json
from pathlib import Path
import numpy as np

def main():
    import torch, importlib.util
    a = Path("/data/aic/experiments/R105_AVIMOS_AIC_UTILITY/features.npz")
    z=np.load(a,allow_pickle=True); x=z["x"].astype(np.float32); y=z["y"].astype(np.float32); st=z["stats"].astype(np.float32); src=z["source"].astype(str); names=z["name"].astype(str); dev=np.char.endswith(names,"dev")
    sp=importlib.util.spec_from_file_location("r70","/root/autodl-tmp/r70_spatial_utility_pilot.py"); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    ck=torch.load("/root/autodl-tmp/experiments/R70_SPATIAL_UTILITY/full/r70_head.pt",map_location="cpu",weights_only=False); net=m.Head.make(torch); net.load_state_dict(ck["state_dict"]); net.eval(); xt=torch.from_numpy((x-ck["mu"])/ck["sd"])
    with torch.inference_mode():
        p=np.concatenate([net(xt[j:j+128]).numpy() for j in range(0,len(x),128)])
    def rank(a): return np.argsort(np.argsort(a,axis=1,kind="mergesort"),axis=1,kind="mergesort")/(a.shape[1]-1)
    def score(pred,mask):
        vals=[]
        for s in sorted(set(src[mask])):
            q=mask&(src==s); vals.append(float(y[q,np.argmax(pred[q],1)].mean()))
        return float(np.mean(vals))
    ss=np.array(sorted(set(src[dev]))); tune=set(ss[::2]); tune_mask=dev&np.isin(src,list(tune)); hold=dev&~np.isin(src,list(tune)); rp=rank(p); out=[]
    for col in range(4):
        rs=rank(st[:,:,col])
        for alpha in np.linspace(-1,1,41): out.append((score(rp+alpha*rs,tune_mask),score(rp+alpha*rs,hold),col,float(alpha)))
    out.sort(reverse=True); base=(score(p,tune_mask),score(p,hold)); report={"baseline_tune":base[0],"baseline_hold":base[1],"best":out[:20],"rows":len(x),"dev_sources":len(ss)}
    Path("/data/aic/experiments/R105_AVIMOS_AIC_UTILITY/blend_metrics.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
