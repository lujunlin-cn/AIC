#!/usr/bin/env python3
"""Audit Feature Bank dimensions before any grouped ablation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from aic.features import load_feature_cache

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    arrays=[]; missing=[]; videos=[]
    for r in rows:
        d=load_feature_cache(r["path"]); x=d.get("aux",np.zeros((len(d["features"]),0),np.float32)); arrays.append(x); videos.append(r.get("video_id"))
    x=np.concatenate(arrays,axis=0); stats=[]
    for j in range(x.shape[1]):
        v=x[:,j]; finite=np.isfinite(v); nz=np.abs(v)>1e-8
        stats.append({"dim":j,"variance":float(np.var(v)),"mean":float(np.mean(v)),"min":float(np.min(v)),"max":float(np.max(v)),"missing_ratio":float(1-finite.mean()),"zero_ratio":float(1-nz.mean())})
    corr=np.corrcoef(x,rowvar=False) if x.shape[1] else np.zeros((0,0))
    pairs=[]
    for i in range(x.shape[1]):
      for j in range(i+1,x.shape[1]):
        if abs(corr[i,j])>=.999:
          pairs.append({"i":i,"j":j,"correlation":float(corr[i,j])})
    out={"manifest":str(a.manifest),"videos":len(rows),"rows":int(len(x)),"dimensions":stats,"near_duplicate_pairs":pairs,"group_hint":{"motion":[0,1,2,3,4,5,6,7,8,9],"quality":[12,13,14,15,16,17,18,19,20,21,22],"audio":[23,24,25,26,27,28,31],"composition":[29,30]}}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps({"rows":len(x),"dims":x.shape[1],"duplicates":pairs},indent=2))
if __name__=='__main__': main()
