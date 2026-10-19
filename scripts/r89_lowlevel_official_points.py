#!/usr/bin/env python3
"""Apply the R89 parameter-free composition prior to official keyframes.

The script reads only official video pixels and the audited B3 observation
structure.  It writes a new point directory; packaging remains a separate
step and keeps the R54 frame mask/one-second hold+EMA contract fixed.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np

from r89_lowlevel_composition_probe import load_gray, frame_features_batch, candidates


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,required=True); ap.add_argument("--frames",type=Path,required=True); ap.add_argument("--coef",type=Path,required=True); ap.add_argument("--index",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    z=np.load(args.coef); beta=z["all_beta"].astype(np.float32); mu=z["all_mu"].astype(np.float32); sd=z["all_sd"].astype(np.float32); keep=z["all_keep"].astype(np.int64)
    src_points=args.source/"points"; src_kfs=args.frames; args.out.mkdir(parents=True,exist_ok=False); (args.out/"points").mkdir()
    rows=[json.loads(x) for x in args.index.read_text().splitlines() if x.strip()]; changed=0; total=0; vids=0
    for r in rows:
        vid=str(r["video_id"]); p=json.loads((src_points/f"{vid}.json").read_text()); W,H=int(p["W"]),int(p["H"]); ratio=list(map(float,p["ratios"]["t"]["ratio"])); kfs=[int(x) for x in p["keyframes"]]
        imgs=[]
        for k in kfs:
            imgs.append(load_gray(src_kfs/vid/f"{k}.png"))
        if imgs:
            cur=np.stack(imgs); prev=np.concatenate([cur[:1],cur[:-1]],axis=0); feat=frame_features_batch(cur,prev,W,H,ratio); scores=np.concatenate([((feat[:,:,keep]-mu[keep])/sd[keep]),np.ones((len(feat),feat.shape[1],1),np.float32)],axis=-1) @ beta
        else: scores=np.zeros((0,129),np.float32)
        q=dict(p); rt=dict(p["ratios"]["t"]); points=[]
        for i,old in enumerate(rt["points"]):
            if old is None or i>=len(scores): points.append(old); continue
            off,w,h,axis=candidates(W,H,ratio); j=int(np.argmax(scores[i]));
            if axis==0: cx=float((off[j]+w/2)/W); cy=.5
            elif axis==1: cx=.5; cy=float((off[j]+h/2)/H)
            else: cx=float(old[0]); cy=float(old[1])
            new=[round(float(np.clip(cx,0,1)),6),round(float(np.clip(cy,0,1)),6),bool(old[2])]
            if new[:2] != list(old[:2]): changed+=1
            points.append(new); total+=1
        rt["points"]=points; rt["status"]=["ok_r89_lowlevel" if x is not None else "none" for x in points]; rt["raw"]=[]; rt["boxes"]=[]; q["ratios"]={"t":rt}; q["model"]="R89 parameter-free low-level gradient/contrast/motion/sharpness composition prior"; q["runtime"]="R89_LOWLEVEL_MOTION_SALIENCY_COMPOSITION"; (args.out/"points"/f"{vid}.json").write_text(json.dumps(q,separators=(",",":"))+"\n"); vids+=1
    man={"protocol":"R89_LOWLEVEL_OFFICIAL_POINTS","source":str(args.source),"coefficient":str(args.coef),"video_count":vids,"keyframe_points":total,"changed_points":changed,"official_labels_used":False,"frame_mask_unchanged":True}
    (args.out/"manifest.json").write_text(json.dumps(man,indent=2)+"\n"); print(json.dumps(man,indent=2))

if __name__=="__main__": main()
