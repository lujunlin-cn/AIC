#!/usr/bin/env python3
"""Fill grpo_pool_v4's target/vpb with VideoPrism-B head frame-level scores
(run9 teacher signal). Reads the jpgs poolv4_build2.py wrote (long side 1024,
same rescale as the 426 scorer), resizes to 288, forwards the frozen
VideoPrism base once per video, then scores with two head ensembles:

  gif_ens   vp_head_temporal_gif_hard_mlp_s13-22 (10 heads)  -> target + vpb
  fused     vp_head_temporal_fused_target_mlp_s13-17 (5 heads) -> target_fused

Same head arch / feature normalization as score_426_vp.py (mu,sd in ckpt).

Usage: vpb_fullaxis_score.py <pool.jsonl> <head_dir> <out.jsonl>
       [--ckpt-path vp.pt] [--npz vp.npz] [--device cuda:0]
Resumable: rows whose target is already set are skipped; output is written
after every video.
"""
import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get("VP_MODEL_ROOT", "/root/autodl-tmp/vpb"))
from videoprism_torch import build_base  # noqa: E402


class HeadMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 1))

    def forward(self, x):  # [B,T,d] -> [B,T]
        return self.net(x).squeeze(-1)


def load_head(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    d = ck["norm"][0].shape[0]
    hd = HeadMLP(d)
    hd.load_state_dict(ck["state"])
    return hd.eval(), ck["norm"]


ap = argparse.ArgumentParser()
ap.add_argument("pool")
ap.add_argument("head_dir")
ap.add_argument("out")
ap.add_argument("--ckpt-path", default="/root/autodl-tmp/vpb/videoprism_base_f16r288.fp16.pt")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

GIF_GLOB = os.path.join(args.head_dir, "vp_head_temporal_gif_hard_mlp_s*.pt")
FUS_GLOB = os.path.join(args.head_dir, "vp_head_temporal_fused_target_mlp_s*.pt")

ensembles = {}
for tag, pat in [("gif_ens", GIF_GLOB), ("fused", FUS_GLOB)]:
    heads, norms = [], []
    for p in sorted(glob(pat)):
        h, nm = load_head(p)
        heads.append(h)
        norms.append(nm)
    assert heads, f"no heads match {pat}"
    ensembles[tag] = (heads, norms)
    print(f"[ens] {tag}: {len(heads)} heads", flush=True)

dev = torch.device(args.device)
ensembles = {t: ([h.to(dev) for h in hs], ns)
             for t, (hs, ns) in ensembles.items()}
m = build_base(ckpt_path=args.ckpt_path).half().to(dev).eval()
muses = {t: [torch.from_numpy(n[0]).float().to(dev) for n in ns]
         for t, (_h, ns) in ensembles.items()}
sdss = {t: [torch.from_numpy(n[1]).float().to(dev) for n in ns]
        for t, (_h, ns) in ensembles.items()}

rows = [json.loads(l) for l in open(args.pool)]
print(f"[fullaxis] {len(rows)} rows", flush=True)

cv2.ocl.setUseOpenCL(False)
done = 0
with torch.no_grad():
    for r in rows:
        if r.get("target"):
            done += 1
            continue
        fdir = r["frames_dir"]
        imgs = []
        for f in r["files"]:
            im = cv2.imread(os.path.join(fdir, f))
            assert im is not None, f"{fdir}/{f}"
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im.shape[:2] != (288, 288):
                im = cv2.resize(im, (288, 288), interpolation=cv2.INTER_AREA)
            imgs.append(im)
        x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0)
        xt = x.half().unsqueeze(0).to(dev)
        feat = m(xt).float().mean(2)  # [1,n,768]
        outs = {}
        for tag, (heads, _ns) in ensembles.items():
            ss = []
            for h, mu, sd in zip(heads, muses[tag], sdss[tag]):
                fn = ((feat[0] - mu) / sd).unsqueeze(0)
                ss.append(h(fn)[0].float())
            outs[tag] = torch.stack(ss).mean(0).cpu().numpy()
        n = len(imgs)
        r["target"] = np.clip(outs["gif_ens"], 0.0, 1.0).round(4).tolist()
        r["target_fused"] = np.clip(outs["fused"], 0.0, 1.0).round(4).tolist()
        r["vpb"] = r["target"]
        done += 1
        with open(args.out, "w") as f:
            for rr in rows:
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        if done % 20 == 0:
            print(f"[fullaxis] {done}/{len(rows)}", flush=True)

with open(args.out, "w") as f:
    for rr in rows:
        f.write(json.dumps(rr, ensure_ascii=False) + "\n")
sp = [float(np.corrcoef(np.argsort(np.asarray(r["target"])).argsort(),
                        np.argsort(np.asarray(r["vpb"])).argsort())[0, 1])
      for r in rows if r.get("target")]
print(f"[fullaxis] DONE {done}/{len(rows)} target==vpb", flush=True)
