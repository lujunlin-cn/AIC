#!/usr/bin/env python3
"""R70 spatial utility readout pilot.

This is a public native-crop diagnostic and deployment preparation tool.  It
keeps the V9 vision tower and the 129 legal max-window contract fixed, then
trains a new candidate-utility readout on source-disjoint RetargetVid/LIVE
crop GT.  It never reads official test labels.

The script is intentionally self-contained because the current AutoDL clone
does not contain the original V8 B3 checkpoint.  A resulting head is therefore
called R70 and must not be described as a B3 reproduction.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np


NC = 129


def geometry(W: float, H: float, ratio: list[float]):
    rw, rh = ratio
    target = rw / rh
    if W / H > target:
        h, w = H, H * target
        return w, h, 0
    if W / H < target:
        w, h = W, W / target
        return w, h, 1
    return W, H, None


def candidate_offsets(W: float, H: float, ratio: list[float]):
    w, h, axis = geometry(W, H, ratio)
    if axis == 0:
        return np.linspace(0.0, W - w, NC), w, h, axis
    if axis == 1:
        return np.linspace(0.0, H - h, NC), w, h, axis
    return np.zeros(NC), w, h, axis


def boxes(offsets, W, H, ratio):
    w, h, axis = geometry(W, H, ratio)
    out = np.zeros((len(offsets), 4), np.float32)
    if axis == 0:
        out[:, :] = np.stack([offsets, np.zeros(len(offsets)), offsets + w, np.full(len(offsets), h)], 1)
    elif axis == 1:
        out[:, :] = np.stack([np.zeros(len(offsets)), offsets, np.full(len(offsets), w), offsets + h], 1)
    else:
        out[:, :] = [0, 0, W, H]
    return out


def iou(a, b):
    # a: [N,4], b: [A,4]
    a = np.asarray(a, np.float32)[:, None, :]
    b = np.asarray(b, np.float32)[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-8)


def feature_from_grid(grid, W, H, ratio):
    """Return [129,2305] V9 candidate features from one vision grid."""
    offsets, w, h, axis = candidate_offsets(W, H, ratio)
    if axis is None:
        axis = 0
    fh, fw, D = grid.shape
    flat = grid.reshape(-1, D).astype(np.float32)
    px_per = np.array([W / fw, H / fh], np.float32)
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((NC, fh * fw), np.float32)
    for j, o in enumerate(offsets):
        if axis == 0:
            x1, y1, x2, y2 = o, 0.0, o + w, h
        else:
            x1, y1, x2, y2 = 0.0, o, w, o + h
        cx1 = int(math.floor(x1 / px_per[0]))
        cx2 = max(int(math.ceil(x2 / px_per[0])), cx1 + 1)
        cy1 = int(math.floor(y1 / px_per[1]))
        cy2 = max(int(math.ceil(y2 / px_per[1])), cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1)
    win = m @ flat / np.maximum(m.sum(1, keepdims=True), 1)
    outm = (1.0 - m) @ flat / np.maximum((1.0 - m).sum(1, keepdims=True), 1)
    pos = (offsets / (offsets[-1] if offsets[-1] > 0 else 1.0))[:, None]
    return np.concatenate([win, outm, win - outm, pos], 1).astype(np.float16), offsets


def load_vision(args):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, Lfm2VlForConditionalGeneration, Lfm2VlImageProcessor

    cfg = AutoConfig.from_pretrained(args.processor_dir)
    for c in (cfg, getattr(cfg, "vision_config", None), getattr(cfg, "text_config", None)):
        if c is not None:
            c._attn_implementation = "eager"
    shell = Lfm2VlForConditionalGeneration(cfg).half()
    vision = shell.model.vision_tower
    sd = load_file(str(Path(args.vision_dir) / "vision_tower.safetensors"))
    # Transformers 5.20 exposes SigLIP keys without vision_model.; 5.1 used
    # the extra prefix.  Accept both while asserting exact tensor coverage.
    fixed = {}
    expected = set(vision.state_dict().keys())
    # The same LFM checkpoint is exposed with two key contracts across the
    # transformers versions used by this project.  Older runtimes expect the
    # ``vision_model.`` namespace; newer ones flatten it.  Normalize against
    # the instantiated tower instead of assuming one library version.
    for k, v in sd.items():
        k = k[len("model.vision_tower.") :]
        k = k.removeprefix("vision_model.")
        if k not in expected and ("vision_model." + k) in expected:
            k = "vision_model." + k
        fixed[k] = v.half()
    missing, unexpected = vision.load_state_dict(fixed, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"vision checkpoint mismatch missing={missing[:3]} unexpected={unexpected[:3]}")
    vision = vision.to(args.device).eval()
    pc = json.loads((Path(args.processor_dir) / "processor_config.json").read_text())["image_processor"]
    pc.pop("image_processor_type", None)
    pc["min_tiles"] = 1
    pc["max_tiles"] = 1
    pc["max_image_tokens"] = 256
    proc = Lfm2VlImageProcessor(**pc)
    return torch, vision, proc


def image_path(args, ds, name, kf):
    if ds == "live":
        root = Path(args.t5_dir) / ("keyframes_val" if name == "live_confirm3" else "keyframes_train")
        # LIVE directories are keyed by video id; the caller supplies it via
        # `name` only for the split, so this helper is overridden in extract.
        raise RuntimeError("LIVE image path requires the video id")
    return Path(args.rv_frames) / "d1" / name / f"{kf}.png"


def table_units(args, name):
    return pickle.loads((Path(args.t5_dir) / "tables" / f"{name}.pkl").read_bytes())


def table_gt(unit, j, offsets):
    gf = {int(f): i for i, f in enumerate(unit["gf"])}
    kf = int(unit["keys"][int(unit["J"][j])])
    if kf not in gf:
        return None
    gt = unit["gt"][:, gf[kf], :]
    W, H, ratio = unit["W"], unit["H"], unit["ratio"]
    return iou(boxes(offsets, W, H, ratio), gt).mean(1).astype(np.float32)


def extract(args, names):
    from PIL import Image
    torch, vision, proc = load_vision(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    shard_idx = 0

    def flush_rows(rows, shard_idx):
        if not rows:
            return shard_idx
        np.savez_compressed(
            shard_dir / f"shard_{shard_idx:05d}.npz",
            x=np.stack([r["x"] for r in rows]),
            y=np.stack([r["y"] for r in rows]),
            source=np.array([r["source"] for r in rows], dtype=object),
            name=np.array([r["name"] for r in rows], dtype=object),
            ratio=np.array(["-".join(str(int(z)) for z in r["ratio"]) for r in rows], dtype=object),
            W=np.array([r["W"] for r in rows]), H=np.array([r["H"] for r in rows]),
            offsets=np.stack([r["offsets"] for r in rows]),
        )
        rows.clear()
        return shard_idx + 1

    t0 = time.time()
    rows = []

    def encode_batch(paths):
        """Encode equal-contract images in batches, returning token grids."""
        encoded = {}
        batch_size = int(getattr(args, "vision_batch", 16))
        for st in range(0, len(paths), batch_size):
            chunk = paths[st:st + batch_size]
            images = [Image.open(p).convert("RGB") for p in chunk]
            try:
                x = proc(images=images, return_tensors="pt")
                x = {k: v.to(args.device) for k, v in x.items() if torch.is_tensor(v)}
                with torch.no_grad():
                    o = vision(pixel_values=x["pixel_values"].half(),
                               spatial_shapes=x["spatial_shapes"],
                               pixel_attention_mask=x["pixel_attention_mask"],
                               return_dict=True)
                for bi, p in enumerate(chunk):
                    valid = int(x["pixel_attention_mask"][bi].sum())
                    fh, fw = [int(v) for v in x["spatial_shapes"][bi]]
                    grid = o.last_hidden_state[bi, :valid].float().detach().cpu().numpy()
                    grid = grid.reshape(fh, fw, -1)
                    # V9 contract discards odd token rows/columns before pooling.
                    grid = grid[: (fh // 2) * 2, : (fw // 2) * 2]
                    encoded[p] = grid
            finally:
                for image in images:
                    image.close()
        return encoded

    for name in names:
        units = table_units(args, name)
        if args.limit:
            units = units[: args.limit]
        for ui, unit in enumerate(units):
            # A unit contains all frames for one source/ratio.  The same
            # source frame is reused for both ratios in adjacent units, but
            # each unit is kept bounded so feature extraction does not retain
            # the whole dataset in RAM.
            cache = {}
            ds = unit["ds"]
            needed = []
            paths = {}
            for j in unit["J"]:
                kf = int(unit["keys"][int(j)])
                key = (ds, unit["vid"], kf)
                if key in cache:
                    continue
                if ds == "live":
                    root = Path(args.t5_dir) / ("keyframes_val" if name == "live_confirm3" else "keyframes_train")
                    p = root / str(unit["vid"]) / f"{kf}.png"
                else:
                    p = image_path(args, ds, unit["vid"], kf)
                if not p.exists():
                    raise FileNotFoundError(p)
                paths[key] = p
                needed.append(key)
            encoded = encode_batch([paths[k] for k in needed])
            for key in needed:
                cache[key] = encoded[paths[key]]
            for jj, j in enumerate(unit["J"]):
                kf = int(unit["keys"][int(j)])
                key = (ds, unit["vid"], kf)
                grid = cache[key]
                feat, offsets = feature_from_grid(grid, unit["W"], unit["H"], unit["ratio"])
                y = table_gt(unit, jj, offsets)
                if y is None:
                    continue
                rows.append({"x": feat, "y": y, "source": f"{ds}:{unit['vid']}",
                             "name": name, "ratio": unit["ratio"], "W": unit["W"], "H": unit["H"],
                             "offsets": offsets.astype(np.float32)})
                if len(rows) >= 256:
                    shard_idx = flush_rows(rows, shard_idx)
            if (ui + 1) % 10 == 0:
                print(json.dumps({"split": name, "units": ui + 1, "rows": len(rows),
                                  "cache_frames": len(cache), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    shard_idx = flush_rows(rows, shard_idx)
    (out / "meta.json").write_text(json.dumps({"shards": shard_idx, "names": names, "limit": args.limit,
                                                 "vision_dir": str(args.vision_dir), "nc": NC,
                                                 "contract": "V9 vision grid + 129 legal max windows; new readout, not B3"}, indent=2) + "\n")


class Head:
    @staticmethod
    def make(torch, d=2305, ch=128, heads=4, ff=256):
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.proj=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.Linear(512,ch)); self.h=heads
                self.qkv=nn.ModuleList([nn.Linear(ch,3*ch) for _ in range(2)]); self.proj_o=nn.ModuleList([nn.Linear(ch,ch) for _ in range(2)])
                self.ln1=nn.ModuleList([nn.LayerNorm(ch) for _ in range(2)]); self.ln2=nn.ModuleList([nn.LayerNorm(ch) for _ in range(2)])
                self.ff1=nn.ModuleList([nn.Linear(ch,ff) for _ in range(2)]); self.ff2=nn.ModuleList([nn.Linear(ff,ch) for _ in range(2)]); self.out=nn.Linear(ch,1)
            def forward(self,x):
                h=self.proj(x)
                for i in range(2):
                    B,N,C=h.shape; q,k,v=self.qkv[i](h).reshape(B,N,3,self.h,C//self.h).permute(2,0,3,1,4).unbind(0)
                    a=torch.softmax(q@k.transpose(-1,-2)/(C//self.h)**.5,dim=-1); h=self.ln1[i](h+self.proj_o[i]((a@v).transpose(1,2).reshape(B,N,C)))
                    h=self.ln2[i](h+self.ff2[i](torch.nn.functional.gelu(self.ff1[i](h))))
                return self.out(h).squeeze(-1)
        return M()


def train(args):
    import torch
    rows_path = Path(args.out) / "rows.npz"
    if rows_path.exists():
        z=np.load(rows_path,allow_pickle=True); x=z["x"].astype(np.float32); y=z["y"].astype(np.float32); src=z["source"].astype(str); names=z["name"].astype(str)
    else:
        parts=[]
        for p in sorted((Path(args.out)/"shards").glob("shard_*.npz")):
            q=np.load(p,allow_pickle=True); parts.append(q)
        if not parts: raise RuntimeError("no R70 feature shards found")
        x=np.concatenate([q["x"] for q in parts]).astype(np.float32)
        y=np.concatenate([q["y"] for q in parts]).astype(np.float32)
        src=np.concatenate([q["source"] for q in parts]).astype(str)
        names=np.concatenate([q["name"] for q in parts]).astype(str)
    train_mask=np.char.endswith(names,"train")
    dev_mask=np.char.endswith(names,"dev")
    if not train_mask.any() or not dev_mask.any(): raise RuntimeError("extract both train and dev tables")
    mu=x[train_mask].reshape(-1,x.shape[-1]).mean(0); sd=x[train_mask].reshape(-1,x.shape[-1]).std(0); sd[sd<1e-5]=1
    model=Head.make(torch).to(args.device).float(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    rng=np.random.default_rng(args.seed); ids=np.flatnonzero(train_mask); best=(-1,None)
    xt=torch.from_numpy((x-mu)/sd); yt=torch.from_numpy(y)
    for step in range(args.steps):
        take=rng.choice(ids,size=min(args.batch,len(ids)),replace=False); pred=model(xt[take].to(args.device)); loss=((pred-yt[take].to(args.device))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if (step+1)%100==0:
            with torch.no_grad(): p=model(xt[dev_mask].to(args.device)).argmax(1).cpu().numpy(); val=float(y[dev_mask][np.arange(dev_mask.sum()),p].mean())
            if val>best[0]: best=(val,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            print(json.dumps({"step":step+1,"loss":float(loss),"dev_iou":val}),flush=True)
    if best[1] is not None:model.load_state_dict(best[1])
    with torch.no_grad():
        pred=model(xt.to(args.device)).argmax(1).cpu().numpy()
    rows=np.arange(len(y)); pred_iou=y[rows,pred]; center_iou=y[:,NC//2]
    report={"rows":int(len(y)),"train_rows":int(train_mask.sum()),"dev_rows":int(dev_mask.sum()),"best_dev_iou":best[0],"all_new_iou":float(pred_iou.mean()),"all_center_iou":float(center_iou.mean()),"dev_new_iou":float(pred_iou[dev_mask].mean()),"dev_center_iou":float(center_iou[dev_mask].mean()),"source_count":int(len(set(src)))}
    # The protocol's independent unit is a source video, not a frame or a
    # keyframe.  Report both held-out pools separately so a LIVE/RV mixture
    # cannot hide a domain failure.
    for split in ("rv_dev", "live_dev"):
        mask = names == split
        per_new, per_center = [], []
        for s in sorted(set(src[mask])):
            q = mask & (src == s)
            per_new.append(float(pred_iou[q].mean()))
            per_center.append(float(center_iou[q].mean()))
        if per_new:
            report[split] = {"sources": len(per_new), "new_source_macro": float(np.mean(per_new)),
                             "center_source_macro": float(np.mean(per_center)),
                             "delta": float(np.mean(np.asarray(per_new) - np.asarray(per_center)))}
    (Path(args.out)/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    torch.save({"state_dict":model.state_dict(),"mu":mu,"sd":sd,"config":{"d":2305,"nc":NC},"report":report},Path(args.out)/"r70_head.pt")
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["extract","train"],required=True); ap.add_argument("--t5-dir",required=True); ap.add_argument("--rv-frames",required=True); ap.add_argument("--vision-dir",required=True); ap.add_argument("--processor-dir",required=True); ap.add_argument("--out",required=True); ap.add_argument("--device",default="cuda"); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--steps",type=int,default=1000); ap.add_argument("--batch",type=int,default=16); ap.add_argument("--vision-batch",type=int,default=16); ap.add_argument("--seed",type=int,default=20261018); ap.add_argument("--names",nargs="*",default=["rv_train","live_train","rv_dev","live_dev"]); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    extract(a,a.names) if a.mode=="extract" else train(a)
if __name__=="__main__": main()
