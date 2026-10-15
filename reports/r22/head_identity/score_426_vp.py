# Score the 426 official eval set with the VideoPrism-B highlight head.
# Sampling points are taken from the ckpt_030 score jsons so fusion is
# point-aligned and the packager accepts the output unchanged.
# Usage: score_426_vp.py <intake.jsonl> <s030_dir> <head_pt> <npz> <out_dir>
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.environ.get("VP_MODEL_ROOT", "/root/autodl-tmp/vpb"))
from videoprism_torch import build_base  # noqa: E402
import torch.nn as nn  # noqa: E402


class HeadMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 1))

    def forward(self, x):  # [B,T,d] -> [B,T]
        return self.net(x).squeeze(-1)


class HeadAttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 1))

    def forward(self, x):
        h = self.ln(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.mlp(x + a).squeeze(-1)


class HeadBinned(nn.Module):
    """K-bin classifier; forward returns [B,T,K] logits. At inference the
    score is the softmax expectation over bin centers (scale distribution)."""

    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, k))

    def forward(self, x):
        return self.net(x)


Head = HeadMLP  # compat alias


def load_head(p):
    """-> (module_on_cpu_eval, arch, bin_centers_or_None)."""
    ck = torch.load(p, map_location="cpu", weights_only=False)
    d = ck["norm"][0].shape[0]
    arch = ck.get("arch", "mlp")
    if arch == "attpool":
        hd = HeadAttnPool(d)
        bc = None
    elif arch == "binned":
        hd = HeadBinned(d, len(ck["bin_centers"]))
        bc = torch.from_numpy(np.asarray(ck["bin_centers"], dtype=np.float32))
    elif arch == "bce":
        hd = HeadMLP(d)
        bc = None
    else:
        hd = HeadMLP(d)
        bc = None
    hd.load_state_dict(ck["state"])
    return hd.eval(), arch, bc

ap = argparse.ArgumentParser()
ap.add_argument("intake")
ap.add_argument("s030_dir")
ap.add_argument("head_pt")
ap.add_argument("npz", nargs="?", default=None)
ap.add_argument("out_dir")
ap.add_argument("--device", default="npu:0")
ap.add_argument("--alpha", type=float, default=0.0)
ap.add_argument("--shard-i", type=int, default=0)
ap.add_argument("--shard-n", type=int, default=1)
ap.add_argument("--ckpt-path", default=None,
                help="fp16 .pt weights; use instead of the flax npz")
ap.add_argument("--model-root", default="/root/autodl-tmp/vpb")
ap.add_argument("--heads-glob", default=None,
                help="e.g. '/x/heads/vp_head_temporal_gif_hard_mlp_s*.pt': "
                     "average logits over all matching heads (ensemble)")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rows = [json.loads(l) for l in open(args.intake)][args.shard_i :: args.shard_n]
if args.heads_glob:
    import glob as _g
    head_paths = sorted(_g.glob(args.heads_glob))
    assert head_paths, f"no heads match {args.heads_glob}"
    heads, archs, bcs = [], [], []
    for p in head_paths:
        hd, arch, bc = load_head(p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        mu, sd = ck["norm"]
        heads.append(hd); archs.append(arch); bcs.append(bc)
    print(f"ensemble of {len(heads)} heads", flush=True)
else:
    hd, arch, bc = load_head(args.head_pt)
    ck = torch.load(args.head_pt, map_location="cpu", weights_only=False)
    mu, sd = ck["norm"]
    heads, archs, bcs = [hd], [arch], [bc]

m = build_base(ckpt_path=args.ckpt_path) if args.ckpt_path \
    else build_base(npz_path=args.npz)
dev = torch.device(args.device)
if dev.type == "npu":
    import torch_npu
    assert torch_npu.npu.device_count() == 1, "npu silently unavailable"
m = m.half().to(dev).eval()
heads = [h.to(dev) for h in heads]
mu_t = torch.from_numpy(mu).float().to(dev)
sd_t = torch.from_numpy(sd).float().to(dev)

vp = {}
done = 0
cv2.ocl.setUseOpenCL(False)
with torch.no_grad():
    for r in rows:
        vid = str(r["video_id"])
        sp = os.path.join(args.s030_dir, f"{vid}.json")
        if not os.path.exists(sp):
            continue
        s030 = json.load(open(sp))
        idxs = s030["frame_indices"]
        n_pts = len(idxs)
        # one sequential decode pass; cv2 POS_FRAMES seek is unusably slow
        # on these h264 files (~2.5 min/video, 10-13)
        want = sorted(set(int(i) for i in idxs))
        cap = cv2.VideoCapture(r["video_path"])
        grabbed = {}
        fi = 0
        while fi <= want[-1]:
            ok, im = cap.read()
            if not ok:
                break
            if fi in want:
                im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                if im2.shape[:2] != (288, 288):
                    im2 = cv2.resize(im2, (288, 288),
                                     interpolation=cv2.INTER_AREA)
                grabbed[fi] = im2
            fi += 1
        cap.release()
        last = grabbed[want[0]] if want else np.zeros((288, 288, 3), np.uint8)
        imgs = []
        for fi in idxs:
            im = grabbed.get(int(fi))
            if im is None:
                im = last
            last = im
            imgs.append(im)
        while len(imgs) < n_pts:
            imgs.append(imgs[-1])
        x = (np.stack(imgs[:n_pts]).astype(np.float32) / 255.0)
        xt = torch.from_numpy(x).half().unsqueeze(0).to(dev)
        feat = m(xt).float().mean(2)  # [1,n_pts,768]
        fn = ((feat[0] - mu_t) / sd_t).unsqueeze(0)
        outs = []
        for h, arch, bc in zip(heads, archs, bcs):
            o = h(fn)
            if arch == "bce":
                o = torch.sigmoid(o)          # [B,T] P(highlight)
            elif arch == "binned":
                o = torch.softmax(o, -1) @ bc  # [B,T] expectation
            outs.append(o[0].float())
        scores = torch.stack(outs).mean(0).cpu().numpy()
        scores = np.clip(scores, 0.0, 1.0).tolist()
        if args.alpha > 0:
            base = np.asarray(s030["scores"], dtype=np.float64)
            scores = ((1 - args.alpha) * base + args.alpha * np.asarray(scores)).tolist()
        vp[vid] = scores
        json.dump({"video_id": vid, "frame_indices": idxs,
                   "n_frames": s030.get("n_frames", n_pts),
                   "n_points": n_pts, "scores": scores,
                   "model": "videoprism_b_head_v1", "alpha": args.alpha},
                  open(os.path.join(args.out_dir, f"{vid}.json"), "w"))
        done += 1
        if done % 40 == 0:
            print(f"{done} scored", flush=True)
print(f"SCORE426_DONE n={done}")
