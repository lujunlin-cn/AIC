#!/usr/bin/env python3
"""P0-b: Mr.HiSum frame-ranking pilot — loss as the single variable.

Data (all on-disk, zero download):
  features: /data/aic/external_datasets/MrHiSum/processed/yt8m/trainNNNN.h5
            per video: rgb (T,1024) uint8, audio (T,128) uint8   [V6 dequant: (q+0.5)*2/255-1]
  labels:   /data/aic/external_datasets/MrHiSum/annotations/mr_hisum.h5  gtscore (T,) float64
  split:    annotations/upstream_repo/dataset/mr_hisum_split.json  (27,892/2,000/2,000)

Model: TRINITY-style light global ranker — input proj 1152->512, Transformer
encoder (depth 3, 8 heads, 1D RoPE), per-frame score head (512->1).

Losses (single variable):
  --loss pairwise  multiple pairwise ranking (Mr.HiSum official recipe):
                   sample K pairs per video with |gs_i - gs_j| >= tau,
                   softplus(s_j - s_i) for gs_i > gs_j
  --loss mse       frame-level MSE to gtscore (our B3 status quo)

Metric: mAP@rho (rho=15/50): per video, predicted top rho%% frames vs
gtscore top rho%% frames, precision; macro-average x100. Baseline to beat:
Mr.HiSum official paper 50.80 / 67.20 (same yt8m features; TRINITY's numbers
use their own CLIP features and are NOT comparable).

Usage:
  python3 p0b_mrhisum_rank.py --loss pairwise --subset 0 --epochs 3 --tag pair_v1
  python3 p0b_mrhisum_rank.py --loss mse     --subset 0 --epochs 3 --tag mse_v1
Smoke first:  --subset 300 --epochs 1 --val-subset 100 --tag smoke
"""
import argparse
import glob
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn

FEAT_GLOB = "/data/aic/external_datasets/MrHiSum/processed/yt8m/train*.h5"
GT_H5 = "/data/aic/external_datasets/MrHiSum/annotations/mr_hisum.h5"
SPLIT = "/data/aic/external_datasets/MrHiSum/annotations/upstream_repo/dataset/mr_hisum_split.json"
OUT_DIR = "/data/aic/experiments_910a/LFM_V11/r12_p0b"

D_MODEL, N_HEADS, N_LAYERS = 512, 8, 3
ROPE_THETA = 10000.0
MAX_T = 512
PAIRS_PER_VIDEO = 64
PAIR_TAU = 0.10


# ---------------- data ----------------
def build_index():
    """vid -> (feature_path, shard_key, gt_idx). One-time scan of 1462 shards."""
    idx = {}
    with h5py.File(GT_H5, "r") as gt:
        gt_keys = set(gt.keys())
    for path in sorted(glob.glob(FEAT_GLOB)):
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k in gt_keys:
                    idx[k] = path
    return idx


def dequant(a):
    return (a.astype(np.float32) + 0.5) * 2.0 / 255.0 - 1.0


# ---------------- model ----------------
def rope_cache(n, d, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(n, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.nh, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        qkv = self.qkv(self.n1(x)).view(b, t, 3, N_HEADS, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, t, -1)
        x = x + self.proj(o)
        x = x + self.mlp(self.n2(x))
        return x


class Ranker(nn.Module):
    def __init__(self, d_in=1152):
        super().__init__()
        self.proj = nn.Linear(d_in, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.head = nn.Linear(D_MODEL, 1)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        h = self.proj(x)
        cos, sin = rope_cache(h.shape[1], D_MODEL // N_HEADS, h.device)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.norm(h)).squeeze(-1)  # (B, T)


# ---------------- losses ----------------
def pairwise_loss(scores, gs, k=PAIRS_PER_VIDEO, tau=PAIR_TAU):
    """scores, gs: (T,). Sample k pairs with |gs_i-gs_j|>=tau; softplus rank."""
    t = scores.shape[0]
    if t < 2:
        return scores.sum() * 0.0
    idx = torch.randint(0, t, (k * 4, 2), device=scores.device)
    i, j = idx[:, 0], idx[:, 1]
    diff = gs[i] - gs[j]
    mask = diff.abs() >= tau
    i, j, diff = i[mask], j[mask], diff[mask]
    if len(i) > k:
        sel = torch.randperm(len(i), device=scores.device)[:k]
        i, j, diff = i[sel], j[sel], diff[sel]
    if len(i) == 0:
        return scores.sum() * 0.0
    margin = scores[i] - scores[j]
    sign = torch.where(diff > 0, 1.0, -1.0)
    return torch.nn.functional.softplus(-sign * margin).mean()


# ---------------- main ----------------
def preload_cache(vids, idx):
    """vid -> (rgb_u8 (T,1024), aud_u8 (T,128), gs (T,) f32). ~9GB RAM for 31,892 vids."""
    cache = {}
    by_path = {}
    for v in vids:
        by_path.setdefault(idx[v], []).append(v)
    for path, vk_list in by_path.items():
        with h5py.File(path, "r") as f:
            for v in vk_list:
                cache[v] = (f[v]["rgb"][()], f[v]["audio"][()])
    with h5py.File(GT_H5, "r") as gt:
        for v in vids:
            rgb, aud = cache[v]
            gs = gt[v]["gtscore"][()].astype(np.float32)
            t = min(len(rgb), len(gs))
            cache[v] = (rgb[:t], aud[:t], gs[:t])
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["pairwise", "mse"], required=True)
    ap.add_argument("--subset", type=int, default=0, help="0=all")
    ap.add_argument("--val-subset", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(32)

    print("[p0b] building index...", flush=True)
    idx = build_index()
    sp = json.load(open(SPLIT))
    tr = [k for k in sp["train_keys"] if k in idx]
    va_all = [k for k in sp["val_keys"] if k in idx]
    te_all = [k for k in sp["test_keys"] if k in idx]
    if args.subset:
        tr = tr[: args.subset]
    va = va_all[: args.val_subset]
    print(f"[p0b] train={len(tr)} val={len(va)} loss={args.loss}", flush=True)

    print("[p0b] preloading features (uint8) ...", flush=True)
    cache = preload_cache(sorted(set(tr) | set(va_all) | set(te_all)), idx)
    print(f"[p0b] cache ready: {len(cache)} vids", flush=True)

    model = Ranker()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log_path = os.path.join(OUT_DIR, f"p0b_{args.tag}.jsonl")
    logf = open(log_path, "a", buffering=1)
    logf.write(json.dumps({"cfg": vars(args), "n_train": len(tr), "n_val": len(va)}) + "\n")

    def feats(vkey):
        rgb, aud, gs = cache[vkey]
        x = np.concatenate([dequant(rgb), dequant(aud)], 1)
        return x, gs

    @torch.no_grad()
    def eval_vids(vid_list, rho):
        ps = []
        model.eval()
        for vkey in vid_list:
            x, gs = feats(vkey)
            s = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            n = max(1, int(round(len(s) * rho)))
            pred_top = set(np.argsort(-s)[:n].tolist())
            gt_top = set(np.argsort(-gs)[:n].tolist())
            ps.append(len(pred_top & gt_top) / n)
        model.train()
        return 100.0 * float(np.mean(ps))

    gstep, t0 = 0, time.time()
    for ep in range(args.epochs):
        order = np.random.permutation(len(tr))
        opt.zero_grad()
        for n_i, ki in enumerate(order):
            vkey = tr[ki]
            x, gs = feats(vkey)
            x = torch.from_numpy(x).unsqueeze(0)
            g = torch.from_numpy(gs)
            s = model(x)[0]
            if args.loss == "pairwise":
                loss = pairwise_loss(s, g)
            else:
                loss = torch.nn.functional.mse_loss(s, g)
            (loss / args.accum).backward()
            if (n_i + 1) % args.accum == 0 or n_i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                gstep += 1
            if (n_i + 1) % 1000 == 0:
                m15 = eval_vids(va, 0.15)
                m50 = eval_vids(va, 0.50)
                rec = {"tag": args.tag, "ep": ep, "step": n_i + 1, "loss": round(float(loss), 5),
                       "map15": round(m15, 2), "map50": round(m50, 2),
                       "min": round((time.time() - t0) / 60, 1)}
                print("P0B " + json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
        print(f"[p0b] epoch {ep} done min={((time.time() - t0) / 60):.1f}", flush=True)

    # final: single-shot test eval (test set touched once, full 2000)
    t15 = eval_vids(te_all, 0.15)
    t50 = eval_vids(te_all, 0.50)
    rec = {"tag": args.tag, "final": True, "test_map15": round(t15, 2), "test_map50": round(t50, 2)}
    print("P0B_FINAL " + json.dumps(rec), flush=True)
    logf.write(json.dumps(rec) + "\n")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"p0b_{args.tag}.pt"))
    logf.close()


if __name__ == "__main__":
    main()
