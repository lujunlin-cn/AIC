#!/usr/bin/env python3
"""R13 F1: TRINITY-recipe Mr.HiSum ranker with ablation switches.

Diffs vs p0b_mrhisum_rank.py (the 8-mAP reproduction-gap suspects):
  --depth {3,5}          TRINITY uses d5; P0-b used d3
  --loss {pairwise,mse,bce}  TRINITY recipe = equal-weight BCE (soft target
                         = normalized gtscore); P0-b used self-rolled pairwise
  --schedule {const,warmcos}  warmup+cosine vs P0-b constant lr
  --feat {rgb+audio,rgb,audio}  official baselines are rgb-1024 only
  --official-dequant     use q*4/255-2 (official yt8m_reader) instead of
                         P0-b's (q+0.5)*2/255-1
Evaluation is DUAL from now on: official seg-AP (5-clip mean -> softmax ->
AP vs binary top-50/15 mask) AND our precision@rho. Val-best ckpt by
official ap50 is saved; test is scored once, dual.

NPU: same recipe as P0-b (jit_compile=False, CPU-init weights).

Usage:
  .../python3.11 r13_f1_rank.py --device npu --loss bce --depth 5 \
      --schedule warmcos --feat rgb+audio --official-dequant --tag f1_main_s0
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
SPLIT = ("/data/aic/external_datasets/MrHiSum/annotations/upstream_repo/"
         "dataset/mr_hisum_split.json")
OUT_DIR = "/data/aic/experiments_910a/LFM_V11/r13_f1"

ROPE_THETA = 10000.0
PAIRS_PER_VIDEO = 64
PAIR_TAU = 0.10


def build_index():
    idx = {}
    with h5py.File(GT_H5, "r") as gt:
        gt_keys = set(gt.keys())
    for path in sorted(glob.glob(FEAT_GLOB)):
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k in gt_keys:
                    idx[k] = path
    return idx


def rope_cache(n, d, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(n, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
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
    def __init__(self, d, h, nh):
        super().__init__()
        self.nh, self.dh = nh, d // nh
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        qkv = self.qkv(self.n1(x)).view(b, t, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, t, -1)
        x = x + self.proj(o)
        x = x + self.mlp(self.n2(x))
        return x


class Ranker(nn.Module):
    def __init__(self, d_in, d_model=512, heads=8, depth=5):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([Block(d_model, d_model, heads) for _ in range(depth)])
        self.head = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dh = d_model // heads

    def forward(self, x):
        h = self.proj(x)
        cos, sin = rope_cache(h.shape[1], self.dh, h.device)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.norm(h)).squeeze(-1)


def pairwise_loss(scores, gs, k=PAIRS_PER_VIDEO, tau=PAIR_TAU):
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


def seg_mean_np(x, clip=5):
    n = (len(x) // clip) * clip
    return x[:n].reshape(-1, clip).mean(1) if n > 0 else np.asarray(x)


def topk_mask(scores, frac):
    n = max(1, int(len(scores) * frac))
    m = np.zeros(len(scores))
    m[np.argsort(-np.asarray(scores))[:n]] = 1
    return m


def ap_score(y_binary, scores):
    y = np.asarray(y_binary, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    return float((prec * ys).sum() / ys.sum()) if ys.sum() > 0 else 0.0


def preload_cache(vids, idx, feat, official_dequant):
    """vid -> (x (T,D) f32, gs (T,) f32). uint8 in RAM, dequant on the fly."""
    scale, off = (4.0 / 255.0, -2.0) if official_dequant else (2.0 / 255.0, -1.0)
    cache = {}
    by_path = {}
    for v in vids:
        by_path.setdefault(idx[v], []).append(v)
    for path, vk in by_path.items():
        with h5py.File(path, "r") as f:
            for v in vk:
                rgb = f[v]["rgb"][()]
                aud = f[v]["audio"][()] if feat == "rgb+audio" else None
                cache[v] = [rgb, aud]
    with h5py.File(GT_H5, "r") as gt:
        for v in vids:
            rgb, aud = cache[v]
            gs = gt[v]["gtscore"][()].astype(np.float32)
            t = min(len(rgb), len(gs))
            rgb = rgb[:t].astype(np.float32) * scale + off
            if feat == "rgb+audio":
                aud = aud[:t].astype(np.float32) * scale + off
                x = np.concatenate([rgb, aud], 1)
            elif feat == "rgb":
                x = rgb
            else:
                x = (aud[:t].astype(np.float32) * scale + off
                     if aud is not None else None)
            cache[v] = (x.astype(np.float32), gs[:t])
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["pairwise", "mse", "bce"], default="bce")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--feat", choices=["rgb+audio", "rgb", "audio"], default="rgb+audio")
    ap.add_argument("--schedule", choices=["const", "warmcos"], default="warmcos")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--official-dequant", action="store_true")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--val-subset", type=int, default=0, help="0=full val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", choices=["cpu", "npu"], default="npu")
    ap.add_argument("--npu-id", type=int, default=0,
                    help="physical card for --device npu (set_device, not .to)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8 if args.device == "npu" else 32)
    DEV = "cpu"
    if args.device == "npu":
        import torch_npu  # noqa: F401
        assert torch.npu.device_count() > 0, "no NPU"
        # card pinning must go through set_device; .to("npu:N") silently
        # lands on card 0 in some torch_npu builds (R13 lesson)
        torch.npu.set_device(args.npu_id)
        assert torch.npu.device_count() > 0
        print(f"[f1] pinned to npu:{args.npu_id} (visible={torch.npu.device_count()})",
              flush=True)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
        DEV = "npu"

    print("[f1] building index...", flush=True)
    idx = build_index()
    sp = json.load(open(SPLIT))
    tr = [k for k in sp["train_keys"] if k in idx]
    va_all = [k for k in sp["val_keys"] if k in idx]
    te_all = [k for k in sp["test_keys"] if k in idx]
    if args.subset:
        tr = tr[: args.subset]
    va = va_all[: args.val_subset] if args.val_subset else va_all
    print(f"[f1] train={len(tr)} val={len(va)} loss={args.loss} depth={args.depth} "
          f"feat={args.feat} sched={args.schedule}", flush=True)

    cache = preload_cache(sorted(set(tr) | set(va_all) | set(te_all)), idx,
                          args.feat, args.official_dequant)
    print(f"[f1] cache ready: {len(cache)} vids", flush=True)

    d_in = cache[tr[0]][0].shape[1]
    model = Ranker(d_in, args.d_model, args.heads, args.depth).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(tr) * args.epochs) // 8 + 1
    warm = max(1, int(total_steps * args.warmup_frac))
    sched = None
    if args.schedule == "warmcos":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(
                math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))))

    logf = open(os.path.join(OUT_DIR, f"f1_{args.tag}.jsonl"), "a", buffering=1)
    logf.write(json.dumps({"cfg": vars(args), "n_train": len(tr), "n_val": len(va)}) + "\n")

    def feats(vkey):
        return cache[vkey]

    def eval_vids(vid_list):
        """dual-metric eval: official seg-AP50/15 + our precision@50."""
        ap50s, ap15s, p50s = [], [], []
        model.eval()
        with torch.no_grad():
            for vkey in vid_list:
                x, gs = feats(vkey)
                s = model(torch.from_numpy(x).unsqueeze(0).to(DEV))[0].cpu().numpy()
                seg_s, seg_g = seg_mean_np(s), seg_mean_np(gs)
                sm = np.exp(seg_s - seg_s.max())
                sm = sm / sm.sum()
                ap50s.append(ap_score(topk_mask(seg_g, 0.5), sm))
                ap15s.append(ap_score(topk_mask(seg_g, 0.15), sm))
                n = max(1, int(round(len(s) * 0.5)))
                p50s.append(len(set(np.argsort(-s)[:n]) & set(
                    np.argsort(-gs)[:n])) / n)
        model.train()
        return 100 * float(np.mean(ap50s)), 100 * float(np.mean(ap15s)), \
            100 * float(np.mean(p50s))

    best = {"ap50": -1.0, "ep": -1}
    gstep, t0 = 0, time.time()
    for ep in range(args.epochs):
        order = np.random.permutation(len(tr))
        opt.zero_grad()
        a50e = -1.0
        for n_i, ki in enumerate(order):
            x, gs = feats(tr[ki])
            xt = torch.from_numpy(x).unsqueeze(0).to(DEV)
            g = torch.from_numpy(gs).to(DEV)
            s = model(xt)[0]
            if args.loss == "pairwise":
                loss = pairwise_loss(s, g)
            elif args.loss == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    s, g.clamp(0.0, 1.0))
            else:
                loss = torch.nn.functional.mse_loss(s, g)
            (loss / 8).backward()
            if (n_i + 1) % 8 == 0 or n_i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                gstep += 1
                if sched is not None:
                    sched.step()
            if (n_i + 1) % 500 == 0:
                a50, a15, p50 = eval_vids(va)
                rec = {"tag": args.tag, "ep": ep, "step": n_i + 1,
                       "loss": round(float(loss), 5), "ap50": round(a50, 2),
                       "ap15": round(a15, 2), "prec50": round(p50, 2),
                       "min": round((time.time() - t0) / 60, 1)}
                print("F1 " + json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n")
        print(f"[f1] epoch {ep} done min={(time.time() - t0) / 60:.1f}", flush=True)
        # per-epoch val-best selection (official ap50) — CPU budget: never
        # lose a good epoch to overshooting
        a50e, a15e, _ = eval_vids(va)
        rec = {"tag": args.tag, "ep_val": ep, "val_ap50": round(a50e, 2),
               "val_ap15": round(a15e, 2),
               "min": round((time.time() - t0) / 60, 1)}
        print("F1_EPOCHVAL " + json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        if a50e > best["ap50"]:
            best = {"ap50": a50e, "ep": ep}
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, f"f1_{args.tag}_valbest.pt"))

    # single-shot test on the val-best ckpt, dual metric
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"f1_{args.tag}_valbest.pt"),
                                     map_location=DEV))
    t50a, t15a, t50p = eval_vids(te_all)
    rec = {"tag": args.tag, "final": True, "best_ep": best["ep"],
           "test_official_ap50": round(t50a, 2), "test_official_ap15": round(t15a, 2),
           "test_prec50": round(t50p, 2), "val_ap50": round(best["ap50"], 2)}
    print("F1_FINAL " + json.dumps(rec), flush=True)
    logf.write(json.dumps(rec) + "\n")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"f1_{args.tag}.pt"))
    logf.close()


if __name__ == "__main__":
    main()
