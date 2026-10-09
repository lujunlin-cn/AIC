#!/usr/bin/env python3
"""R13 F1 step-1: rescore existing P0-b checkpoints under the OFFICIAL
Mr.HiSum mAP convention, zero training.

Official evaluate (solver.evaluate + utils/evaluate_map.py):
  frame scores -> uniform_clip=5 segment mean -> softmax
  gt gtscore   -> same segment mean -> binary top-50% / top-15% mask
  mAP = sklearn average_precision_score(binary_mask, softmax_scores) x 100

Our P0-b convention: per-video precision of top-rho% predicted vs top-rho%
gtscore frames, x100.  59.17 (ours) vs 67.20 (official) mixed these two.

Also rescored: no-val-best selection here (final-epoch ckpts only) --
isolates the metric convention alone.
"""
import glob
import json
import os
import signal
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/root/AIC")
sys.path.insert(0, "/data/aic/external_datasets/MrHiSum/annotations/upstream_repo")

GT_H5 = "/data/aic/external_datasets/MrHiSum/annotations/mr_hisum.h5"
FEAT_GLOB = "/data/aic/external_datasets/MrHiSum/processed/yt8m/train*.h5"
SPLIT = "/data/aic/external_datasets/MrHiSum/annotations/upstream_repo/dataset/mr_hisum_split.json"
CKPT_DIR = "/data/aic/experiments_910a/LFM_V11/r12_p0b"
OUT = "/data/aic/experiments_910a/LFM_V11/r13_f1/rescore_p0b.json"

# ---- our P0-b head (must match training script) ----
D_MODEL, N_HEADS, N_LAYERS = 512, 8, 3


def dequant(a):
    return (a.astype(np.float32) + 0.5) * 2.0 / 255.0 - 1.0


def rope_cache(n, d, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=device).float() / d))
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


class Block(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.nh, self.dh = h, d // h
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d))
        self.n1, self.n2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        qkv = self.qkv(self.n1(x)).view(b, t, 3, N_HEADS, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
        return x + self.mlp(self.n2(x + self.proj(o)))


class Ranker(torch.nn.Module):
    def __init__(self, d_in=1152):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, D_MODEL)
        self.blocks = torch.nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.head = torch.nn.Linear(D_MODEL, 1)
        self.norm = torch.nn.LayerNorm(D_MODEL)

    def forward(self, x):
        h = self.proj(x)
        cos, sin = rope_cache(h.shape[1], D_MODEL // N_HEADS, h.device)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.norm(h)).squeeze(-1)


def seg_mean(x, clip=5):
    n = (len(x) // clip) * clip
    return x[:n].reshape(-1, clip).mean(1)


def ap_score(y_binary, scores):
    """sklearn average_precision_score, dependency-free reimplementation."""
    y = np.asarray(y_binary, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    total_pos = y.sum()
    return float((precision * y_sorted).sum() / total_pos) if total_pos > 0 else 0.0


def build_feat_index():
    import glob as g
    idx = {}
    with h5py.File(GT_H5, "r") as gt:
        gt_keys = set(gt.keys())
    for path in sorted(g.glob(FEAT_GLOB)):
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if k in gt_keys:
                    idx[k] = path
    return idx


def main():
    dev = "cpu"
    torch.set_num_threads(32)
    idx = build_feat_index()
    sp = json.load(open(SPLIT))
    te = [k for k in sp["test_keys"] if k in idx]
    print(f"test vids: {len(te)}", flush=True)

    # preload features
    feats = {}
    by_path = {}
    for v in te:
        by_path.setdefault(idx[v], []).append(v)
    for _pi, (path, vk) in enumerate(by_path.items()):
        if _pi % 200 == 0:
            print(f"preload shard {_pi}/{len(by_path)}", flush=True)
        signal.alarm(20)
        with h5py.File(path, "r") as f:
            pass
        signal.alarm(0)
        with h5py.File(path, "r") as f:
            for v in vk:
                feats[v] = f[v]["rgb"][()], f[v]["audio"][()]
    with h5py.File(GT_H5, "r") as gt:
        gts = {v: gt[v]["gtscore"][()].astype(np.float32) for v in te}
    for v in te:
        rgb, aud = feats[v]
        t = min(len(rgb), len(gts[v]))
        feats[v] = np.concatenate([dequant(rgb[:t]), dequant(aud[:t])], 1)

    torch.set_num_threads(32)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "p0b_pair_npu_s*.pt")))
    rep = {"official_ap50": [], "official_ap15": [], "our_prec50": []}
    for ck in ckpts:
        model = Ranker()
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()
        ap50s, ap15s, p50s = [], [], []
        with torch.no_grad():
            for v in te:
                x = torch.from_numpy(feats[v]).unsqueeze(0)
                s = model(x)[0].numpy()
                g = gts[v][: len(s)]
                # official convention
                seg_s = seg_mean(s)
                seg_g = seg_mean(g)
                soft = np.exp(seg_s - seg_s.max())
                soft = soft / soft.sum()
                ap50s.append(ap_score(np.array(_topk_mask(seg_g, 0.5)), soft))
                ap15s.append(ap_score(np.array(_topk_mask(seg_g, 0.15)), soft))
                # our convention
                n = max(1, int(round(len(s) * 0.5)))
                p50s.append(len(set(np.argsort(-s)[:n]) & set(np.argsort(-g)[:n])) / n)
        ap50, ap15, p50 = 100 * np.mean(ap50s), 100 * np.mean(ap15s), 100 * np.mean(p50s)
        rep["official_ap50"].append(round(ap50, 2))
        rep["official_ap15"].append(round(ap15, 2))
        rep["our_prec50"].append(round(p50, 2))
        print(f"{os.path.basename(ck)}: OFFICIAL ap50={ap50:.2f} ap15={ap15:.2f} | our prec50={p50:.2f}", flush=True)
    rep["mean"] = {k: round(float(np.mean(v)), 2) for k, v in rep.items() if isinstance(v[0], float) if v}
    rep["reference"] = {"official_paper_ap": {"map15": 50.80, "map50": 67.20},
                        "our_p0b_prec50_4seed": 59.17}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print("RESCORE_DONE")


def _topk_mask(scores, frac):
    n = max(1, int(len(scores) * frac))
    m = np.zeros(len(scores))
    m[np.argsort(-np.asarray(scores))[:n]] = 1
    return m


if __name__ == "__main__":
    main()
