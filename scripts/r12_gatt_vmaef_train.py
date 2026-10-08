#!/usr/bin/env python3
"""R12-GATT / R12-VMAE-F: PHD2-domain frame-slot keep ranking, feature is the
single variable.

Data contract (all on-disk):
  frags   : /data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl
            {video_id, src, t0, L, fps, n_highlight_iv}  (5,886 rows)
  labels  : PHD2 selections train+test GIF intervals per src (user->[{t0,t1}])
            slot j covers [t0 + L*j/8, t0 + L*(j+1)/8]; y[j]=1 if any interval
            overlaps  (GIF-union keep label, B3 supervision family)
  feats   : --feat vmae  /data/.../LFM_V11/phd2_vmae_feats/<frag>.npz  ord/rep [8,768]
            --feat lfm   /data/.../LFM_V11/r12_p0c/phd2_lfm_pooled/<frag>.npz pooled [8,768]
            (B3-representation family: LFM2VL vision tower, port of
             v8_live_extract.vision, with the siglip2 NPU patch applied)

Split: source-level 80/20 by hash(src)%10 (train/dev), leakage-free.
Model: TRINITY-style light ranker (768->512, 3-layer Transformer, 8 heads,
       1D RoPE) shared with scripts/p0b_mrhisum_rank.py.
Loss : --loss {pairwise, mse, bce}  (frozen by the P0-b final verdict)
Eval : dev slot-AP (predicted ranking vs positive slots), per-source mean.

Usage:
  python3 r12_gatt_vmaef_train.py --feat vmae --loss pairwise --tag vmae_pair
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

FRAG_INDEX = "/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl"
SEL_TRAIN = "/data/aic/external_datasets/PHD2/annotations/selections/train.json"
SEL_TEST = "/data/aic/external_datasets/PHD2/annotations/selections/test.json"
FEAT_DIRS = {
    "vmae": "/data/aic/experiments_910a/LFM_V11/phd2_vmae_feats",
    "lfm": "/data/aic/experiments_910a/LFM_V11/r12_p0c/phd2_lfm_pooled",
}
OUT_DIR = "/data/aic/experiments_910a/LFM_V11/r12_gatt"
D_MODEL, N_HEADS, N_LAYERS = 512, 8, 3
SLOTS = 8


# ---------------- model (same as p0b) ----------------
def rope_cache(n, d, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(n, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
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
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
        return x + self.mlp(self.n2(x + self.proj(o)))


class Ranker(nn.Module):
    def __init__(self, d_in=768):
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


# ---------------- data ----------------
def build_dataset(feat_kind):
    sel = {}
    for p in (SEL_TRAIN, SEL_TEST):
        sel.update(json.load(open(p)))
    rows, n_missing = [], 0
    fdir = FEAT_DIRS[feat_kind]
    for line in open(FRAG_INDEX):
        r = json.loads(line)
        fp = os.path.join(fdir, r["video_id"] + ".npz")
        if not os.path.exists(fp):
            n_missing += 1
            continue
        ivs = [iv for lst in sel.get(r["src"], {}).values() for iv in lst]
        t0, L = float(r["t0"]), float(r["L"])
        y = np.zeros(SLOTS, dtype=np.float32)
        for j in range(SLOTS):
            a, b = t0 + L * j / SLOTS, t0 + L * (j + 1) / SLOTS
            y[j] = float(any(iv["t1"] > a and iv["t0"] < b for iv in ivs))
        rows.append({"frag": r["video_id"], "src": r["src"], "y": y, "feat_path": fp})
    return rows, n_missing


def load_feat(path, feat_kind):
    z = np.load(path)
    if feat_kind == "vmae":
        return z["ord"].astype(np.float32)  # motion condition; rep is the control
    return z["pooled"].astype(np.float32)


def src_split(rows):
    tr, dv = [], []
    for i, r in enumerate(rows):
        (tr if int(hashlib.md5(r["src"].encode()).hexdigest(), 16) % 10 < 8 else dv).append(i)
    return tr, dv


# ---------------- losses ----------------
def pairwise_loss(scores, y, k=16):
    t = scores.shape[0]
    idx = torch.randint(0, t, (k * 4, 2))
    i, j = idx[:, 0], idx[:, 1]
    diff = y[i] - y[j]
    mask = diff != 0
    i, j, diff = i[mask], j[mask], diff[mask]
    if len(i) == 0:
        return scores.sum() * 0.0
    if len(i) > k:
        sel = torch.randperm(len(i))[:k]
        i, j, diff = i[sel], j[sel], diff[sel]
    sign = torch.where(diff > 0, 1.0, -1.0)
    return torch.nn.functional.softplus(-sign * (scores[i] - scores[j])).mean()


# ---------------- eval ----------------
@torch.no_grad()
def slot_ap(model, rows, idx, feat_kind):
    """Precision@pos with the same convention as p0b mAP@rho."""
    aps = []
    model.eval()
    for i in idx:
        r = rows[i]
        x = torch.from_numpy(load_feat(r["feat_path"], feat_kind)).unsqueeze(0)
        s = model(x)[0].numpy()
        pos = int((r["y"] > 0).sum())
        if pos == 0 or pos == SLOTS:
            continue
        pred_top = set(np.argsort(-s)[:pos].tolist())
        gt_top = set(np.where(r["y"] > 0)[0].tolist())
        aps.append(len(pred_top & gt_top) / pos)
    model.train()
    return 100.0 * float(np.mean(aps)) if aps else float("nan")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=["vmae", "lfm"], required=True)
    ap.add_argument("--loss", choices=["pairwise", "mse", "bce"], required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(16)

    rows, n_missing = build_dataset(args.feat)
    tr, dv = src_split(rows)
    print(f"[gatt:{args.feat}/{args.loss}] rows={len(rows)} missing_feat={n_missing} "
          f"train={len(tr)} dev={len(dv)}", flush=True)
    pos = float(np.mean([r["y"].mean() for r in rows]))
    print(f"[gatt] slot keep-rate={pos:.3f}", flush=True)

    model = Ranker()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    logf = open(os.path.join(OUT_DIR, f"{args.tag}.jsonl"), "a", buffering=1)
    logf.write(json.dumps({"cfg": vars(args), "n_rows": len(rows), "n_train": len(tr),
                           "n_dev": len(dv), "keep_rate": round(pos, 4)}) + "\n")

    t0, best = time.time(), -1.0
    for ep in range(args.epochs):
        order = np.random.permutation(tr)
        ep_loss, nb = 0.0, 0
        for i in order:
            r = rows[i]
            x = torch.from_numpy(load_feat(r["feat_path"], args.feat)).unsqueeze(0)
            y = torch.from_numpy(r["y"])
            s = model(x)[0]
            if args.loss == "pairwise":
                loss = pairwise_loss(s, y)
            elif args.loss == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(s, y)
            else:
                loss = torch.nn.functional.mse_loss(s, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            nb += 1
        d_ap = slot_ap(model, rows, dv, args.feat)
        t_ap = slot_ap(model, rows, tr[:200], args.feat)
        rec = {"tag": args.tag, "ep": ep, "loss": round(ep_loss / max(nb, 1), 5),
               "dev_ap": round(d_ap, 2), "train_ap": round(t_ap, 2),
               "min": round((time.time() - t0) / 60, 1)}
        print("GATT " + json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        if d_ap == d_ap and d_ap > best:
            best = d_ap
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{args.tag}_best.pt"))

    rec = {"tag": args.tag, "final": True, "best_dev_ap": round(best, 2)}
    print("GATT_FINAL " + json.dumps(rec), flush=True)
    logf.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
