#!/usr/bin/env python3
"""R15b 判决实验第二半：均匀切段池上，冻结 VPB 特征 + MLP 头还能不能学出排序？

与 vp_ref/judge_unfreeze.py 的 freeze=0 控制组**完全同配方**（同架构、同损失、
同优化器、同早停、同 seed），唯一变量是训练池。因此读数可直接对比：

    原池（高光窗居中）  holdout = 0.5671
    均匀切段池           holdout = ?

判读（诊断记录 §判决）：
  - 均匀池 holdout ≫ 0.5671  → 内容信号被切段器掩盖，L 可救，VPB 线值得继续投
  - 均匀池 holdout ≈ 0.5671  → 无差别，说明 0.5671 主要来自位置先验而非内容
  - 均匀池 holdout ≪ 0.5671  → 内容信号本身不存在，35.52 是终局

用法: r15b_train_head_uniform.py <pool.jsonl> <frames_npy> <base_ckpt> <out_json> [seed]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POOL, FRAMES = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
CKPT = os.path.abspath(sys.argv[3])
OUT = os.path.abspath(sys.argv[4])
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 13
DEV = "cuda"
VP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VP_ROOT)
from videoprism_torch import build_base  # noqa: E402

REF_FROZEN_HOLDOUT = 0.5671  # 原池 judge_unfreeze freeze=0 控制组


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
    return r


def spear(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def pair_rank_loss(s, y, margin=0.02):
    t = s.shape[1]
    ii, jj = torch.triu_indices(t, t, offset=1)
    sgn = torch.sign(y[:, ii] - y[:, jj])
    m = sgn != 0
    if m.sum() == 0:
        return s.sum() * 0.0
    return torch.relu(margin - (s[:, ii] - s[:, jj]) * sgn)[m].mean()


class HeadMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


rows = [json.loads(l) for l in open(POOL)]
hard = np.array([r["gif_hard"] for r in rows], dtype=np.float32)
N = len(rows)
print(f"pool N={N}  non-all-zero={int((hard.sum(1) > 0).sum())}")

rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
n_hold = max(125, N // 5)
hold, tr = perm[:n_hold], perm[n_hold:]
tr_mask = np.zeros(N, bool); tr_mask[tr] = True
nz_hold = int((hard[hold].sum(1) > 0).sum())
print(f"holdout={len(hold)} (其中非全零 {nz_hold})  train={len(tr)}")

MM = np.load(FRAMES, mmap_mode="r")
DIM = 768
m = build_base(ckpt_path=CKPT).to(DEV).eval()
for p in m.parameters():
    p.requires_grad_(False)


def feats(idxs):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(idxs), 16):
            sel = idxs[k:k + 16]
            x = torch.from_numpy(np.ascontiguousarray(MM[sel])).to(DEV)
            x = x.float() / 255.0
            out.append(m(x).float().mean(2).cpu())
    return torch.cat(out)


F_all = feats(np.arange(N))
mu = F_all[tr_mask].reshape(-1, DIM).mean(0)
sd = F_all[tr_mask].reshape(-1, DIM).std(0) + 1e-6
X = ((F_all - mu.view(1, 1, -1)) / sd.view(1, 1, -1)).to(DEV)
Y = torch.from_numpy(hard).to(DEV)
del F_all, MM

torch.manual_seed(SEED)
hd = HeadMLP(DIM).to(DEV)
opt = torch.optim.Adam(hd.parameters(), 1e-3)
best, best_state, patience = -1, None, 0
tr_idx = np.where(tr_mask)[0]
for ep in range(40):
    hd.train()
    p_ = np.random.permutation(len(tr_idx))
    for k in range(0, len(tr_idx), 8):
        sel = tr_idx[p_[k:k + 8]]
        s = hd(X[sel])
        loss = F.mse_loss(s, Y[sel]) + 0.5 * pair_rank_loss(s, Y[sel])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    hd.eval()
    with torch.no_grad():
        pr = hd(X[hold]).cpu().numpy()
    sp = [spear(pr[i], hard[hold][i]) for i in range(len(hold))
          if hard[hold][i].std() > 1e-9]
    cur = float(np.mean(sp))
    if cur > best + 1e-4:
        best, patience = cur, 0
        best_state = {k: v.detach().cpu().clone() for k, v in hd.state_dict().items()}
    else:
        patience += 1
        if patience >= 8:
            break

res = {"pool": POOL, "n": N, "n_hold": int(len(hold)), "nz_hold": nz_hold,
       "holdout": best, "ref_frozen_orig_pool": REF_FROZEN_HOLDOUT,
       "best_ep": ep + 1}
print(f"\nuniform-pool holdout = {best:.4f}  (orig frozen pool = "
      f"{REF_FROZEN_HOLDOUT}, best_ep={ep + 1})")
print(f"delta = {best - REF_FROZEN_HOLDOUT:+.4f}")
if best > REF_FROZEN_HOLDOUT + 0.05:
    print("VERDICT: 内容信号被切段器掩盖 —— L 可救，VPB 线值得继续投")
elif best < REF_FROZEN_HOLDOUT - 0.05:
    print("VERDICT: 内容信号本身不存在 —— 35.52 为终局")
else:
    print("VERDICT: 无显著差异 —— 0.5671 主要来自位置先验而非内容")
json.dump(res, open(OUT, "w"), indent=1)
torch.save({"state": best_state, "norm": (mu.numpy(), sd.numpy()),
            "feat": "temporal", "target": "gif_hard_uniform", "arch": "mlp",
            "seed": SEED, "holdout_spearman": best},
           OUT.replace(".json", "_head.pt"))
print("TRAIN_UNIFORM_DONE")
