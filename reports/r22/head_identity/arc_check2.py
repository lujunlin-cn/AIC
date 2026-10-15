# Does the production VPB head mostly re-read the position arcline?
# Loads the 10-seed production heads on cached features, computes holdout
# predictions, then: (1) re-confirm the holdout spearman valid-segment count,
# (2) how much of the head output is explained by the arcline (mean per-seg
# corr of head output vs curve), (3) net content contribution: spearman of
# (pred - curve) vs (gif - curve) on the non-all-zero holdout segments.
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

DEV = 'cuda' if os.environ.get('VP_DEV') == 'cuda' and torch.cuda.is_available() else 'cpu'
V = '/root/autodl-tmp/vpb'
A = '/root/autodl-tmp/vpb_artifacts'

rows = [json.loads(l) for l in open(f'{A}/sft_data_v2.jsonl')]
gif = np.array([r['gif_hard'] for r in rows], dtype=np.float64)
tgt = np.array([r['_target'] for r in rows], dtype=np.float64)
N = len(rows)
hold = np.random.default_rng(13).permutation(N)[:125]
tr = np.ones(N, bool); tr[hold] = False

# cached features, same dedup+order as training
feat, vids = [], []
for f in sorted(glob.glob(f'{A}/feat/shard_*of*.npz')):
    z = np.load(f)
    vids.extend(z['vid'].tolist() if hasattr(z['vid'], 'tolist') else list(z['vid']))
    feat.append(z['feat_temporal'])
feat = np.concatenate(feat)
seen = {}
for i, v in enumerate(vids):
    seen[v] = i
keep = np.array(sorted(seen.values()))
feat, vids = feat[keep], [vids[i] for i in keep]
vid2row = {r['vid']: i for i, r in enumerate(rows)}
order = np.argsort(np.array([vid2row[v] for v in vids]))
feat = feat[order]

curve_t = tgt[tr].mean(0)   # arcline fitted on TRAIN only
curve_g = gif[tr].mean(0)

d = feat.shape[-1]
mu = feat[tr].reshape(-1, d).mean(0)
sd = feat[tr].reshape(-1, d).std(0) + 1e-6
nrm = lambda a: (a - mu) / sd
Xh = torch.from_numpy(nrm(feat[hold]).astype(np.float32)).to(DEV)


class HeadMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a)); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0; i = j + 1
    return r


def spear(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


# 10-seed production ensemble on holdout
preds = []
for seed in range(13, 23):
    ck = torch.load(f'{V}/heads/vp_head_temporal_gif_hard_mlp_s{seed}.pt',
                    map_location='cpu', weights_only=False)
    hd = HeadMLP(d).to(DEV).eval()
    hd.load_state_dict(ck['state'])
    with torch.no_grad():
        preds.append(hd(Xh).cpu().numpy())
pred = np.mean(preds, 0)          # [125,12]

curve_broadcast = np.broadcast_to(curve_g, pred.shape)
per_seg_corr = np.array([np.corrcoef(pred[i], curve_g)[0, 1]
                         for i in range(len(hold)) if np.std(pred[i]) > 1e-9])
gif_h = gif[hold]  # [125,12] -- all segment metrics index THIS, not gif[i]
nz = np.array([np.std(g) > 1e-9 for g in gif_h])       # non-all-zero segs
sp_pred = [spear(pred[i], gif_h[i]) for i in range(len(hold)) if nz[i]]
sp_curve = [spear(curve_g, gif_h[i]) for i in range(len(hold)) if nz[i]]

# net content: rank of (pred - curve) vs (gif - curve) per segment
sp_resid = [spear(pred[i] - curve_g, gif_h[i] - curve_g)
            for i in range(len(hold)) if nz[i]]

print(f"non-all-zero holdout segs: {int(nz.sum())}/{len(hold)}")
print(f"pred ens10 spearman vs gif = {np.nanmean(sp_pred):.4f} ({int(nz.sum())}-seg subset)")
print(f"arcline spearman vs gif    = {np.nanmean(sp_curve):.4f}")
print(f"pred<-arcline re-read corr = {np.nanmean(per_seg_corr):.4f} "
      f"(mean per-seg corr of head output vs arcline)")
print(f"residual spearman (pred-curve vs gif-curve) = {np.nanmean(sp_resid):.4f}")
print('ARC2_DONE')
