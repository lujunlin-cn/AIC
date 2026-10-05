"""R6 S-REREAD shortlist diagnosis: top-k x grid-density regret curve.
Finds the smallest candidate set whose oracle regret passes the <=0.01 gate.
CPU only; reads the same dev124 sample cache. Output: one printed table.
"""
import sys
import numpy as np
import torch
sys.path.insert(0, '/root/AIC')
d = np.load('/data/aic/experiments_910a/LFM_V8/samples/live_dev.npz', allow_pickle=True)
feat, u = d['feat'], d['u'].astype(np.float32)
vid = d['vid'].astype(str)
N, NC = u.shape
from scripts.v11_r6_s_shortlist_oracle import Head            # noqa: E402
ck = torch.load('/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt',
                map_location='cpu', weights_only=False)
head = Head(feat.shape[-1], NC).float()
head.load_state_dict(ck['state_dict']); head.eval()
torch.set_grad_enabled(False)
sc = np.concatenate([head(torch.from_numpy(feat[i:i + 64]).float()).numpy()
                     for i in range(0, N, 64)], 0)
u_max = u.max(1)


def regret(topk, grid):
    regs = np.empty(N, np.float32)
    for i in range(N):
        sl = sorted(set(np.argsort(-sc[i])[:topk].tolist()) | set(range(0, 129, grid)))
        regs[i] = u_max[i] - u[i][sl].max()
    vs = sorted(set(vid))
    idx = {v: [i for i in range(N) if vid[i] == v] for v in vs}
    rng = np.random.RandomState(99)
    ms = []
    for _ in range(1000):
        pick = rng.choice(len(vs), len(vs), replace=True)
        ms.append(float(np.mean([regs[i] for p in pick for i in idx[vs[p]]])))
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return regs.mean(), lo, hi, float((regs <= 0.01).mean())


print('config                      mean    CI95            frac(regret<=0.01)')
for topk in (1, 3, 5, 8):
    for grid, label in ((16, 'g9'), (8, 'g17'), (4, 'g33'), (2, 'g65')):
        m, lo, hi, p = regret(topk, grid)
        ns = len(set(range(0, 129, grid)))
        print(f'top{topk}+{label:4s}(~{topk + ns:2d} cand)   {m:.4f}  '
              f'[{lo:.4f},{hi:.4f}]   {p:.3f}', flush=True)
