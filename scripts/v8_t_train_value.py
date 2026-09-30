"""V8 T-line: domain-internal frame-value head.

Regresses the DHF1K human saliency value (domain-internal temporal GT, source
level separated: train 001-030+621-700, select 601-620, diag 031-100) from
the V7 cached LFM pooled features on the same 1s keyframe grid.  TCN as in
V7 (394,241 params); trained on CPU per the V7 NPU small-operator lesson.

Purpose: give the student a temporal keep/drop signal that lives in the same
domain as the crops (unlike TVSum).  Downstream: synthetic joint f1 evaluation
on DIAG with saliency-top frames as GT (v8_joint_eval with a keep mask).
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--feats', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/feats'))
ap.add_argument('--saliency', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/dhf1k_saliency_value.jsonl'))
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=1500)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--device', default='cpu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

sal = {json.loads(l)['vid']: np.array(json.loads(l)['values']) for l in open(args.saliency)}


def load_split(lo_hi):
    seqs = []
    for vid, vals in sorted(sal.items()):
        n = len(vals)
        if not (lo_hi[0] <= vid <= lo_hi[1]):
            continue
        fdir = args.feats / vid
        kfs = sorted(int(p.stem) for p in fdir.glob('*.npz'))
        kfs = [k for k in kfs if k < n]
        if len(kfs) < 8:
            continue
        x = np.stack([np.load(fdir / f'{k}.npz')['pooled'].astype(np.float32) for k in kfs])  # (T,768)
        v = vals[kfs].astype(np.float32)
        # rank-percentile label within the sequence: robust to the heavy
        # zero-skew of raw saliency means and directly aligned with ranking
        order = np.argsort(np.argsort(v))
        y = ((order + 0.5) / len(v)).astype(np.float32)
        seqs.append((vid, x, y, v))
    return [(a, b, c) for a, b, c, _ in seqs], [(a, b, v) for a, b, _, v in seqs]


train = load_split(('001', '030'))[0] + load_split(('621', '700'))[0]
dev, dev_raw = load_split(('601', '620'))
diag, diag_raw = load_split(('031', '100'))
print(f'seqs train={len(train)} dev={len(dev)} diag={len(diag)}', flush=True)


class Block(torch.nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.f = torch.nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.g = torch.nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.out = torch.nn.Conv1d(ch, ch, 1)
        self.ln = torch.nn.LayerNorm(ch)

    def forward(self, x):  # (B,C,T)
        a = torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))
        y = self.out(a) + x
        return self.ln(y.transpose(1, 2)).transpose(1, 2)


class TCN(torch.nn.Module):
    def __init__(self, d=768, ch=128):
        super().__init__()
        self.inp = torch.nn.Conv1d(d, ch, 1)
        self.blocks = torch.nn.ModuleList([Block(ch, 2 ** i) for i in range(6)])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):  # (B,T,D)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2).squeeze(-1)  # (B,T)


model = TCN().to(args.device)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


def run_eval(seqs, raws):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for (v, x, y), (_, _, raw) in zip(seqs, raws):
            p = model(torch.from_numpy(x[None]).to(args.device))[0].cpu().numpy()
            ps.append(p)
            ys.append(raw)  # evaluate against raw saliency values
    model.train()
    pc = np.concatenate(ps)
    yc = np.concatenate(ys)
    pear = float(np.corrcoef(pc, yc)[0, 1])
    rho = float(np.corrcoef(np.argsort(np.argsort(pc)), np.argsort(np.argsort(yc)))[0, 1])
    ndcg = []
    for p, y in zip(ps, ys):
        k = max(1, int(round(len(y) * 0.15)))
        top = set(np.argsort(-y)[:k].tolist())
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(np.argsort(-p)[:k]) if i in top)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(k))
        ndcg.append(dcg / idcg)
    return pear, rho, float(np.mean(ndcg))


WIN = 16  # fixed-length random window per training draw (sequences vary in T)


def rand_window(x, y):
    if len(y) <= WIN:
        i = 0
        xx, yy = x, y
    else:
        i = int(rng.integers(0, len(y) - WIN + 1))
        xx, yy = x[i:i + WIN], y[i:i + WIN]
    if len(yy) < WIN:  # pad by repeating the tail
        pad = WIN - len(yy)
        xx = np.concatenate([xx, np.repeat(xx[-1:], pad, 0)])
        yy = np.concatenate([yy, np.repeat(yy[-1:], pad)])
    return xx, yy


t0 = time.time()
hist, best = [], (-9.0, None)
for step in range(args.steps):
    idx = rng.integers(0, len(train), args.batch)
    xs, ys = zip(*[rand_window(train[i][1], train[i][2]) for i in idx])
    xb = torch.from_numpy(np.stack(xs)).to(args.device)
    yb = torch.from_numpy(np.stack(ys)).to(args.device)
    pred = model(xb)
    d = pred - yb
    ad = d.abs()
    loss = torch.where(ad <= 0.1, 0.5 * d * d, 0.1 * (ad - 0.05)).mean()  # NPU-safe huber (cpu here)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % 100 == 0:
        pear, rho, nd = run_eval(dev, dev_raw)
        hist.append({'step': step + 1, 'loss': float(loss), 'dev_pearson': pear, 'dev_spearman': rho, 'dev_ndcg15': nd})
        print(f'STEP {step + 1} loss={float(loss):.4f} dev pear={pear:.4f} rho={rho:.4f} ndcg15={nd:.4f}', flush=True)
        score = rho + nd
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

args.output_dir.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': best[1], 'config': {'ch': 128, 'd': 768, 'seed': args.seed}},
           args.output_dir / 'value_head.pt')
model.load_state_dict(best[1])
out = {'seed': args.seed, 'steps': args.steps, 'params': n_par, 'dev_hist': hist,
       'final': {'dev': dict(zip(('pearson', 'spearman', 'ndcg15'), run_eval(dev, dev_raw))),
                 'diag': dict(zip(('pearson', 'spearman', 'ndcg15'), run_eval(diag, diag_raw)))},
       'wall_s': round(time.time() - t0, 1)}
(args.output_dir / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
print('FINAL', json.dumps(out['final']), flush=True)
