"""R8 temporal-head DUAL-BASE comparison (user decision 2026-10-06, after the
P/O/L run9 GO): train the SAME head from scratch on the two feature bases and
pick the temporal-head backbone on dev, BEFORE the preregistered r7 step-3
gate (temporal F +0.007 vs champion on the common pool) is attempted.

Bases (frozen encoders, feature_contract read layers already recorded):
  A  old backbone   /data/.../QVH_V10/frag_feats_train        (8,768)/window
  B  InternVideo2-1B /data/.../LFM_V11/r8_iv2/iv2_frag_feats_train

Frozen protocol (recorded before launch):
  windows  the 9,066 aligned windows of run9; identical set for both bases
           (asserted); labels = window-level mean saliency (run9 recipe);
           source-level 80/20 split seed 77 (identical to run9)
  head     TCN identical to run9 (proj 768->128, dils 1/2/4/8/16, out 1),
           trained FROM SCRATCH per r7 ("old head weights are not a control")
  train    lr 3e-4 (run9 scan), 3,000 steps batch 8 (3.3x run9 budget),
           seeds 510/511/512, per-base single process, two processes in
           parallel (parallel-first rule), incremental JSON every seed
  primary  POOLED dev Spearman over ALL dev windows (pred = mean of the 8
           frame logits, label = window scalar), with SOURCE-CLUSTER
           bootstrap CI (resample sources, not windows - fixes run9's
           8-source power limitation)
  secondary  per-source mean Spearman (run9 readout, 8-src subset)
  verdict  A-B primary paired cluster-bootstrap CI: lower>0 -> old base;
           upper<0 -> IV2 base; crossing 0 -> record equivalent, default to
           the higher primary mean (pre-registered here)
  checks   window-set equality, label equality, feature shape (8,768)

Usage: python3 scripts/r8_temporal_dualbase.py --base A --out .../dualA.json
"""
import argparse, glob, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True, choices=['A', 'B'])
ap.add_argument('--feat-a', default='/data/aic/experiments_910a/QVH_V10/frag_feats_train')
ap.add_argument('--feat-b', default='/data/aic/experiments_910a/LFM_V11/r8_iv2/iv2_frag_feats_train')
ap.add_argument('--frag-root', default='/data/aic/experiments_910a/QVH_V10/frag_train')
ap.add_argument('--out', required=True)
ap.add_argument('--seeds', nargs='*', type=int, default=[510, 511, 512])
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--split-seed', type=int, default=77)
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()

FEAT = args.feat_a if args.base == 'A' else args.feat_b
OTHER = args.feat_b if args.base == 'A' else args.feat_a

# base A is FLAT (8,998 npz), base B is sharded p0/p1/p2 (9,066): glob both
# layouts; train only on the INTERSECTION so both processes see the same
# window set (the merge script asserts equality)
other_files = {Path(fp).stem
               for fp in glob.glob(f'{OTHER}/*.npz') + glob.glob(f'{OTHER}/p*/*.npz')}


class TCN(torch.nn.Module):
    """Identical to run9's TCN (r8_iv2_pol.py)."""

    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4, 8, 16)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def soft_at(t, gt_t, gt_soft):
    i = np.clip(np.searchsorted(gt_t, t) - 1, 0, len(gt_t) - 2)
    w = np.clip((t - gt_t[i]) / np.maximum(gt_t[i + 1] - gt_t[i], 1e-6), 0, 1)
    return gt_soft[i] * (1 - w) + gt_soft[i + 1] * w


def spearman(s, y):
    if float(np.std(s)) < 1e-6 or float(np.std(y)) < 1e-6:
        return 0.0
    rs = np.argsort(np.argsort(s)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rs = (rs - rs.mean()) / max(rs.std(), 1e-9)
    ry = (ry - ry.mean()) / max(ry.std(), 1e-9)
    return float((rs * ry).mean())


t0 = time.time()
wins, Xl, Yl, Gl = [], [], [], []
seen = set()
for fp in sorted(glob.glob(f'{FEAT}/*.npz') + glob.glob(f'{FEAT}/p*/*.npz')):
    win = Path(fp).stem
    if win in seen:
        continue   # 66 windows were written to TWO shards (2026-10-06 probe)
    seen.add(win)
    lp = Path(args.frag_root) / 'frag_labels' / f'{win}.npz'
    if not lp.exists() or win not in other_files:
        continue
    z = np.load(fp)
    X = z['pooled'].astype(np.float32)
    if X.ndim != 2 or X.shape[1] != 768:
        continue
    lz = np.load(lp)
    y = float(np.mean(soft_at(z['t'].astype(np.float32),
                              lz['t'].astype(np.float32),
                              lz['saliency'].astype(np.float32))))
    wins.append(win)
    Xl.append(X)
    Yl.append(y)
    Gl.append(win.rsplit('_q', 1)[0])
print(f'{args.base}: windows {len(wins)} srcs {len(set(Gl))} ({time.time()-t0:.0f}s)', flush=True)

rec = {'protocol': 'R8 temporal dual-base; TCN from scratch; pooled dev '
                   'Spearman + source-cluster bootstrap; verdict rule '
                   'pre-recorded in docstring',
       'base': args.base, 'feat': FEAT, 'windows': len(wins),
       'steps': args.steps, 'lr': args.lr, 'seeds': args.seeds,
       'status': 'RUNNING', 'per_seed': []}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')

rng = np.random.RandomState(args.split_seed)
srcs = sorted(set(Gl))
rng.shuffle(srcs)
tr_set = set(srcs[:int(0.8 * len(srcs))])
tr = [i for i, g in enumerate(Gl) if g in tr_set]
dv = [i for i, g in enumerate(Gl) if g not in tr_set]
dv_srcs = sorted(set(Gl[i] for i in dv))
dv_by_src = {g: [i for i in dv if Gl[i] == g] for g in dv_srcs}
print(f'{args.base}: train {len(tr)} dev {len(dv)} ({len(srcs)-len(dv_srcs)} vs {len(dv_srcs)} srcs)', flush=True)

for sd in args.seeds:
    ts = time.time()
    torch.manual_seed(sd)
    run_rng = np.random.RandomState(sd)
    model = TCN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        idxs = run_rng.choice(len(tr), args.batch)
        opt.zero_grad()
        for j in idxs:
            logits = model(torch.from_numpy(Xl[tr[j]])[None])[0]
            loss = torch.nn.functional.mse_loss(
                logits, torch.full_like(logits, Yl[tr[j]]),
                reduction='mean') / args.batch
            loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f'{args.base} s{sd} step {step+1}/{args.steps}', flush=True)
    model.eval()
    with torch.no_grad():
        pred = {i: float(np.mean(model(torch.from_numpy(Xl[i])[None])[0].numpy()))
                for i in dv}
    pooled = spearman(np.array([pred[i] for i in dv]),
                      np.array([Yl[i] for i in dv]))
    per_src = [spearman(np.array([pred[i] for i in dv_by_src[g]]),
                        np.array([Yl[i] for i in dv_by_src[g]]))
               for g in dv_srcs if len(dv_by_src[g]) >= 3]
    entry = {'seed': sd,
             'pooled_spearman': round(pooled, 5),
             'per_source_mean': round(float(np.mean(per_src)), 4) if per_src else None,
             'n_per_source_srcs': len(per_src),
             'train_min': round((time.time() - ts) / 60, 1),
             'pred': {wins[i]: round(pred[i], 6) for i in dv}}
    rec['per_seed'].append(entry)
    Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
    print(f"{args.base} seed {sd}: pooled {pooled:.4f} per-src {entry['per_source_mean']}", flush=True)

rec['status'] = 'DONE'
Path(args.out).write_text(json.dumps(rec, indent=1) + '\n')
print('WROTE', args.out, flush=True)
