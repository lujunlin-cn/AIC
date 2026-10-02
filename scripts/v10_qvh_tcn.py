"""Temporal saliency head trained on QVHighlights dense annotations.

Why a separate trainer from v9_phd2_tcn.py: that one consumes fixed-length
8-frame PHD2 fragments.  QVHighlights is the opposite regime - full ~150 s
clips labelled on a dense 1 Hz grid - which is exactly the temporal structure
the semifinal set needs (median 14.1 s, and the time axis covers 100 % of the
426 videos where the spatial axis only covers 62.7 %).

Two supervision signals coexist, and the loss can weight each:

  * saliency  -- dense annotator-mean highlight-worthiness, 0..1.  This is the
    richer target: a graded importance score per second, the same dense
    objective every QVHighlights model (Moment-DETR onward) regresses.  It
    directly supports the keep-ratio frame ranking the deployment mask needs.
  * soft      -- exp(-d/tau) distance to the nearest relevant_window, the same
    weak-interval label the PHD2 TCN used.  Lets the head stay comparable with
    the PHD2 head on shared eval.

Variable-length sequences are padded to the split max and masked, so the
loss only touches real timesteps.  Source = video is the resampling unit for
the held-out split (QV assigns ~1 query per video so qid split == vid split).

Runs on CPU: the input is the pooled 768-d vector per second, so a batch of
B x 150 x 768 is small and the NPU is left for the teacher / feature jobs.
"""
import argparse, json, os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')   # small CPU pool is fine; do not starve it to 1

from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, required=True, help='dir of <vid>.npz pooled')
ap.add_argument('--label-root', type=Path, required=True, help='dir of <vid>.npz labels')
ap.add_argument('--val-feat-root', type=Path, default=None)
ap.add_argument('--val-label-root', type=Path, default=None,
                help='when set, val comes from this split instead of a random holdout')
ap.add_argument('--target', choices=['saliency', 'soft'], default='saliency')
ap.add_argument('--saliency-weight', type=float, default=1.0)
ap.add_argument('--soft-weight', type=float, default=0.0,
                help='>0 adds the window soft label as a second term')
ap.add_argument('--annot-weight', action='store_true',
                help='weight each timestep by its annotator count (low-count clips are noisy)')
ap.add_argument('--val-frac', type=float, default=0.1)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ch', type=int, default=128)
ap.add_argument('--dils', type=int, nargs='*', default=[1, 2, 4, 8, 16])
ap.add_argument('--steps', type=int, default=4000)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--device', default='cpu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def load(feat_root, label_root):
    """Return per-video (pooled [T,768], saliency [T], soft [T], n_annot [T])."""
    out = []
    for vf in sorted(Path(feat_root).glob('*.npz')):
        vid = vf.stem
        lf = Path(label_root) / f'{vid}.npz'
        if not lf.exists():
            continue
        F, L = np.load(vf), np.load(lf)
        pooled = F['pooled'].astype(np.float32)   # [T,768]
        # align to the shorter of the two grids
        T = min(len(pooled), len(L['saliency']))
        out.append({'vid': vid,
                    'x': pooled[:T],
                    'sal': L['saliency'][:T].astype(np.float32),
                    'soft': L['soft'][:T].astype(np.float32),
                    'nann': L['n_annot'][:T].astype(np.float32)})
    return out


def pad(rows):
    """Pad variable-length sequences to the split max; return arrays + mask."""
    T = max(len(r['x']) for r in rows)
    X = np.zeros((len(rows), T, 768), np.float32)
    S = np.zeros((len(rows), T), np.float32)
    Y = np.zeros((len(rows), T), np.float32)
    W = np.zeros((len(rows), T), np.float32)
    M = np.zeros((len(rows), T), np.float32)
    for i, r in enumerate(rows):
        n = len(r['x'])
        X[i, :n] = r['x']
        S[i, :n] = r['sal']
        Y[i, :n] = r['soft']
        W[i, :n] = r['nann']
        M[i, :n] = 1.0
    return X, S, Y, W, M


tr_all = load(args.feat_root, args.label_root)
if args.val_feat_root and args.val_label_root:
    tr = tr_all
    va = load(args.val_feat_root, args.val_label_root)
else:
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(tr_all))
    nv = max(1, int(len(tr_all) * args.val_frac))
    va = [tr_all[i] for i in idx[:nv]]
    tr = [tr_all[i] for i in idx[nv:]]

Xtr, Str, Ytr, Wtr, Mtr = pad(tr)
Xva, Sva, Yva, Wva, Mva = pad(va)
print(f'QV_TCN train={len(tr)} val={len(va)} '
      f'sal_mean={Str[Mtr>0].mean():.3f} pos(soft>0.5)={float((Ytr>0.5).mean()):.3f}', flush=True)


class TCN(torch.nn.Module):
    """Multi-scale dilated TCN.  dils beyond the PHD2 (1,2,4) give the ~5-120 s
    receptive field QVHighlights' long clips need - the single-scale head the
    PHD2 TCN used tops out at 15 frames and cannot see a 60 s highlight."""

    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4, 8, 16)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)
        self.dils = dils

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for conv, d in zip(self.convs, self.dils):
            h = torch.nn.functional.gelu(conv(h) + h)
        return self.out(h).squeeze(1)


model = TCN(ch=args.ch, dils=tuple(args.dils)).to(args.device).float()
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
Xt = torch.from_numpy(Xtr).to(args.device)
St = torch.from_numpy(Str).to(args.device)
Yt = torch.from_numpy(Ytr).to(args.device)
Wt = torch.from_numpy(Wtr).to(args.device)
Mt = torch.from_numpy(Mtr).to(args.device)
Xv = torch.from_numpy(Xva).to(args.device)
print(f'QV_TCN params={n_par} receptive_field={1 + 2 * sum(args.dils)} frames', flush=True)


def metrics(sc, y, m):
    """AP of the predicted score against the binary in-window label, masked."""
    sc = np.asarray(sc.detach().cpu() if torch.is_tensor(sc) else sc)
    y = np.asarray(y); m = np.asarray(m)
    sel = m > 0
    sc, y = sc[sel], (y[sel] > 0.5).astype(np.float32)
    if y.sum() == 0:
        return {'ap': float('nan'), 'base': float('nan')}
    order = np.argsort(-sc)
    ys = y[order]
    cum = np.cumsum(ys)
    prec = cum / np.arange(1, len(ys) + 1)
    ap = float((prec * ys).sum() / y.sum())
    return {'ap': ap, 'base': float(y.mean())}


best = (-1.0, None)
hist = []
for step in range(1, args.steps + 1):
    model.train()
    idx = torch.randint(0, len(Xt), (args.batch,))
    xb, sb, yb, wb, mb = Xt[idx], St[idx], Yt[idx], Wt[idx], Mt[idx]
    pred = model(xb)                                   # [B,T]
    w = mb.clone()
    if args.annot_weight:
        w = w * (wb / wb.clamp(min=1).max(1, keepdim=True).values)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    # masked MSE on saliency (continuous) and/or soft window label
    lsal = (w * (torch.sigmoid(pred) - sb) ** 2).sum(1).mean()
    lsoft = (w * (torch.sigmoid(pred) - yb) ** 2).sum(1).mean()
    loss = args.saliency_weight * lsal + args.soft_weight * lsoft
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 200 == 0 or step == args.steps:
        model.eval()
        with torch.no_grad():
            vs = model(Xv)
        m = metrics(torch.sigmoid(vs), Yva, Mva)
        hist.append({'step': step, 'loss': round(float(loss), 4), **m})
        print(f'step {step} loss={float(loss):.4f} val_ap={m["ap"]:.4f} '
              f'(base {m["base"]:.4f})', flush=True)
        if m['ap'] == m['ap'] and m['ap'] > best[0]:
            best = (m['ap'], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

args.output_dir.mkdir(parents=True, exist_ok=True)
if best[1] is not None:
    model.load_state_dict(best[1])
torch.save({'state_dict': model.state_dict(), 'ch': args.ch, 'dils': list(args.dils),
            'params': n_par, 'val_ap': best[0], 'target': args.target},
           args.output_dir / f'qvh_tcn_s{args.seed}.pt')
(args.output_dir / f'history_s{args.seed}.json').write_text(json.dumps(
    {'train': len(tr), 'val': len(va), 'params': n_par,
     'receptive_field_frames': 1 + 2 * sum(args.dils),
     'target': args.target, 'sal_w': args.saliency_weight, 'soft_w': args.soft_weight,
     'annot_weight': bool(args.annot_weight), 'best_val_ap': best[0],
     'history': hist,
     'label_semantics': ('QVHighlights annotator-mean saliency (dense, 0..1) '
                         'and/or exp-distance window soft label'),
     'weak_supervision': True}, indent=1))
print(f'SUMMARY params={n_par} best_val_ap={best[0]:.4f}', flush=True)
