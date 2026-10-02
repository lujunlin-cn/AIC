"""Temporal saliency head on QV fragments - the time-axis lever.

A fragment carries 8 pooled frame vectors (768-d) at absolute clip times and a
1 Hz saliency/soft label on a fragment-local grid.  The label is interpolated
to the 8 frame times so the head regresses highlight-worthiness per frame -
the exact quantity the deployment keep-mask ranks by.

This is the F1-denominator lever: the official F1 = 2*sum(matched IoU) /
(N_pred + N_gt) is crushed by predicting all ~550 frames when highlights are
~30-40% of a video.  A head that scores frames accurately lets us cut N_pred
toward N_gt; the model below is the scorer.

Eval is on the binary in-window label (soft > 0.5) so AP is comparable across
arms, but the regression target is the dense saliency mean by default - the
finer signal QV provides and PHD2 does not.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, required=True)   # frag_feats_train/<frag>.npz
ap.add_argument('--frag-label-root', type=Path, required=True)  # frag_train/frag_labels/<frag>.npz
ap.add_argument('--index', type=Path, required=True)       # frag_train/index.jsonl (has t0)
ap.add_argument('--target', choices=['saliency','soft'], default='saliency')
ap.add_argument('--val-frac', type=float, default=0.1)
ap.add_argument('--val-sources', type=Path, default=None)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ch', type=int, default=128)
ap.add_argument('--dils', type=int, nargs='*', default=[1, 2])
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--device', default='cpu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)

# t0 per fragment so frame absolute times map onto the fragment-local label grid
t0_of = {r['video_id']: float(r['t0']) for r in
         (json.loads(l) for l in args.index.read_text().splitlines() if l.strip())}

def load():
    out = []
    for vf in sorted(Path(args.feat_root).glob('*.npz')):
        fid = vf.stem
        lf = Path(args.frag_label_root) / f'{fid}.npz'
        if not lf.exists() or fid not in t0_of:
            continue
        F, L = np.load(vf), np.load(lf)
        pooled = F['pooled'].astype(np.float32)         # [8,768]
        tabs = F['t'].astype(np.float32)                # absolute src seconds
        tloc = tabs - t0_of[fid]                        # fragment-local seconds
        # interpolate the 1 Hz label onto the frame times
        lt, lsal, lsoft = L['t'], L['saliency'], L['soft']
        ys = np.interp(tloc, lt, lsal, left=lsal[0], right=lsal[-1]).astype(np.float32)
        yb = np.interp(tloc, lt, lsoft, left=lsoft[0], right=lsoft[-1]).astype(np.float32)
        out.append({'src': fid.rsplit('_q', 1)[0], 'x': pooled,
                    'ys': ys, 'yb': yb})
    return out

rows = load()
rng = np.random.default_rng(args.seed)
# split by SOURCE video so fragments of one clip stay together
srcs = sorted({r['src'] for r in rows})
if args.val_sources and Path(args.val_sources).exists():
    vset = set(json.loads(Path(args.val_sources).read_text()))
else:
    rng.shuffle(srcs)
    nv = max(1, int(len(srcs) * args.val_frac))
    vset = set(srcs[:nv])
    (args.output_dir.parent / 'val_sources.json').write_text(json.dumps(sorted(vset)))
tr = [r for r in rows if r['src'] not in vset]
va = [r for r in rows if r['src'] in vset]

Xtr = np.stack([r['x'] for r in tr]); Ytr = np.stack([r['ys'] if args.target=='saliency' else r['yb'] for r in tr])
Ytr_b = np.stack([r['yb'] for r in tr])
Xva = np.stack([r['x'] for r in va]);  Yva_b = np.stack([r['yb'] for r in va])
print(f'QV_FRAG_TCN train={len(tr)} val={len(va)} val_src={len(vset)} '
      f'pos(soft>0.5)={float((Yva_b>0.5).mean()):.3f} T={Xtr.shape[1]}', flush=True)


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2)):
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


model = TCN(ch=args.ch, dils=tuple(args.dils)).to(args.device).float()
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
Xt = torch.from_numpy(Xtr).to(args.device); Yt = torch.from_numpy(Ytr).to(args.device)
Xv = torch.from_numpy(Xva).to(args.device)
print(f'QV_FRAG_TCN params={n_par} receptive_field={1+2*sum(args.dils)} frames', flush=True)


def metrics(sc, y):
    sc = np.asarray(sc.detach().cpu() if torch.is_tensor(sc) else sc).reshape(-1)
    y = (np.asarray(y).reshape(-1) > 0.5).astype(np.float32)
    if y.sum() == 0:
        return {'ap': float('nan'), 'base': float('nan')}
    order = np.argsort(-sc); ys = y[order]; cum = np.cumsum(ys)
    ap = float((cum / np.arange(1, len(ys)+1) * ys).sum() / y.sum())
    return {'ap': ap, 'base': float(y.mean())}


best = (-1.0, None); hist = []
for step in range(1, args.steps + 1):
    model.train()
    idx = torch.randint(0, len(Xt), (args.batch,))
    pred = model(Xt[idx])
    loss = torch.nn.functional.mse_loss(torch.sigmoid(pred), Yt[idx])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 200 == 0 or step == args.steps:
        model.eval()
        with torch.no_grad():
            vs = model(Xv)
        m = metrics(torch.sigmoid(vs), Yva_b)
        hist.append({'step': step, 'loss': round(float(loss), 4), **m})
        print(f'step {step} loss={float(loss):.4f} val_ap={m["ap"]:.4f} (base {m["base"]:.4f})', flush=True)
        if m['ap'] == m['ap'] and m['ap'] > best[0]:
            best = (m['ap'], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

args.output_dir.mkdir(parents=True, exist_ok=True)
if best[1] is not None:
    model.load_state_dict(best[1])
torch.save({'state_dict': model.state_dict(), 'ch': args.ch, 'dils': list(args.dils),
            'params': n_par, 'val_ap': best[0], 'target': args.target},
           args.output_dir / f'qvh_frag_tcn_s{args.seed}.pt')
(args.output_dir / f'history_s{args.seed}.json').write_text(json.dumps(
    {'train': len(tr), 'val': len(va), 'val_sources': len(vset), 'params': n_par,
     'receptive_field_frames': 1+2*sum(args.dils), 'target': args.target,
     'best_val_ap': best[0], 'history': hist,
     'label_semantics': 'QV annotator-mean saliency interpolated to the 8-frame grid',
     'note': 'auxiliary temporal domain; fragment length matched to official p50 14s'},
    indent=1))
print(f'SUMMARY params={n_par} best_val_ap={best[0]:.4f}', flush=True)
