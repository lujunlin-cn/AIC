"""8-sub-window refinement: 8-level binary staircase, 64-step TCN input.

Same recipe as the 4-sub-window experiment (dp_bin / mse_bin at lr 1e-4,
paired seeds), with the label grid refined from 4 to 8 levels and the
sub-windows halved.  Pool filter is the integer binary one; pools differ
from the 4-window experiment, so only within-experiment comparisons
count.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/phd2_vmae_native8'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/native8_control.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--seeds', type=int, nargs='*', default=[20261009, 20261010, 20261011])
args = ap.parse_args()


def load(feat_root, index, sel, ev_src):
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC = [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        t0, L = float(r['t0']), float(r['L'])
        ivs = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
                if b_ > a_:
                    ivs.append((a_, b_))
        if not ivs:
            continue
        y8 = np.zeros(8, np.float32)
        for i in range(8):
            c = (i + 0.5) * L / 8
            for a_, b_ in ivs:
                if a_ <= c < b_:
                    y8[i] = 1.0
                    break
        if not (0 < y8.sum() < 8):
            continue
        X.append({'ord': m['ord4'].astype(np.float32).reshape(64, 768)})
        Y.append(np.repeat(y8, 8))
        SRC.append(r['src'])
    return X, Y, SRC


def expected_f_dp(logits, y):
    G = float(y.sum())
    p = torch.sigmoid(logits)
    prob = logits.new_ones(1)
    mass = logits.new_zeros(1)
    for t in range(logits.shape[0]):
        old_p, old_m = prob, mass
        prob = ((1 - p[t]) * torch.nn.functional.pad(old_p, (0, 1))
                + p[t] * torch.nn.functional.pad(old_p, (1, 0)))
        mass = ((1 - p[t]) * torch.nn.functional.pad(old_m, (0, 1))
                + p[t] * torch.nn.functional.pad(old_m + y[t] * old_p, (1, 0)))
    k = torch.arange(prob.numel(), dtype=logits.dtype, device=logits.device)
    return (2.0 * mass / (k + G)).sum()


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
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


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def f1_keep(s, y, keep=0.8):
    k = max(1, int(round(keep * len(y))))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def run_arm(name, loss_kind, Xtr, Ytr, Xev, Yev, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = TCN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        idx = rng.choice(len(Xtr), args.batch)
        opt.zero_grad()
        losses = []
        for i in idx:
            logits = model(torch.from_numpy(Xtr[i]['ord'])[None])[0]
            y = torch.from_numpy(Ytr[i])
            if loss_kind == 'mse':
                losses.append(((torch.sigmoid(logits) - y) ** 2).mean())
            else:
                losses.append(-expected_f_dp(logits, y))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    aps, f1s = [], []
    with torch.no_grad():
        for x, y in zip(Xev, Yev):
            s = model(torch.from_numpy(x['ord'])[None])[0].numpy()
            aps.append(ap_of(s, y))
            f1s.append(f1_keep(s, y))
    return {'ap': round(float(np.mean(aps)), 4),
            'f1_keep080': round(float(np.mean(f1s)), 4),
            'seed': seed,
            'state': {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}


def main():
    sel = json.loads(args.selections.read_text())
    ev_src = set(json.loads(args.eval_sources.read_text()))
    X, Y, SRC = load(args.feat_root, args.index, sel, ev_src)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    Xtr = [X[i] for i in tr]; Ytr = [Y[i] for i in tr]
    Xev = [X[i] for i in ev]; Yev = [Y[i] for i in ev]
    print(f'frags train {len(Xtr)} eval {len(Xev)}', flush=True)
    out = {'pool': {'train': len(Xtr), 'eval': len(Xev), 'lr': args.lr,
                    'note': '8-sub-window binary staircase, 64-step input, lr 1e-4'},
           'arms': {}}
    for name, loss_kind in (('dp_bin', 'dp'), ('mse_bin', 'mse')):
        out['arms'][name] = {}
        for seed in args.seeds:
            r = run_arm(name, loss_kind, Xtr, Ytr, Xev, Yev, seed)
            st = r.pop('state')
            torch.save({'state_dict': {k: torch.from_numpy(v) for k, v in st.items()},
                        'ch': 128, 'dils': [1, 2, 4], 'arm': name, 'seed': seed},
                       args.out.parent / f'native8_{name}_s{seed}.pt')
            out['arms'][name][str(seed)] = r
            print(name, seed, r, flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
