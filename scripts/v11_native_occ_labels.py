"""Occupancy labels for the native x DP intersection (D1 direction).

The 4-level binary staircase locks keep-0.80 F1 at 0.597 regardless of
AP.  The DP objective consumes per-block gains; occupancy labels give it
exactly that: q_t = |[sub_t0, sub_t0+subL] n GT intervals| / subL (time
measure, no extra decoding).  DP semantics: costs d_t = frames per
sub-window (integer), gains w_t = q_t * d_t (binary-temporal reward:
each GT frame contributes IoU 1), G = sum of w_t.

Arms (paired seeds, lr 1e-4 for ALL arms to deconfound):
  dp_bin   exact DP, binary labels (the lr1e4 run from native_exactdp)
  dp_occ   exact DP, occupancy gains
  mse_bin  MSE, binary labels
  mse_occ  MSE, occupancy soft targets (gradient-equivalence check)
Evaluation is ALWAYS on the binary labels (AP + keep-0.80 sim F1).
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/phd2_vmae_native'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/native_occ.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--seeds', type=int, nargs='*', default=[20261009, 20261010, 20261011])
args = ap.parse_args()


def load(feat_root, index, sel, ev_src):
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, YBIN, QOCC, SRC = [], [], [], []
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
        subL = L / 4
        q4 = np.zeros(4, np.float32)
        for i in range(4):
            a_, b_ = i * subL, (i + 1) * subL
            inter = sum(max(0.0, min(b_, vb) - max(a_, va)) for va, vb in ivs)
            q4[i] = inter / subL
        if not (0 < q4.sum() < 4):
            continue
        fps = r.get('fps') or 25.0
        d_t = max(1, int(round(fps * subL)))
        w = (q4 * d_t).astype(np.float32)
        G = float(w.sum())
        X.append({'ord': m['ord4'].astype(np.float32).reshape(32, 768),
                  'rep': m['rep4'].astype(np.float32).reshape(32, 768)})
        YBIN.append(np.repeat((q4 > 0).astype(np.float32), 8))
        QOCC.append({'q32': np.repeat(q4, 8), 'w4': w, 'd_t': d_t, 'G': G})
        SRC.append(r['src'])
    return X, YBIN, QOCC, SRC


def expected_f_dp(logits, w4, d_t, G):
    """Exact E[F] with sub-window blocks: costs = d_t slots?  No - the
    TCN acts per 32-step slot, but the block reward semantics use the
    sub-window frame counts.  We act per sub-window: logits are averaged
    over the 8 slots of a sub-window to give p_t, then the recursion runs
    over 4 blocks with (d_t, w_t, G)."""
    p4 = torch.sigmoid(logits.view(4, 8).mean(1))
    prob = p4.new_ones(1)
    mass = p4.new_zeros(1)
    for i in range(4):
        old_p, old_m = prob, mass
        prob = ((1 - p4[i]) * torch.nn.functional.pad(old_p, (0, d_t))
                + p4[i] * torch.nn.functional.pad(old_p, (d_t, 0)))
        mass = ((1 - p4[i]) * torch.nn.functional.pad(old_m, (0, d_t))
                + p4[i] * torch.nn.functional.pad(old_m + w4[i] * old_p, (d_t, 0)))
    k = torch.arange(prob.numel(), dtype=p4.dtype, device=p4.device)
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


def run_arm(name, key, loss_kind, Xtr, Ytr, Qtr, Xev, Yev, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = TCN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        idx = rng.choice(len(Xtr), args.batch)
        opt.zero_grad()
        losses = []
        for i in idx:
            logits = model(torch.from_numpy(Xtr[i][key])[None])[0]
            if loss_kind == 'mse_bin':
                y = torch.from_numpy(Ytr[i])
                losses.append(((torch.sigmoid(logits) - y) ** 2).mean())
            elif loss_kind == 'mse_occ':
                q = torch.from_numpy(Qtr[i]['q32'])
                losses.append(((torch.sigmoid(logits) - q) ** 2).mean())
            elif loss_kind == 'dp_bin':
                y = torch.from_numpy(Ytr[i])
                d_t = Qtr[i]['d_t']
                G = float(Ytr[i].sum())
                losses.append(-expected_f_dp(logits, y[::8] * d_t, d_t, G))
            elif loss_kind == 'dp_occ':
                q = Qtr[i]
                losses.append(-expected_f_dp(logits, torch.from_numpy(q['w4']),
                                             q['d_t'], q['G']))
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
    X, YBIN, QOCC, SRC = load(args.feat_root, args.index, sel, ev_src)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    Xtr = [X[i] for i in tr]; Ytr = [YBIN[i] for i in tr]; Qtr = [QOCC[i] for i in tr]
    Xev = [X[i] for i in ev]; Yev = [YBIN[i] for i in ev]
    print(f'frags train {len(Xtr)} eval {len(Xev)}', flush=True)
    out = {'pool': {'train': len(Xtr), 'eval': len(Xev), 'lr': args.lr,
                    'note': 'occupancy-gain DP vs binary, all arms at lr 1e-4, eval on binary'},
           'arms': {}}
    plan = [('dp_bin', 'ord', 'dp_bin'), ('dp_occ', 'ord', 'dp_occ'),
            ('mse_bin', 'ord', 'mse_bin'), ('mse_occ', 'ord', 'mse_occ')]
    for name, key, loss_kind in plan:
        out['arms'][name] = {}
        for seed in args.seeds:
            r = run_arm(name, key, loss_kind, Xtr, Ytr, Qtr, Xev, Yev, seed)
            st = r.pop('state')
            torch.save({'state_dict': {k: torch.from_numpy(v) for k, v in st.items()},
                        'ch': 128, 'dils': [1, 2, 4], 'arm': name, 'seed': seed},
                       args.out.parent / f'native_{name}_s{seed}.pt')
            out['arms'][name][str(seed)] = r
            print(name, seed, r, flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
