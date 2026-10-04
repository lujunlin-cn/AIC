"""Four-arm comparison of training objectives on frozen PHD2 features.

Arms (same TCN, same init seed, same sampler stream, same budget):
  mse        binary-MSE ranking supervision (the incumbent)
  softf      differentiable soft-F on continuous probabilities
  exactdp    exact expected binary-temporal F via the DP recursion
             (vendor/v11_audit/expected_f.py, exhaustive-verified)
  reinforce  sampled actions + leave-one-out baseline (control)

PHD2 in-domain diagnosis ONLY: AP and simulated keep-F1 do not predict the
platform; the output ckpt goes through a mechanism-backed package or the
M01 line before any submission claim.  CPU.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, '/data/aic/experiments_910a/LFM_V11')

def load_fragments(feat_root, index, sel, keep_src):
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC, FID = [], [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        if 'mean' not in m or 't' not in m:
            continue
        Xf = m['mean'].astype(np.float32)
        L = len(Xf)
        t0 = float(r['t0'])
        ivs = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']), float(rec['t1'])
                if b_ > a_:
                    ivs.append((a_ - t0, b_ - t0))
        stamps = sorted(float(x) for x in m['t'])
        times = np.array([stamps[i] - t0 for i in range(L)], np.float32)
        y = np.zeros(L, np.float32)
        half = ((stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1)) * 0.5 if L > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if not (0 < y.sum() < L):
            continue  # mixed only, same as probe protocol
        X.append(Xf); Y.append(y); SRC.append(r['src']); FID.append(r['video_id'])
    return X, Y, SRC, FID


def ap_of(scores, y):
    o = np.argsort(-scores)
    ys = y[o]
    if ys.sum() == 0:
        return None
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def f1_at_keep(scores, y, keep):
    n = len(y)
    k = max(1, int(round(keep * n)))
    idx = set(np.argsort(-scores)[:k].tolist())
    hit = sum(y[i] for i in idx)
    return float(2 * hit / (k + y.sum()))


def expected_f_binary(logits, y):
    """Exact E[F] with per-frame blocks (d_t=1, gains=y_t, G=sum(y))."""
    G = float(y.sum())
    if G == 0:
        p = torch.sigmoid(logits)
        return (1 - p).prod()
    p = torch.sigmoid(logits)
    T = logits.shape[0]
    prob = logits.new_ones(1)
    mass = logits.new_zeros(1)
    for t in range(T):
        old_p, old_m = prob, mass
        prob = (1 - p[t]) * torch.nn.functional.pad(old_p, (0, 1)) + p[t] * torch.nn.functional.pad(old_p, (1, 0))
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


def train_arm(name, Xtr, Ytr, Xev, Yev, steps=900, batch=8, lr=1e-3, seed=20261006):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = TCN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xtr)
    hist = []
    for step in range(steps):
        idx = rng.choice(n, batch)
        opt.zero_grad()
        losses = []
        for i in idx:
            x = torch.from_numpy(Xtr[i])[None]
            logits = model(x)[0]
            y = torch.from_numpy(Ytr[i])
            if name == 'mse':
                loss = ((torch.sigmoid(logits) - y) ** 2).mean()
            elif name == 'softf':
                p = torch.sigmoid(logits)
                loss = -(2 * (p * y).sum() / (p.sum() + y.sum() + 1e-6))
            elif name == 'exactdp':
                loss = -expected_f_binary(logits, y)
            elif name == 'reinforce':
                p = torch.sigmoid(logits)
                a = torch.bernoulli(p)
                R = float(2 * (a.detach().numpy() * y.numpy()).sum() / (a.sum().item() + y.sum().item() + 1e-6))
                # leave-one-out baseline over a K=4 resample within the same fragment
                Rs = [R]
                for _ in range(3):
                    aj = torch.bernoulli(p)
                    Rj = float(2 * (aj.detach().numpy() * y.numpy()).sum() / (aj.sum().item() + y.sum().item() + 1e-6))
                    Rs.append(Rj)
                b = sum(Rs[1:]) / len(Rs[1:])
                adv = R - b
                logp = (a * torch.log(p.clamp_min(1e-6)) + (1 - a) * torch.log((1 - p).clamp_min(1e-6))).sum()
                loss = -adv * logp
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 300 == 0:
            hist.append(float(loss.item()))
    # eval
    model.eval()
    aps, f1s, keepall, ev_scores = [], [], [], []
    with torch.no_grad():
        for x, y in zip(Xev, Yev):
            s = model(torch.from_numpy(x)[None])[0].numpy()
            ev_scores.append(s.astype(np.float32).tolist())
            a_ = ap_of(s, y)
            if a_ is not None:
                aps.append(a_)
            f1s.append(f1_at_keep(s, y, 0.80))
            keepall.append(float(2 * y.sum() / (len(y) + y.sum())))
    return {'arm': name, 'ap': round(float(np.mean(aps)), 4),
            'f1_keep080': round(float(np.mean(f1s)), 4),
            'f1_keepall': round(float(np.mean(keepall)), 4),
            'ev_scores_f1keep080_list': [round(v, 6) for v in f1s],
            'loss_hist': hist,
            'state': {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}


def main():
    args = None
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
    p.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    p.add_argument('--selections', type=Path, default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
    p.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
    p.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/expected_f_4arm.json'))
    p.add_argument('--steps', type=int, default=900)
    a = p.parse_args()
    sel = json.loads(a.selections.read_text())
    ev_src = set(json.loads(a.eval_sources.read_text()))
    X, Y, SRC, FID = load_fragments(a.feat_root, a.index, sel, ev_src)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    Xtr = [X[i] for i in tr]; Ytr = [Y[i] for i in tr]
    Xev = [X[i] for i in ev]; Yev = [Y[i] for i in ev]
    print(f'train frags {len(Xtr)} eval frags {len(Xev)}', flush=True)
    out = {'pool': {'train_frags': len(Xtr), 'eval_frags': len(Xev),
                    'protocol': 'frozen pool_feats mean, mixed-only, source-disjoint eval'},
           'arms': {}}
    seeds = [20261006, 20261007, 20261008]
    for arm in ('mse', 'exactdp'):
        out['arms'][arm] = {}
        for sd in seeds:
            r = train_arm(arm, Xtr, Ytr, Xev, Yev, steps=a.steps, seed=sd)
            st = r.pop('state')
            torch.save({'state_dict': {k: torch.from_numpy(v) for k, v in st.items()},
                        'ch': 128, 'dils': [1, 2, 4], 'arm': arm, 'seed': sd},
                       a.out.parent / f'expected_f_arm_{arm}_s{sd}.pt')
            out['arms'][arm][str(sd)] = r
            print(arm, sd, json.dumps(r), flush=True)
    # paired per-fragment deltas need per-fragment scores; store summary stats
    a.out.write_text(json.dumps(out, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
