"""Direct budget utility probe: the last decision-layer candidate.  (P2 route B)

Route A (calibrate probabilities, then expected-F) failed: Platt and isotonic
improved Brier and prevalence MAE but the resulting decision rule did not beat
a fixed keep of 0.80.  Route B skips probability entirely.  For each video it
summarises the head's score distribution into cheap statistics, then learns a
direct mapping

    h_kappa(score_summary) -> E[F_v(kappa)]

by cross-fitted ridge regression - one model per candidate keep rate, fit out
of fold so every video's prediction comes from a model that never saw it.  At
deployment the video keeps kappa* = argmax_kappa h_kappa(summary).

Why this can work where route A failed: the target is the F1 the video will
actually get, not a probability the video was never trained to emit.  The head
only ever saw mixed fragments, so its absolute outputs are not prevalence
estimates - but the SHAPE of its score distribution still carries the
information "this video has a tight peak" vs "this video is uniformly good",
and that shape, not a calibrated probability, is what choosing a keep rate
needs.

Verdict metric: real binary temporal F1 of the kappa* decision against the
fixed-0.80 baseline and Oracle A, with a source-level paired bootstrap CI.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def temporal_f1(mask, y):
    keep = int(mask.sum()); gt = int(y.sum())
    if gt == 0:
        return 1.0 if keep == 0 else 0.0
    return 2.0 * float(y[mask].sum()) / (keep + gt)


def summary(s):
    """Cheap score-distribution statistics; no labels involved."""
    n = len(s)
    q = np.percentile(s, (10, 25, 50, 75, 90))
    top10 = np.sort(s)[-max(1, n // 10):]
    p = np.clip(s - s.min(), 0, None)
    p = p / (p.sum() + 1e-9)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return np.array([n, s.mean(), s.std(), s.max(), s.min(), *q,
                     top10.mean(), top10.std(), s.max() / (np.abs(s).mean() + 1e-9),
                     float((s > 0.5).mean()), float((s > 0.6).mean()), float((s > 0.7).mean()),
                     ent, float(np.mean(np.abs(np.diff(s))))], np.float64)


def ridge_fit(X, y, lam=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    A = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return w, mu, sd


def ridge_apply(m, X):
    w, mu, sd = m
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/lfm_feats'))
    ap.add_argument('--native-ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'))
    ap.add_argument('--keeps', type=float, nargs='*', default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/budget_utility.json'))
    args = ap.parse_args()

    ck = torch.load(args.native_ckpt, map_location='cpu', weights_only=False)
    model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    model.load_state_dict(ck['state_dict']); model.eval()

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]

    per_src = {}
    for r in rows:
        d = args.feat_root / r['video_id']
        if not d.exists():
            continue
        fs = sorted(d.glob('*.npz'), key=lambda p: float(p.stem))
        if len(fs) < 4:
            continue
        X = np.stack([np.load(f)['pooled'].astype(np.float32) for f in fs])
        L = len(X)
        t0 = float(r['t0'])
        ivs = [(float(x['t0']) - t0, float(x['t1']) - t0)
               for recs in sel.get(r['src'], {}).values() for x in recs if float(x['t1']) > float(x['t0'])]
        stems = sorted(float(f.stem) for f in fs)
        times = np.array([stems[i] - t0 for i in range(L)], np.float32)
        y = np.zeros(L, np.float32)
        half = ((stems[-1] - stems[0]) / max(len(stems) - 1, 1)) * 0.5 if len(stems) > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if y.sum() == 0 or y.sum() == L:
            continue
        with torch.no_grad():
            s = torch.sigmoid(model(torch.from_numpy(X)[None]))[0].numpy()
        per_src.setdefault(r['src'], []).append((s, y))

    # one row per SOURCE (concatenated fragments): summary, per-kappa F1, oracle
    srcs = sorted(per_src)
    Z, targets, f_fixed, f_oracle, f_all = [], [], [], [], []
    for src in srcs:
        parts = per_src[src]
        s = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        n = len(s)
        order = np.argsort(-s)
        cum = np.cumsum(y[order])
        ks = np.arange(1, n + 1)
        tf = 2.0 * cum / (ks + float(y.sum()))
        Z.append(summary(s))
        targets.append([tf[min(k, n) - 1] for k in (np.round(np.array(args.keeps) * n)).astype(int)])
        f_all.append(tf[-1])
        k08 = max(1, int(round(0.8 * n)))
        f_fixed.append(tf[k08 - 1])
        f_oracle.append(tf.max())
    Z = np.stack(Z); targets = np.stack(targets)
    print(f'sources={len(srcs)} summary_dim={Z.shape[1]}', flush=True)

    # cross-fitted ridge: one model per kappa, predict out of fold
    rng = np.random.default_rng(args.seed)
    fold_of = rng.integers(0, args.folds, len(srcs))
    pred = np.zeros_like(targets)
    for k_idx in range(len(args.keeps)):
        for f in range(args.folds):
            tr = fold_of != f
            m = ridge_fit(Z[tr], targets[tr, k_idx])
            pred[~tr, k_idx] = ridge_apply(m, Z[~tr])

    kappa_star = pred.argmax(1)
    f_budget = np.array([targets[i, kappa_star[i]] for i in range(len(srcs))])
    # the chosen kappa's realised F1 uses the REAL per-kappa target (not the
    # predicted one) - the prediction only selects the keep rate
    best_fixed_idx = int(np.argmax(targets.mean(0)))
    f_bestfixed = targets[:, best_fixed_idx]

    d = f_budget - f_bestfixed
    boot = np.random.default_rng(args.seed).integers(0, len(srcs), (args.boot, len(srcs)))
    bd = d[boot].mean(1)
    out = {
        'n_sources': len(srcs),
        'keeps': args.keeps,
        'temporal_f1': {
            'keep_all': round(float(np.mean(f_all)), 4),
            'best_fixed_keep': round(float(np.mean(f_bestfixed)), 4),
            'best_fixed_kappa': args.keeps[best_fixed_idx],
            'budget_utility_crossfit': round(float(np.mean(f_budget)), 4),
            'oracle_A_best_k_per_video': round(float(np.mean(f_oracle)), 4),
        },
        'budget_minus_best_fixed': round(float(d.mean()), 4),
        'ci95': [round(float(np.percentile(bd, 2.5)), 4), round(float(np.percentile(bd, 97.5)), 4)],
        'significant': bool(np.percentile(bd, 2.5) > 0),
        'kappa_star_distribution': {str(k): int((kappa_star == i).sum())
                                    for i, k in enumerate(args.keeps)},
    }
    print(json.dumps(out, indent=1), flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()