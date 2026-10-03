"""Calibration probe: is the decision gap harvestable without GT?  (P2 route A)

Oracle A said the current ranking supports binary temporal F1 0.78 with
per-video oracle keep, vs 0.66 at the fixed 0.80 we deploy.  Harvesting that
needs an unbiased estimate of each video's positive count - i.e. calibration.

Three things this probe does differently from every previous temporal experiment:

1. NO fragment filtering.  The pool builder dropped all-positive and
   all-negative fragments because they carry no ranking signal.  But those are
   exactly the videos that teach a head what a typical prevalence looks like;
   training and calibrating on the mixed-only subset induces
   P(y=1 | x, mixed) instead of P(y=1 | x).  Here every fragment with at least
   one scored frame enters, and the selection filter is recorded as a covariate
   so its effect is measurable.

2. Source-disjoint calibration.  Platt and isotonic are fit on the complement
   of the evaluation sources; nothing about the eval videos touches the
   calibrator.

3. The deployable decision rule is scored against real temporal F1.  The rule
   is GPT-6-PRO's eq. 7.2: rank by q_i = p_i (probability head), estimate the
   video's positive count as G_hat = sum_i p_i, then pick k* = argmax_k
   2*sum_{j<=k} q_(j) / (k + G_hat).  No oracle quantity enters.

Reported on the held-out sources: prevalence MAE before/after calibration, ECE,
Brier, and the temporal F1 of the calibrated decision rule against fixed-keep
baselines and Oracle A as the ceiling.
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


def platt_fit(logits, y):
    """Logistic regression on the logit, 1-D Newton with a couple of safeguards."""
    z = logits.astype(np.float64)
    X = np.stack([np.ones_like(z), z], 1)
    w = np.zeros(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + 1e-4 * w          # small L2 for separable folds
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-4 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def platt_apply(w, logits):
    return 1.0 / (1.0 + np.exp(-(w[0] + w[1] * logits)))


def isotonic_fit(x, y):
    """Pool-adjacent-violators on sorted x; returns step function via interp."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(np.float64)
    # PAV
    blocks = [[v] for v in ys]
    weights = [[1.0] for _ in ys]
    i = 0
    vals, wts = [], []
    stack = []
    for v in ys:
        stack.append([v, 1.0])
        while len(stack) > 1 and stack[-1][0] < stack[-2][0]:
            v2, n2 = stack.pop(); v1, n1 = stack.pop()
            stack.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    vals = [s[0] for s in stack]
    # block boundaries in x-space
    # assign each sorted sample its block mean
    sizes = [int(round(s[1])) for s in stack]
    yhat = np.empty_like(ys)
    idx = 0
    for v, n in zip([s[0] for s in stack], sizes):
        yhat[idx:idx + n] = v
        idx += n
    return xs, yhat


def isotonic_apply(fit, x):
    xs, yhat = fit
    return np.interp(x, xs, yhat, left=yhat[0], right=yhat[-1])


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/lfm_feats'))
    ap.add_argument('--native-ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'))
    ap.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/val_sources.json'))
    ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/calibration_probe.json'))
    args = ap.parse_args()

    ck = torch.load(args.native_ckpt, map_location='cpu', weights_only=False)
    model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    model.load_state_dict(ck['state_dict']); model.eval()

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
    eval_src = set(json.loads(args.eval_sources.read_text())) if args.eval_sources.exists() else None
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]

    # ---- score EVERY fragment (no mixed-only filter), record the filter flag
    calib, ev = [], []
    n_allpos = n_allneg = n_mixed = 0
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
        kind = 'mixed' if 0 < y.sum() < L else ('allpos' if y.sum() == L else 'allneg')
        n_allpos += kind == 'allpos'; n_allneg += kind == 'allneg'; n_mixed += kind == 'mixed'
        with torch.no_grad():
            logits = model(torch.from_numpy(X)[None])[0].numpy()
        rec = {'src': r['src'], 'logit': logits.astype(np.float64), 'y': y, 'kind': kind}
        (ev if (eval_src and r['src'] in eval_src) else calib).append(rec)

    print(f'fragments: mixed={n_mixed} allpos={n_allpos} allneg={n_allneg} | '
          f'calib_sources={len(set(r["src"] for r in calib))} eval_sources={len(set(r["src"] for r in ev))}', flush=True)

    # ---- fit calibrators on the calib fold (raw logits, all fragment types)
    cl = np.concatenate([r['logit'] for r in calib]); cy = np.concatenate([r['y'] for r in calib])
    w = platt_fit(cl, cy)
    iso = isotonic_fit(cl, cy)

    # ---- evaluate on the eval fold: prevalence error + decision rules
    res = {k: {'temporal_f1': [], 'prev_err': [], 'ece': [], 'brier': []}
           for k in ('raw', 'platt', 'isotonic', 'fixed08', 'oracleA')}
    prev_detail = []
    for r in ev:
        y = r['y']; n = len(y)
        gt = float(y.sum())
        preds = {
            'raw': 1.0 / (1.0 + np.exp(-r['logit'])),
            'platt': platt_apply(w, r['logit']),
            'isotonic': isotonic_apply(iso, r['logit']),
        }
        order = np.argsort(-preds['raw'])          # ranking is calibration-free
        for name, p in preds.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            g_hat = float(p.sum())
            # decision rule: k* = argmax_k 2*sum q_(j) / (k + G_hat)
            cum = np.cumsum(p[order])
            ks = np.arange(1, n + 1)
            vf = 2.0 * cum / (ks + max(g_hat, 1e-6))
            k_star = int(ks[np.argmax(vf)])
            m = np.zeros(n, bool); m[order[:k_star]] = True
            res[name]['temporal_f1'].append(temporal_f1(m, y))
            res[name]['prev_err'].append(abs(g_hat / n - gt / n))
            res[name]['ece'].append(ece(p, y))
            res[name]['brier'].append(float(((p - y) ** 2).mean()))
        # baselines
        k08 = max(1, int(round(0.8 * n)))
        m08 = np.zeros(n, bool); m08[order[:k08]] = True
        res['fixed08']['temporal_f1'].append(temporal_f1(m08, y))
        best = -1.0
        for k in range(0, n + 1):
            m = np.zeros(n, bool); m[order[:k]] = True
            f = temporal_f1(m, y)
            if f > best:
                best = f
        res['oracleA']['temporal_f1'].append(best)
        prev_detail.append({'src': r['src'], 'gt_rate': round(gt / n, 3),
                            'raw_rate': round(float(preds['raw'].mean()), 3),
                            'platt_rate': round(float(preds['platt'].mean()), 3),
                            'kind': r['kind']})

    out = {'n_eval_sources': len(ev),
           'fragment_mix_on_eval': {
               'mixed': sum(1 for r in ev if r['kind'] == 'mixed'),
               'allpos': sum(1 for r in ev if r['kind'] == 'allpos'),
               'allneg': sum(1 for r in ev if r['kind'] == 'allneg')},
           'variants': {}}
    for name, d in res.items():
        if not d['temporal_f1']:
            continue
        out['variants'][name] = {
            'temporal_f1': round(float(np.mean(d['temporal_f1'])), 4),
            'prevalence_MAE': round(float(np.mean(d['prev_err'])), 4) if d['prev_err'] else None,
            'ECE': round(float(np.mean(d['ece'])), 4) if d['ece'] else None,
            'Brier': round(float(np.mean(d['brier'])), 4) if d['brier'] else None,
        }
    print(json.dumps(out, indent=1), flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    with args.out.with_suffix('.per_video.csv').open('w') as fh:
        fh.write('src,gt_rate,raw_rate,platt_rate,kind\n')
        for d in prev_detail:
            fh.write(f"{d['src']},{d['gt_rate']},{d['raw_rate']},{d['platt_rate']},{d['kind']}\n")


if __name__ == '__main__':
    main()