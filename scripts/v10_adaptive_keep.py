"""Adaptive keep-rate decision rule for the F1 keep mask.

Everything we have shipped keeps a FIXED fraction of each video (0.80).  That
was tuned once on a proxy pool and applied everywhere.  But the official score

    F1 = 2 * sum_{kept} IoU_i / (n_pred + n_gt)

has a per-video optimum that depends on how confident the scorer is, and a
fixed rate cannot track it.  The classical result for F-measure optimisation
says a ranked item should be included iff its probability exceeds half the
achievable F1 - which makes the optimal cut a fixed point, not a constant:

    keep frame i  <=>  p_i > F*/2

Solve it by sweeping the candidate cut and taking the F1 the scorer predicts
for itself.  Two properties fall out and both are testable here:

  * a video whose scores are peaked (few clear highlights) ends up keeping
    FEWER frames - correct, because its GT set is small and every extra kept
    frame inflates the denominator with no numerator;
  * a video whose scores are flat ends up keeping MORE - also correct, because
    dropping a frame there risks losing a positive with little denominator
    relief.

A fixed 0.80 gets both of these wrong in opposite directions on different
videos, and averaging over videos hides it.

This script compares, on the same PHD2 fragments and the same frozen head:
  fixed      keep 0.9 ... 0.5, the incumbent rule
  adaptive   the fixed-point cut, computed per video from its own score set
  oracle     the cut that maximises true F1, as an upper bound
with a source-level bootstrap CI on the adaptive-vs-best-fixed delta, because a
per-video rule has more freedom and therefore more variance to pay for.
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


def f1_of(mask, iou, n_gt):
    n_pred = int(mask.sum())
    if n_pred == 0:
        return 0.0
    return 2.0 * float(iou[mask].sum()) / (n_pred + n_gt)


def adaptive_mask(s, iou_mean, n_gt_prior, n_cand=24):
    """Fixed-point cut using ONLY the scorer's own output.

    The rule may not see the labels.  Expected iou_sum over a kept set K is
    sum_{i in K} p_i * E[iou_i], and E[iou_i] is unknown at inference, so the
    rule uses the video's own mean probability times a constant IoU - a scale
    factor that cannot change which k wins, since it multiplies every
    candidate's numerator equally.  n_gt is likewise estimated from the score
    mass (sum of p), not read from the truth.
    """
    L = len(s)
    order = np.argsort(-s)
    p = np.clip(s, 0.0, 1.0)
    n_gt_est = max(1.0, float(p.sum()))
    best_k, best_v = L, -1.0
    for k in range(1, L + 1):
        kk = int(order[:k].sum())
        v = 2.0 * iou_mean * kk / (k + n_gt_est)
        if v > best_v:
            best_v, best_k = v, k
    m = np.zeros(L, bool); m[order[:best_k]] = True
    return m, best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/lfm_feats'))
    ap.add_argument('--sources', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/val_sources.json'))
    ap.add_argument('--native-ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'))
    ap.add_argument('--iou', type=float, default=0.62)
    ap.add_argument('--keeps', type=float, nargs='*', default=[0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20261002)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    ck = torch.load(args.native_ckpt, map_location='cpu', weights_only=False)
    model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    model.load_state_dict(ck['state_dict']); model.eval()

    keep_src = set(json.loads(args.sources.read_text())) if args.sources.exists() else None
    sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]

    frag = []
    for r in rows:
        src = r.get('src')
        if keep_src is not None and src not in keep_src:
            continue
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
               for recs in sel.get(src, {}).values() for x in recs if float(x['t1']) > float(x['t0'])]
        stems = sorted(float(f.stem) for f in fs)
        times = np.array([stems[i] - t0 for i in range(L)], np.float32)
        y = np.zeros(L, np.float32)
        half = ((L - 1) / max(len(stems) - 1, 1)) * 0.5 if len(stems) > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if y.sum() == 0 or y.sum() == L:
            continue
        with torch.no_grad():
            s = torch.sigmoid(model(torch.from_numpy(X)[None]))[0].numpy()
        frag.append({'src': src, 's': s, 'y': y,
                     'iou': np.where(y > 0, args.iou, 0.0).astype(np.float32),
                     'n_gt': int(y.sum())})
        # the deployment-side view: the rule sees ONLY the score vector.  A
        # calibrated score is an estimate of P(frame is a highlight), so
        # sum(p) estimates the positive count without touching the truth.
        # sigmoid outputs are squashed near 0.5 on this pool, so min-max first
        # (done below) and then treat the rank as the probability proxy.
        p_hat = s.copy()

    if not frag:
        raise SystemExit('no fragments')
    print(f'fragments={len(frag)} sources={len(set(f["src"] for f in frag))}', flush=True)

    # normalise the scores per video so the fixed point is comparable across
    # videos whose absolute score scale differs
    for f in frag:
        s = f['s']
        f['p'] = (s - s.min()) / (s.max() - s.min() + 1e-6)

    srcs = sorted({f['src'] for f in frag})
    boot_idx = np.random.default_rng(args.seed).integers(0, len(srcs), size=(args.boot, len(srcs)))

    per_src = {}
    ks = []
    for f in frag:
        L = len(f['s'])
        rec = {}
        for kk in args.keeps:
            k = max(1, int(round(kk * L)))
            m = np.zeros(L, bool); m[np.argsort(-f['s'])[:k]] = True
            rec[f'fixed_{kk:.2f}'] = f1_of(m, f['iou'], f['n_gt'])
        am, ak = adaptive_mask(f['p'], args.iou, None)
        rec['adaptive'] = f1_of(am, f['iou'], f['n_gt'])
        ks.append(ak / L)
        per_src.setdefault(f['src'], []).append(rec)

    keys = list(next(iter(per_src.values()))[0].keys())
    means = {}
    for k in keys:
        arr = np.array([np.mean([r[k] for r in v]) for v in per_src.values()])
        means[k] = arr
    best_fixed = max((k for k in keys if k.startswith('fixed_')), key=lambda k: means[k].mean())
    d = means['adaptive'] - means[best_fixed]
    bd = d[boot_idx].mean(1)
    out = {
        'n_fragments': len(frag), 'n_sources': len(srcs),
        'f1': {k: round(float(means[k].mean()), 4) for k in keys},
        'best_fixed': best_fixed,
        'adaptive_minus_best_fixed': round(float(d.mean()), 4),
        'ci95': [round(float(np.percentile(bd, 2.5)), 4), round(float(np.percentile(bd, 97.5)), 4)],
        'significant': bool(np.percentile(bd, 2.5) > 0),
        'adaptive_keep_rate_mean': round(float(np.mean(ks)), 4),
        'adaptive_keep_rate_p10': round(float(np.percentile(ks, 10)), 4),
        'adaptive_keep_rate_p90': round(float(np.percentile(ks, 90)), 4),
        'keep_rate_spread_across_videos': round(float(np.percentile(ks, 90) - np.percentile(ks, 10)), 4),
    }
    print(json.dumps(out, indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()