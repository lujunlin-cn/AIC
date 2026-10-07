"""R10 family-A pilot head: AST audio embeddings -> residual scores.

Pre-registered protocol (R10 report 1.x/3.4/8.2 + official-domain constraint
from the 174-video audit):
  readout R1  prob residual     score_s = a_s + l*avalid_s*g(x_s),  a_s = folded
              logit slot prior, g = TCN (champion family), exactdp/mse judging
  readout R2  utility residual  q_s = b_s + l*avalid_s*h(x_s),  u = 2Y/(K+G),
              MSE fit -> POLICY_ONLY_CANDIDATE semantics
  CONSTRUCTIVE fallback: avalid gates the residual; missing audio can never
              be 'learned' from the ~3% missing slots here (official 23%).
  5-fold source-grouped CV inside the pilot sources: prior, standardisation,
              residual model and the amplitude grid l are all fitted per fold;
              reported delta = fold-val pooled, paired per fragment.
  Controls  : (i) frozen position prior, (ii) frozen champion f1_list,
              (iii) within-video slot-PERMUTED audio (same recipe) - if real
              ~= permuted, any gain is distribution leakage, not alignment.
  Variance  : source-level influence sd -> MDE at M=64/984/768 (R10 8.3).
Output: r10_audio/pilot_head.json (incremental after every arm).
CPU only.  No official media.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r10_audio'))
ap.add_argument('--manifest-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit'))
ap.add_argument('--champion-gate', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/temporal_gate.json'))
ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r10_audio/pilot_head.json'))
ap.add_argument('--feature', default='ast_pool')
ap.add_argument('--k', type=int, default=6)
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--seed', type=int, default=20261007)
ap.add_argument('--lams', nargs='*', type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()


class TCN(torch.nn.Module):
    """Champion family (r8_temporal_gate.py): dils (1,2,4)."""

    def __init__(self, d_in, ch=128, dils=(1, 2, 4)):
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


def expected_f_binary(logits, y):
    G = float(y.sum())
    p = torch.sigmoid(logits)
    if G == 0:
        return (1 - p).prod()
    T = logits.shape[0]
    prob, mass = logits.new_ones(1), logits.new_zeros(1)
    for t in range(T):
        old_p, old_m = prob, mass
        prob = (1 - p[t]) * torch.nn.functional.pad(old_p, (0, 1)) + \
            p[t] * torch.nn.functional.pad(old_p, (1, 0))
        mass = ((1 - p[t]) * torch.nn.functional.pad(old_m, (0, 1))
                + p[t] * torch.nn.functional.pad(old_m + y[t] * old_p, (1, 0)))
    k = torch.arange(prob.numel(), dtype=logits.dtype)
    return (2.0 * mass / (k + G)).sum()


def f1_at_keep(scores, y, keep):
    scores, yv = np.asarray(scores, np.float64), np.asarray(y, np.float64)
    k = max(1, int(round(keep * len(yv))))
    idx = set(np.argsort(-scores)[:k].tolist())
    return float(2 * yv[list(idx)].sum() / (k + yv.sum()))


def topk_f1_scores(a_base, resid, avalid, lam):
    """Slot scores with CONSTRUCTIVE fallback: the residual is gated to zero
    wherever avalid=0, so those slots fall back to the position prior."""
    return a_base + lam * avalid * resid


def cluster_boot_paired(delta, srcs, boot, seed):
    us = sorted(set(srcs))
    idx = {s: [i for i, x in enumerate(srcs) if x == s] for s in us}
    rng = np.random.RandomState(seed)
    ms = []
    for _ in range(boot):
        pick = rng.choice(len(us), len(us), replace=True)
        ms.append(float(np.mean([delta[i] for q in pick for i in idx[us[q]]])))
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def influence_sigma(delta, srcs):
    """R10 8.3: psi_g source influence values -> sd (fragment-macro kept)."""
    d = np.asarray(delta, np.float64)
    us = sorted(set(srcs))
    cnt = np.array([sum(1 for x in srcs if x == s) for s in us], np.float64)
    summ = np.array([sum(d[i] for i, x in enumerate(srcs) if x == s) for s in us])
    psi = (summ - cnt * d.mean()) / cnt.mean()
    return float(psi.std(ddof=1))


def main():
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec = {'protocol': 'R10 family-A pilot; 5-fold source-grouped CV; '
                       'constructive avalid fallback; permuted-audio control',
           'feature': args.feature, 'status': 'RUNNING', 'arms': {}}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    man = []
    for split in ('train_public', 'eval_public'):
        for l in (args.manifest_dir / f'{split}.jsonl').read_text().splitlines():
            if l.strip():
                man.append(json.loads(l))
    have = {f.stem for f in args.feat_root.glob('p*/*.npz')}
    man = [r for r in man if r['fragment_id'] in have]
    SRC = [r['source_id'] for r in man]
    Y = [np.array(r['labels'], np.float32) for r in man]
    FID = [r['fragment_id'] for r in man]
    feats, avals = {}, {}
    for f in sorted(args.feat_root.glob('p*/*.npz')):
        m = np.load(f)
        feats[f.stem] = m[args.feature].astype(np.float32)
        avals[f.stem] = m['avalid'].astype(np.float32)
    X = [feats[r['fragment_id']] for r in man]
    AV = [avals[r['fragment_id']] for r in man]
    us = sorted(set(SRC))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(us))
    folds = [set(us[i] for i in perm[o::args.folds]) for o in range(args.folds)]
    print(f'pilot frags {len(man)} sources {len(us)} folds {args.folds}', flush=True)

    # champion reference (frozen model output) - ONLY eval-split fragments
    # have it; paired champion deltas are computed on that subset alone.
    gate = json.loads(args.champion_gate.read_text())
    ev_rows = [json.loads(l) for l in (args.manifest_dir / 'eval_public.jsonl')
               .read_text().splitlines() if l.strip()]
    if 'f1_list' in gate.get('champion', {}):
        champ_by_fid = {r['fragment_id']: v
                        for r, v in zip(ev_rows, gate['champion']['f1_list'])}
    else:
        champ_by_fid = {r['fragment_id']: f1_at_keep(
            [float(v) for v in r['scores']], [int(v) for v in r['labels']], 0.80)
            for r in ev_rows}
    CHAMP = np.array([champ_by_fid.get(f, np.nan) for f in FID])
    CMASK = ~np.isnan(CHAMP)
    CSRC = [SRC[i] for i in range(len(man)) if CMASK[i]]
    K = args.k

    def cv_arm(Xl, AVl, loss_kind, tag):
        """5-fold CV; returns pooled per-fragment F1 (fold-val only) and the
        chosen lambda per fold.  Priors and standardisation fold-internal."""
        nf1 = np.full(len(man), np.nan)
        lams_used = []
        for fo in range(args.folds):
            te_s = folds[fo]
            tr = [i for i in range(len(man)) if SRC[i] not in te_s]
            te = [i for i in range(len(man)) if SRC[i] in te_s]
            big = np.stack([Xl[i] for i in tr])
            mu, sd = big.mean(axis=(0, 1)), big.std(axis=(0, 1)) + 1e-6
            Xn = [(Xl[i] - mu) / sd for i in range(len(man))]
            Ytr = np.stack([Y[i] for i in tr])
            Gtr = Ytr.sum(1)
            p0 = np.clip(Ytr.mean(0), 1e-4, 1 - 1e-4)
            a_base = np.log(p0 / (1 - p0)).astype(np.float32)
            b_base = (2 * Ytr / (K + Gtr[:, None])).mean(0).astype(np.float32)
            base = a_base if loss_kind == 'prob' else b_base
            model = TCN(Xn[0].shape[1])
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            rng2 = np.random.RandomState(args.seed + fo)
            for step in range(args.steps):
                idx = rng2.choice(len(tr), args.batch)
                opt.zero_grad()
                losses = []
                for i in idx:
                    j = tr[i]
                    g = model(torch.from_numpy(Xn[j])[None])[0]
                    av_t = torch.from_numpy(AVl[j])
                    sc = torch.from_numpy(base) + av_t * g
                    y = torch.from_numpy(Y[j])
                    if loss_kind == 'prob':
                        losses.append(-expected_f_binary(sc, y))
                    else:
                        u = 2 * y / (K + y.sum())
                        losses.append(((sc - u) ** 2).mean())
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                G = [model(torch.from_numpy(Xn[j])[None])[0].numpy().astype(np.float32)
                     for j in te]
            best, best_l = -1, 0.0
            for lam in args.lams:
                f1s = [f1_at_keep(topk_f1_scores(base, G[t], AVl[te[t]], lam), Y[te[t]], 0.80)
                       for t in range(len(te))]
                m_ = float(np.mean(f1s))
                if m_ > best:
                    best, best_l = m_, lam
            lams_used.append(best_l)
            for t, j in enumerate(te):
                nf1[j] = f1_at_keep(topk_f1_scores(base, G[t], AVl[te[t]], best_l),
                                    Y[j], 0.80)
        return nf1, lams_used

    # ---- controls that need no training ----
    # fold-internal prior baseline (lam=0 path of R1): reuse cv_arm with a
    # zero-feature input to get the folded prior F1 (identical recipe)
    Z = [np.zeros((8, 1), np.float32)] * len(man)
    ZAV = [np.zeros(8, np.float32)] * len(man)     # gate closed -> pure prior
    nf1_prior, _ = cv_arm(Z, ZAV, 'prob', 'prior')
    rec['arms']['prior_folded'] = {'f1_mean': round(float(np.nanmean(nf1_prior)), 4)}
    nf1_champ = np.where(np.isnan(CHAMP), np.nanmean(CHAMP), CHAMP)
    rec['arms']['champion_frozen'] = {
        'f1_mean': round(float(np.nanmean(CHAMP)), 4),
        'f1_mean_filled': round(float(np.nanmean(nf1_champ)), 4),
        'n_covered': int(CMASK.sum())}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(f"prior_folded {rec['arms']['prior_folded']['f1_mean']} "
          f"champion {rec['arms']['champion_frozen']['f1_mean']}", flush=True)

    # permuted-audio control (within-video slot permutation, fixed seed)
    rngp = np.random.RandomState(args.seed)
    Xp, AVp = [], []
    for i in range(len(man)):
        pp = rngp.permutation(8)
        Xp.append(X[i][pp])
        AVp.append(AV[i][pp])
    nf1_perm, lams_perm = cv_arm(Xp, AVp, 'prob', 'permuted')
    rec['arms']['permuted_prob'] = {
        'f1_mean': round(float(np.nanmean(nf1_perm)), 4), 'lams': lams_perm}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(f"permuted_prob {rec['arms']['permuted_prob']['f1_mean']}", flush=True)

    # ---- real arms ----
    nf1_r1, lams_r1 = cv_arm(X, AV, 'prob', 'R1_prob_resid')
    d_c = (nf1_r1[CMASK] - CHAMP[CMASK]).tolist()
    d_p = (nf1_r1 - nf1_prior).tolist()
    d_pm = (nf1_r1 - nf1_perm).tolist()
    ci_c = cluster_boot_paired(d_c, CSRC, args.boot, args.seed)
    ci_p = cluster_boot_paired(d_p, SRC, args.boot, args.seed)
    rec['arms']['R1_prob_resid'] = {
        'f1_mean': round(float(np.nanmean(nf1_r1)), 4), 'lams': lams_r1,
        'delta_vs_champion': round(float(np.mean(d_c)), 5), 'ci_champion': ci_c,
        'n_champ_paired': int(CMASK.sum()),
        'delta_vs_prior': round(float(np.nanmean(d_p)), 5), 'ci_prior': ci_p,
        'delta_vs_permuted': round(float(np.nanmean(d_pm)), 5)}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(f"R1_prob_resid {rec['arms']['R1_prob_resid']}", flush=True)

    nf1_r2, lams_r2 = cv_arm(X, AV, 'util', 'R2_util_resid')
    d_c2 = (nf1_r2[CMASK] - CHAMP[CMASK]).tolist()
    d_p2 = (nf1_r2 - nf1_prior).tolist()
    ci_c2 = cluster_boot_paired(d_c2, CSRC, args.boot, args.seed)
    ci_p2 = cluster_boot_paired(d_p2, SRC, args.boot, args.seed)
    rec['arms']['R2_util_resid'] = {
        'f1_mean': round(float(np.nanmean(nf1_r2)), 4), 'lams': lams_r2,
        'delta_vs_champion': round(float(np.mean(d_c2)), 5), 'ci_champion': ci_c2,
        'delta_vs_prior': round(float(np.nanmean(d_p2)), 5), 'ci_prior': ci_p2,
        'semantics': 'POLICY_ONLY_CANDIDATE unless R1 also passes'}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(f"R2_util_resid {rec['arms']['R2_util_resid']}", flush=True)

    # ---- variance / MDE pricing (R10 8.3) ----
    from scipy.stats import norm as _n
    sig = influence_sigma((nf1_r1 - nf1_prior).tolist(), SRC)
    mdes = {}
    for M in (64, 984, 768):
        mdes[M] = round(float((_n.ppf(0.99) + _n.ppf(0.80)) * sig / np.sqrt(M)), 5)
    rec['pricing'] = {'sigma_influence': round(sig, 5),
                      'mde_one_sided_a01_power80': mdes,
                      'note': 'MDE at M sources; 64-source pilot cannot '
                              'reliably exclude 0.007 (R10 8.3)',
                      'official_coverage_discount': 0.77}
    rec['status'] = 'DONE'
    rec['elapsed_min'] = round((time.time() - t0) / 60, 1)
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
