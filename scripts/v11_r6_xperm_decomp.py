"""R6 Q7-A: xperm intervention decomposition on the frozen champion.

R6 section Q7-A finding: the round-5 xperm (+0.003) alone cannot certify
"content is video-level only" - in-pool similar videos can shrink the
permutation effect, and zeros can inflate it.  This script runs the five
preregistered controls on the SAME frozen checkpoints and pools:

  rep_mu      x -> tile(mean_v(x))          keep video semantics, kill
                                           within-video temporal variation
              (deterministic, 1 draw)
  res_shuffle x -> mu_v + perm(x - mu_v)    keep video identity + residual
                                           distribution, kill event-time
              (20 permutation seeds)        correspondence
  mu_xres     x_i -> mu_i + (x_j - mu_j)    keep own semantics, replace
              j = within-length-group       local dynamics with another
              derangement, no self-map,     video's residuals
              same-source excluded
              (20 permutation seeds)
  derange     x_i -> x_j, within-length     keep known conditions, kill
              group derangement, no         video identity (round-5 xperm
              self-map, same-source         hardened with src exclusion)
              excluded  (20 seeds)
  zeros       x -> 0                        network bias; out-of-distribution
              (deterministic, 1 draw)

Preregistration (frozen before this run):
  * 20 permutation seeds: 5201..5220.  A seed fixes ALL permutation
    conditions that use randomness; deterministic conditions ignore it.
  * Strata: equal fragment length L.  Same-source mapping excluded
    (a source may own several fragments); self-mapping excluded.
    Ratio / sampling density are NOT in the strata (not in the stored
    index) - recorded as a known ID-level limitation in the manifest.
  * Readout per fragment: f1_keep08, ap, gamma(s_K - s_K+1), tie_ratio,
    topk flip under +-1e-6 jitter, mask_xor = |topk(real) ^ topk(cond)|/K.
  * Primary delta per condition: mean-per-fragment(real F1 - cond F1),
    source-cluster bootstrap CI95, 2000 resamples.  Pools: dev eval
    1954-fragment pool + old confirm 585 pool (both dev-exposed; DEV
    AUDIT ONLY, never a gate).
  * No rounding of anything into a claim about OFFICIAL mechanics.

CPU only.  Output: r6_xperm_intervention_audit.json + _per_video.csv
"""
import argparse, csv, hashlib, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--dev-feat', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
ap.add_argument('--dev-index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--conf-feat', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1/pool_feats'))
ap.add_argument('--conf-index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1/index.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--v10-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10'))
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--seeds', nargs='*', type=int,
                default=list(range(5201, 5221)))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_xperm_intervention_audit.json'))
args = ap.parse_args()

CONDS_DET = ('real', 'rep_mu', 'zeros')
CONDS_PERM = ('res_shuffle', 'mu_xres', 'derange')


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


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


def load_fragments(feat_root, index, sel, keep_src=None):
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC, FID = [], [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        if keep_src is not None and r['src'] not in keep_src:
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
            continue
        X.append(Xf); Y.append(y); SRC.append(r['src']); FID.append(r['video_id'])
    return X, Y, SRC, FID


def f1_keep(s, y, keep):
    k = max(1, int(round(keep * len(y))))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def frag_stats(s, y, keep):
    s = np.asarray(s, np.float32)
    y = np.asarray(y, np.float32)
    assert len(s) == len(y)
    L = len(y)
    k = max(1, int(round(keep * L)))
    order = np.argsort(-s)
    top = order[:k]
    gamma = float(s[order[k - 1]] - s[order[k]]) if k < L else float('nan')
    tie = float(len(np.unique(np.round(s, 6))) / L)
    jit = s + np.random.RandomState(7).uniform(-1e-6, 1e-6, L)
    flip = int(len(set(top.tolist()) ^ set(np.argsort(-jit)[:k].tolist())) > 0)
    return f1_keep(s, y, keep), ap_of(s, y), gamma, tie, flip


def mask_xor(s_real, s_cond, keep):
    k = max(1, int(round(keep * len(s_real))))
    a = set(np.argsort(-s_real)[:k].tolist())
    b = set(np.argsort(-s_cond)[:k].tolist())
    return len(a ^ b) / k


def source_boot_ci(delta, src, boot, rng):
    srcs = sorted(set(src))
    sidx = {s: [i for i, x in enumerate(src) if x == s] for s in srcs}
    means = []
    for _ in range(boot):
        pick = rng.choice(len(srcs), len(srcs), replace=True)
        vals = [delta[i] for p_ in pick for i in sidx[srcs[p_]]]
        means.append(float(np.mean(vals)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)], round(float(np.mean(delta)), 4)


def strat_derangement(n, srcs, rng, tries=12):
    """Permutation of range(n) within one length group: no self-map, no
    same-source map.  Best-of-tries if a perfect derangement is not found;
    residual violations are returned for the manifest."""
    idx = np.arange(n)
    best, best_bad = None, None
    for _ in range(tries):
        cand = idx[rng.permutation(n)]
        bad = int(np.sum(cand == idx) + np.sum(srcs[cand] == srcs[idx]))
        if best_bad is None or bad < best_bad:
            best, best_bad = cand, bad
        if bad == 0:
            break
    return best, best_bad


def build_conditions(X, SRC, seed):
    """Return dict cond -> list of feature matrices (len(X) entries)."""
    rng = np.random.RandomState(seed)
    out = {}
    out['real'] = X
    out['zeros'] = [np.zeros_like(x) for x in X]
    out['rep_mu'] = [np.tile(x.mean(0, keepdims=True), (len(x), 1)).astype(np.float32)
                     for x in X]
    # residual shuffle: within-fragment frame permutation of (x - mu) + mu
    rs = []
    for x in X:
        mu = x.mean(0, keepdims=True)
        p = rng.permutation(len(x))
        rs.append((mu + (x - mu)[p]).astype(np.float32))
    out['res_shuffle'] = rs
    # cross-video transfers within equal-length strata
    lens = [len(x) for x in X]
    src_arr = np.array(SRC)
    xres = [None] * len(X)
    der = [None] * len(X)
    n_self_bad = n_src_bad = 0
    for Lv in sorted(set(lens)):
        grp = np.array([i for i in range(len(X)) if lens[i] == Lv])
        if len(grp) < 2:
            for i in grp:
                xres[i] = X[i]
                der[i] = X[i]
            continue
        # mu_xres: own mean + other's residual
        m1, bad1 = strat_derangement(len(grp), src_arr[grp], rng)
        m2, bad2 = strat_derangement(len(grp), src_arr[grp], rng)
        n_self_bad += bad1 + bad2
        for pos, i in enumerate(grp):
            j1, j2 = grp[m1[pos]], grp[m2[pos]]
            mu_i = X[i].mean(0, keepdims=True)
            xres[i] = (mu_i + (X[j1] - X[j1].mean(0, keepdims=True))).astype(np.float32)
            der[i] = X[j2]
    out['mu_xres'] = xres
    out['derange'] = der
    return out, {'residual_same_src_or_self_maps': int(n_self_bad)}


def main():
    sel = json.loads(args.selections.read_text())
    eval_srcs = set(json.loads(args.eval_sources.read_text()))
    pools = {}
    X, Y, SRC, FID = load_fragments(args.dev_feat, args.dev_index, sel, keep_src=eval_srcs)
    pools['dev'] = (X, Y, SRC, FID)
    X, Y, SRC, FID = load_fragments(args.conf_feat, args.conf_index, sel)
    pools['confirm_audit'] = (X, Y, SRC, FID)
    for pn, (X, Y, SRC, FID) in pools.items():
        print(f'pool {pn}: {len(X)} fragments, {len(set(SRC))} sources', flush=True)

    ck = torch.load(args.v10_dir / 'probe_deploy_head.pt', map_location='cpu')
    model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    model.load_state_dict(ck['state_dict']); model.eval()
    torch.set_grad_enabled(False)

    rows_csv = []
    summary = {}
    for pn, (X, Y, SRC, FID) in pools.items():
        S_real = [model(torch.from_numpy(x)[None])[0].numpy() for x in X]
        f1_real = np.array([frag_stats(s, y, args.keep)[0]
                            for s, y in zip(S_real, Y)], np.float64)
        # per-condition score lists: det conds once, perm conds per seed
        # build all conditions per seed once (cheap), score each
        per_seed_scores = {c: [] for c in CONDS_PERM}
        det_scores = {}
        man = None
        for sd in args.seeds:
            conds, man_sd = build_conditions(X, SRC, sd)
            man = man_sd
            for cond in CONDS_DET + CONDS_PERM:
                S = [model(torch.from_numpy(x)[None])[0].numpy()
                     for x in conds[cond]]
                if cond in CONDS_PERM:
                    per_seed_scores[cond].append(S)
                elif cond != 'real' and sd == args.seeds[0]:
                    det_scores[cond] = S
        summary[pn] = {'n_frag': len(X), 'n_src': len(set(SRC)),
                       'manifest': man, 'conds': {}}
        for cond in ('real',) + CONDS_DET + CONDS_PERM:
            if cond == 'real':
                stats = [frag_stats(s, y, args.keep) for s, y in zip(S_real, Y)]
                f1 = f1_real
                xor = [0.0] * len(X)
                n_draws = 1
            elif cond in CONDS_DET:
                S = det_scores[cond]
                stats = [frag_stats(s, y, args.keep) for s, y in zip(S, Y)]
                f1 = np.array([r[0] for r in stats], np.float64)
                xor = [mask_xor(sr, sc, args.keep)
                       for sr, sc in zip(S_real, S)]
                n_draws = 1
            else:
                # mean per-fragment F1 over seeds first (round-5 convention)
                f1s = []
                xors = []
                stats_last = None
                for S in per_seed_scores[cond]:
                    f1s.append([frag_stats(s, y, args.keep)[0] for s, y in zip(S, Y)])
                    xors.append([mask_xor(sr, sc, args.keep)
                                 for sr, sc in zip(S_real, S)])
                    stats_last = [frag_stats(s, y, args.keep) for s, y in zip(S, Y)]
                f1 = np.mean(np.array(f1s, np.float64), axis=0)
                xor = np.mean(np.array(xors, np.float64), axis=0)
                stats = stats_last
                n_draws = len(args.seeds)
            (ci, mean_d) = source_boot_ci(list(f1_real - f1), SRC, args.boot,
                                          np.random.RandomState(99))
            g = np.array([r[2] for r in stats], np.float64)
            summary[pn]['conds'][cond] = {
                'f1_mean': round(float(f1.mean()), 4),
                'real_minus_cond_f1': mean_d,
                'real_minus_cond_ci95': ci,
                'ap_mean': round(float(np.mean([r[1] for r in stats])), 4),
                'gamma_median': round(float(np.nanmedian(g)), 5),
                'flip_rate': round(float(np.mean([r[4] for r in stats])), 4),
                'mask_xor_mean': round(float(np.mean(xor)), 4),
                'n_draws': n_draws}
            print(f'{pn}/{cond}: dF1 {mean_d} CI {ci} xor {summary[pn]["conds"][cond]["mask_xor_mean"]}',
                  flush=True)
            for i in range(len(X)):
                rows_csv.append([pn, cond, FID[i], SRC[i], round(float(f1_real[i]), 4),
                                 round(float(f1[i]), 4), round(float(f1_real[i] - f1[i]), 4),
                                 round(float(xor[i]), 4)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        'protocol': 'R6 Q7-A xperm intervention decomposition; frozen champion; '
                    'DEV_AUDIT_ONLY; 20 preregistered seeds 5201-5220; strata=L; '
                    'same-src+self excluded (best-of-12); ratio/density not in strata',
        'ckpt_sha16': {'champion_v10_probe_deploy_head': sha16(args.v10_dir / 'probe_deploy_head.pt')},
        'perm_seeds': args.seeds, 'keep': args.keep, 'boot': args.boot,
        'summary': summary}, indent=1) + '\n')
    csvp = args.out.with_name(args.out.stem + '_per_video.csv')
    with open(csvp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pool', 'cond', 'video_id', 'src', 'f1_real', 'f1_cond',
                    'delta', 'mask_xor'])
        w.writerows(rows_csv)
    print('WROTE', args.out, 'and', csvp, flush=True)


if __name__ == '__main__':
    main()
