"""Round-5 P0-CONTENT: champion content ablation (R5 section 1.3).

Question: does the VTREPLAY champion head read visual content, or is its
output a position-prior ordering?  Same question for the saturated DP
lineage and the healthy lr1e-4 MSE heads.

Arms (frozen checkpoints, NO training in this script):
  champion   LFM_V10/probe_deploy_head.pt  (the VTREPLAY deployment head)
  exactdp    expected_f_arm_exactdp_s202610{06,07,08}.pt  (package lineage)
  mse_lr1e4  expected_f_arm_mse_s2026100{6,7,8}_lr1e4.pt  (healthy lr)

Conditions (read-only over stored features):
  real       features as stored
  zeros      all-zero feature matrices (lengths preserved)
  xperm      cross-video permutation of feature matrices, 3 rng seeds,
             within equal-length groups
  slotprior  NO model: rank slots by TRAINING-pool per-slot positive rate
             (pure position prior, same prior vector for every eval
             fragment; confirm pool is read with the dev-trained prior)

Per-fragment records: f1_keep08, ap, gamma (s_(K) - s_(K+1)),
tie_ratio (unique scores / slots), topk_flip under +-1e-6 jitter.

Pools (BOTH already dev-exposed; this is a dev audit, NOT a confirm read):
  dev      LFM_V10/pool_feats + PHD2_FRAG_V1/index_clean.jsonl,
           eval sources only (same 1954-fragment eval pool as expected_f)
  confirm  PHD2_FRAG_CONFIRM_V1/pool_feats + index.jsonl (920; demoted to
           dev-audit set per round-5)

Preregistered read: if champion F1(real) - F1(zeros) source-cluster CI95
contains 0 on BOTH pools, the champion is content-insensitive -> per R5
1.4 no more loss/calibration tuning on it; the push moves to re-slicing
and content inputs.

CPU only.  BLAS threads capped.
Output: /data/aic/experiments_910a/LFM_V11/round5_content_ablation.json
        /data/aic/experiments_910a/LFM_V11/round5_content_ablation_per_video.csv
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
ap.add_argument('--v11-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11'))
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/round5_content_ablation.json'))
args = ap.parse_args()

SEEDS = [20261006, 20261007, 20261008]
PERM_SEEDS = [5101, 5102, 5103]


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
    assert len(s) == len(y), f'len mismatch s={len(s)} y={len(y)}'
    L = len(y)
    k = max(1, int(round(keep * L)))
    order = np.argsort(-s)
    top = order[:k]
    # gamma = s_(K) - s_(K+1): 0-based order[k-1] is rank K, order[k] rank K+1
    gamma = float(s[order[k - 1]] - s[order[k]]) if k < L else float('nan')
    tie = float(len(np.unique(np.round(s, 6))) / L)
    jit = s + np.random.RandomState(7).uniform(-1e-6, 1e-6, L)
    flip = int(len(set(top.tolist()) ^ set(np.argsort(-jit)[:k].tolist())) > 0)
    return f1_keep(s, y, keep), ap_of(s, y), gamma, tie, flip


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


def main():
    sel = json.loads(args.selections.read_text())
    eval_srcs = set(json.loads(args.eval_sources.read_text()))
    pools = {}
    X, Y, SRC, FID = load_fragments(args.dev_feat, args.dev_index, sel, keep_src=eval_srcs)
    pools['dev'] = (X, Y, SRC, FID)
    X, Y, SRC, FID = load_fragments(args.conf_feat, args.conf_index, sel)
    pools['confirm_audit'] = (X, Y, SRC, FID)
    for pn, (X, Y, SRC, FID) in pools.items():
        print(f'pool {pn}: {len(X)} fragments, {len(set(SRC))} sources, '
              f'L values {sorted(set(len(x) for x in X))}', flush=True)

    ckpts = {}
    p = args.v10_dir / 'probe_deploy_head.pt'
    ck = torch.load(p, map_location='cpu')
    m = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    m.load_state_dict(ck['state_dict']); m.eval()
    ckpts[('champion', 0)] = m
    for arm, base, suffix in (('exactdp', 'exactdp', ''),
                              ('mse_lr1e4', 'mse', '_lr1e4')):
        for sd in SEEDS:
            p = args.v11_dir / f'expected_f_arm_{base}_s{sd}{suffix}.pt'
            ck = torch.load(p, map_location='cpu')
            m = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
            m.load_state_dict(ck['state_dict']); m.eval()
            ckpts[(arm, sd)] = m
    torch.set_grad_enabled(False)

    rows_csv = []
    summary = {}
    for pn, (X, Y, SRC, FID) in pools.items():
        # per-slot positive rate from the DEV pool TRAIN sources only
        # (for the dev pool: sources NOT in eval_sources; for confirm: same
        # dev-trained prior - no confirm labels are ever used to fit it)
        Xt, Yt, SRCt, _ = load_fragments(args.dev_feat, args.dev_index, sel,
                                         keep_src=None)
        Xt = [x for x, s in zip(Xt, SRCt) if s not in eval_srcs]
        Yt = [y for y, s in zip(Yt, SRCt) if s not in eval_srcs]
        nslot = max(len(y) for y in Yt)
        pos = np.zeros(nslot); tot = np.zeros(nslot)
        for y in Yt:
            for i in range(len(y)):
                pos[i] += y[i]; tot[i] += 1
        prior = (pos / np.maximum(tot, 1)).astype(np.float32)
        print(f'pool {pn}: slot prior from {len(Yt)} train frags = '
              f'{np.round(prior, 3).tolist()}', flush=True)

        conds = {}
        for (arm, sd), model in ckpts.items():
            S_real = [model(torch.from_numpy(x)[None])[0].numpy() for x in X]
            S_zero = [model(torch.zeros_like(torch.from_numpy(x))[None])[0].numpy()
                      for x in X]
            S_perm = {}
            lens = [len(x) for x in X]
            for ps in PERM_SEEDS:
                rng = np.random.RandomState(ps)
                order = np.arange(len(X))
                for Lv in sorted(set(lens)):
                    grp = np.array([i for i in range(len(X)) if lens[i] == Lv])
                    order[grp] = grp[rng.permutation(len(grp))]
                Sp = [X[order[i]] for i in range(len(X))]
                S_perm[ps] = [model(torch.from_numpy(x)[None])[0].numpy()
                              for x in Sp]
            conds[(arm, sd)] = {'real': S_real, 'zeros': S_zero, 'xperm': S_perm}

        for (arm, sd), c in conds.items():
            for cond in ('real', 'zeros'):
                per = [frag_stats(s, Y[i], args.keep)
                       for i, s in enumerate(c[cond])]
                key = f'{pn}|{arm}|s{sd}|{cond}'
                summary[key] = {
                    'f1_mean': round(float(np.mean([p[0] for p in per])), 4),
                    'ap_mean': round(float(np.mean([p[1] for p in per])), 4),
                    'gamma_median': round(float(np.nanmedian([p[2] for p in per])), 6),
                    'tie_ratio_median': round(float(np.median([p[3] for p in per])), 4),
                    'topk_flip_rate': round(float(np.mean([p[4] for p in per])), 4)}
                for i, (f1v, apv, gm, tr, fl) in enumerate(per):
                    rows_csv.append([pn, arm, sd, cond, '', FID[i], SRC[i],
                                     round(f1v, 4), round(apv, 4), gm, tr, fl])
            # xperm: average F1 over the 3 permutation seeds per fragment
            per = []
            for i in range(len(X)):
                f1s, aps = [], []
                for ps in PERM_SEEDS:
                    f1v, apv, gm, tr, fl = frag_stats(c['xperm'][ps][i], Y[i],
                                                      args.keep)
                    f1s.append(f1v); aps.append(apv)
                per.append((float(np.mean(f1s)), float(np.mean(aps)), None, None, 0))
            summary[f'{pn}|{arm}|s{sd}|xperm'] = {
                'f1_mean': round(float(np.mean([p[0] for p in per])), 4),
                'ap_mean': round(float(np.mean([p[1] for p in per])), 4)}
            for i, (f1v, apv, gm, tr, fl) in enumerate(per):
                rows_csv.append([pn, arm, sd, 'xperm', 'mean3', FID[i], SRC[i],
                                 round(f1v, 4), round(apv, 4), gm, tr, fl])
            # slotprior (no model)
            per = [frag_stats(prior[:len(Y[i])], Y[i], args.keep)
                   for i in range(len(X))]
            summary[f'{pn}|{arm}|s{sd}|slotprior'] = {
                'f1_mean': round(float(np.mean([p[0] for p in per])), 4),
                'ap_mean': round(float(np.mean([p[1] for p in per])), 4)}
            for i, (f1v, apv, gm, tr, fl) in enumerate(per):
                rows_csv.append([pn, arm, sd, 'slotprior', '', FID[i], SRC[i],
                                 round(f1v, 4), round(apv, 4), gm, tr, fl])

        # paired deltas and source-cluster CIs per arm (seed-mean for the
        # 3-seed arms, single head for the champion)
        deltas = {}
        for arm in ('champion', 'exactdp', 'mse_lr1e4'):
            sd_list = [0] if arm == 'champion' else SEEDS
            for base in ('zeros', 'xperm', 'slotprior'):
                d = []
                for i in range(len(X)):
                    if base == 'slotprior':
                        b = f1_keep(prior[:len(Y[i])], Y[i], args.keep)
                        r = float(np.mean([frag_stats(ckpts[(arm, sd)](
                            torch.from_numpy(X[i])[None])[0].numpy(), Y[i],
                            args.keep)[0] for sd in sd_list]))
                    else:
                        if base == 'zeros':
                            b = float(np.mean([
                                f1_keep(conds[(arm, sd)]['zeros'][i], Y[i],
                                        args.keep) for sd in sd_list]))
                        else:
                            b = float(np.mean([
                                np.mean([f1_keep(conds[(arm, sd)]['xperm'][ps][i],
                                                 Y[i], args.keep)
                                         for ps in PERM_SEEDS])
                                for sd in sd_list]))
                        r = float(np.mean([
                            f1_keep(conds[(arm, sd)]['real'][i], Y[i],
                                    args.keep) for sd in sd_list]))
                    d.append(r - b)
                ci, mean = source_boot_ci(d, SRC, args.boot,
                                          np.random.RandomState(20261004))
                deltas[f'{arm}_real_minus_{base}'] = {'mean': mean, 'ci95': ci}
        summary[f'{pn}|paired_deltas'] = deltas
        print(f'pool {pn} deltas: {json.dumps(deltas, indent=1)}', flush=True)

    out_csv = args.out.with_suffix('').with_name(
        args.out.stem + '_per_video.csv')
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pool', 'arm', 'seed', 'condition', 'perm', 'fid', 'src',
                    'f1_keep08', 'ap', 'gamma', 'tie_ratio', 'topk_flip'])
        w.writerows(rows_csv)

    meta = {
        'protocol': 'P0-CONTENT dev audit, R5 section 1.3; frozen checkpoints, '
                    'no training; slot prior fitted on dev TRAIN sources only',
        'keep': args.keep,
        'ckpt_sha16': {'champion': sha16(args.v10_dir / 'probe_deploy_head.pt'),
                       **{f'exactdp_s{s}': sha16(args.v11_dir / f'expected_f_arm_exactdp_s{s}.pt')
                          for s in SEEDS},
                       **{f'mse_lr1e4_s{s}': sha16(args.v11_dir / f'expected_f_arm_mse_s{s}_lr1e4.pt')
                          for s in SEEDS}},
        'pool_sizes': {pn: {'frags': len(v[0]), 'sources': len(set(v[2]))}
                       for pn, v in pools.items()},
        'perm_seeds': PERM_SEEDS, 'boot': args.boot,
        'preregistered_read': 'champion real-minus-zeros CI95 containing 0 on '
                              'both pools => content-insensitive; stop tuning '
                              'champion loss/calibration (R5 1.4)',
    }
    args.out.write_text(json.dumps({'meta': meta, 'summary': summary},
                                   indent=1) + '\n')
    slim = {k: v for k, v in summary.items() if 'paired' in k or 'champion' in k}
    print(json.dumps(slim, indent=1))
    print('WROTE', args.out, out_csv, flush=True)


if __name__ == '__main__':
    main()
