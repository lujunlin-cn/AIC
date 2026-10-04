"""Round-4 clean confirmation: frozen pooled MSE vs exact-DP heads on the
500-source confirm set.

Protocol (GPT-6-PRO Q6.5 steps 4-8, preregistered before this script ran on
any confirm-set feature):
  * the comparison is the FROZEN dev checkpoints expected_f_arm_{mse,exactdp}_
    s2026100{6,7,8}.pt - no retraining, no tuning, one pass;
  * the metric is the deployment-simulated keep-0.80 binary-temporal F on
    8 x 1 s slots (the same simulated F1 the dev experiments report), plus
    mixed-subset AP as a diagnostic;
  * paired per-fragment deltas, aggregated (a) per seed and (b) by a
    SOURCE-level cluster bootstrap, because fragments of one source are not
    independent (the dev bootstrap treated them as iid and overstated
    confidence);
  * every input identity (checkpoint sha, confirm-set sha, feature pool
    size) is recorded in the output.

Read-only over the confirm features.  CPU only.
Output: /data/aic/experiments_910a/LFM_V11/round4_confirm_eval.json
"""
import argparse, hashlib, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1/pool_feats'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1/index.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--confirm-manifest', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/confirm_sources_500.json'))
ap.add_argument('--ckpt-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11'))
ap.add_argument('--ckpt-suffix', default='',
                help='checkpoint filename suffix, e.g. _lr1e4 (GROUP B)')
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--boot', type=int, default=10000)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round4_confirm_eval.json'))
args = ap.parse_args()

SEEDS = [20261006, 20261007, 20261008]
ARMS = ['mse', 'exactdp']


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


def load_fragments(feat_root, index, sel):
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
            continue
        X.append(Xf)
        Y.append(y)
        SRC.append(r['src'])
        FID.append(r['video_id'])
    return X, Y, SRC, FID


def f1_keep(s, y, keep):
    k = max(1, int(round(keep * len(y))))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def main():
    man = json.loads(args.confirm_manifest.read_text())
    sel = json.loads(args.selections.read_text())
    X, Y, SRC, FID = load_fragments(args.feat_root, args.index, sel)
    n_src = len(set(SRC))
    print(f'confirm pool: {len(X)} mixed fragments, {n_src} sources', flush=True)

    ckpts = {}
    for arm in ARMS:
        for sd in SEEDS:
            p = args.ckpt_dir / f'expected_f_arm_{arm}_s{sd}{args.ckpt_suffix}.pt'
            assert p.exists(), f'missing frozen checkpoint {p}'
            ck = torch.load(p, map_location='cpu')
            model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', [1, 2, 4])))
            model.load_state_dict(ck['state_dict'])
            model.eval()
            ckpts[(arm, sd)] = model
    torch.set_grad_enabled(False)

    with torch.no_grad():
        scores = {}
        for (arm, sd), model in ckpts.items():
            scores[(arm, sd)] = [model(torch.from_numpy(x)[None])[0].numpy()
                                 for x in X]

    per_frag = {}
    summary = {}
    for arm in ARMS:
        f1s = [float(np.mean([f1_keep(s, Y[i], args.keep)
                              for i, s in enumerate(scores[(arm, sd)])]))
               for sd in SEEDS]
        aps = [float(np.mean([ap_of(s, Y[i]) for i, s in enumerate(scores[(arm, sd)])]))
               for sd in SEEDS]
        summary[arm] = {'f1_per_seed': [round(v, 4) for v in f1s],
                        'f1_mean': round(float(np.mean(f1s)), 4),
                        'ap_per_seed': [round(v, 4) for v in aps],
                        'ap_mean': round(float(np.mean(aps)), 4)}
    for sd in SEEDS:
        per_frag[str(sd)] = {
            'fid': FID,
            'src': SRC,
            'mse_f1': [f1_keep(scores[('mse', sd)][i], Y[i], args.keep)
                       for i in range(len(X))],
            'dp_f1': [f1_keep(scores[('exactdp', sd)][i], Y[i], args.keep)
                      for i in range(len(X))],
        }

    deltas = {str(sd): [per_frag[str(sd)]['dp_f1'][i] - per_frag[str(sd)]['mse_f1'][i]
                        for i in range(len(X))] for sd in SEEDS}
    seed_means = [float(np.mean(deltas[str(sd)])) for sd in SEEDS]

    srcs = sorted(set(SRC))
    src_idx = {s: [i for i, x in enumerate(SRC) if x == s] for s in srcs}
    rs = np.random.RandomState(20261004)
    boots = {}
    for sd in SEEDS:
        d = deltas[str(sd)]
        means = []
        for _ in range(args.boot):
            pick = rs.choice(len(srcs), len(srcs), replace=True)
            vals = [d[i] for p_ in pick for i in src_idx[srcs[p_]]]
            means.append(float(np.mean(vals)))
        lo_, hi_ = np.percentile(means, [2.5, 97.5])
        boots[str(sd)] = {'mean': round(float(np.mean(d)), 4),
                          'ci95': [round(float(lo_), 4), round(float(hi_), 4)]}
    all_means = []
    for _ in range(args.boot):
        pick = rs.choice(len(srcs), len(srcs), replace=True)
        vals = [float(np.mean([deltas[str(sd)][i] for p_ in pick
                               for i in src_idx[srcs[p_]]]))
                for sd in SEEDS]
        all_means.append(float(np.mean(vals)))
    lo_, hi_ = np.percentile(all_means, [2.5, 97.5])

    out = {
        'meta': {
            'protocol': 'preregistered: frozen dev checkpoints on the frozen '
                        'confirm set, one pass, no tuning; source-cluster '
                        'bootstrap only',
            'confirm_sources_sha16': man['sources_sha256'][:16],
            'confirm_n_declared': man['n_sources'],
            'n_fragments_evaluated': len(X),
            'n_sources_evaluated': n_src,
            'pooling_contract': 'pool_feats mean [8,768] fp16 (v10_temporal_attnpool, '
                                'valid-token grid mean)',
            'keep': args.keep, 'group': 'B_lr1e4' if args.ckpt_suffix else 'A_package_lineage',
            'checkpoint_sha16': {f'{a}_s{s}': sha16(args.ckpt_dir / f'expected_f_arm_{a}_s{s}{args.ckpt_suffix}.pt')
                                 for a in ARMS for s in SEEDS},
        },
        'summary': summary,
        'seed_paired_delta': {str(sd): round(v, 4) for sd, v in zip(SEEDS, seed_means)},
        'delta_mean_across_seeds': round(float(np.mean(seed_means)), 4),
        'source_cluster_bootstrap': boots,
        'source_cluster_bootstrap_seedmean_ci95':
            [round(float(lo_), 4), round(float(hi_), 4)],
        'note': ('3-seed mean CI crosses zero in the dev read; this is the '
                 'clean-confirm replacement.  Decision rule (preregistered in '
                 'round-4 plan): promote only if the seed-mean delta lower '
                 'bound > 0 and mean >= 0.007.'),
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'meta'} |
                     {'meta_fragments': out['meta']['n_fragments_evaluated']}, indent=1))


if __name__ == '__main__':
    main()
