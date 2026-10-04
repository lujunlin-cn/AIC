"""Round-4 pooled-line diagnostics on the dev held-out pool (CPU).

Three preregistered audits from the GPT-6-PRO review, all run on the SAME
frozen checkpoints and the SAME dev held-out fragments the 3-seed read used:

 1. seed-identical or learned? (Q6.4 counter-hypothesis: the DP arm's near-
    zero seed variance is a saturation artifact - same logits, same tie-break)
    -> pairwise top-0.80 mask Jaccard across the 3 DP seeds and the 3 MSE
       seeds, |logit| saturation rates, and logit spread per fragment.
 2. does the head read pixels at all? (Q6.4: stability must not survive
    content destruction)
    -> each DP seed re-scored on (a) real features, (b) all-zero features,
       (c) per-fragment time-shuffled features, (d) normalized time index
       only.  If (b)-(d) match (a), the "same solution" is a prior, not
       learning.
 3. mask exchange (Q3.3) and oracles (Q3.5): where DP beats MSE inside the
    kept set, how many slots actually enter/leave the top-0.80 set and with
    what label mass; the fixed-budget action oracle and the fixed-ranking
    prefix oracle bound what any re-ranking / re-cutting could still give.

Output: /data/aic/experiments_910a/LFM_V11/round4_pooled_diagnostics.json
"""
import argparse, hashlib, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--ckpt-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11'))
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--ckpt-suffix', default='',
                help='checkpoint filename suffix, e.g. _lr1e4')
ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round4_pooled_diagnostics.json'))
args = ap.parse_args()

SEEDS = [20261006, 20261007, 20261008]


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
            continue
        if keep_src is not None and r['src'] not in keep_src:
            continue
        X.append(Xf)
        Y.append(y)
        SRC.append(r['src'])
        FID.append(r['video_id'])
    return X, Y, SRC, FID


def top_mask(s, y, keep):
    k = max(1, int(round(keep * len(y))))
    m = np.zeros(len(y), bool)
    m[np.argsort(-s)[:k]] = True
    return m


def jaccard(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def f1_of(m, y):
    return float(2 * y[m].sum() / (m.sum() + y.sum()))


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    ev_src = set(json.loads(args.eval_sources.read_text()))
    sel = json.loads(args.selections.read_text())
    X, Y, SRC, FID = load_fragments(args.feat_root, args.index, sel, ev_src)
    print(f'dev held-out: {len(X)} mixed fragments', flush=True)

    models = {}
    for arm in ('mse', 'exactdp'):
        for sd in args.seeds:
            p = args.ckpt_dir / f'expected_f_arm_{arm}_s{sd}{args.ckpt_suffix}.pt'
            ck = torch.load(p, map_location='cpu')
            m = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', [1, 2, 4])))
            m.load_state_dict(ck['state_dict'])
            m.eval()
            models[(arm, sd)] = m
    torch.set_grad_enabled(False)

    def score(model, x):
        return model(torch.from_numpy(x)[None])[0].numpy()

    res = {'meta': {'n_fragments': len(X), 'keep': args.keep,
                    'ckpt_sha16': {f'{a}_s{s}': sha16(args.ckpt_dir / f'expected_f_arm_{a}_s{s}.pt')
                                   for a in ('mse', 'exactdp') for s in args.seeds}}}

    # ---- 1. seed-identical or learned? -----------------------------------
    seeds_u = args.seeds
    masks = {(a, s): [top_mask(score(models[(a, s)], x), y, args.keep)
                      for x, y in zip(X, Y)]
             for a in ('mse', 'exactdp') for s in args.seeds}
    pair = {}
    for arm in ('mse', 'exactdp'):
        js = [jaccard(masks[(arm, a)][i], masks[(arm, b)][i])
              for i in range(len(X)) for a, b in [(seeds_u[0], seeds_u[1]), (seeds_u[0], seeds_u[2]), (seeds_u[1], seeds_u[2])]]
        pair[arm] = {'mask_jaccard_mean_across_seedpairs': round(float(np.mean(js)), 4),
                     'fraction_identical_masks': round(float(np.mean([j == 1.0 for j in js])), 4)}
    sat = {}
    for arm in ('mse', 'exactdp'):
        zs = np.concatenate([np.abs(np.concatenate([score(models[(arm, s)], x)
                                                    for x, y in zip(X, Y)]))
                             for s in SEEDS])
        sat[arm] = {'logit_abs_mean': round(float(zs.mean()), 3),
                    'frac_abs_gt_8': round(float((zs > 8).mean()), 4)}
    res['seed_consistency'] = {'pairwise': pair, 'logits': sat,
                               'reading': 'DP jaccard ~1 with high saturation '
                                          'supports the same-tie-break hypothesis; '
                                          'jaccard <1 means genuinely similar-but-'
                                          'distinct policies'}

    # ---- 2. content destruction ------------------------------------------
    rng = np.random.RandomState(7)
    destr = {}
    for arm in ('exactdp', 'mse'):
        variants = {'real': [], 'zeros': [], 'shuffled': [], 'time_only': []}
        for s in SEEDS:
            f1s = {k: [] for k in variants}
            for i, (x, y) in enumerate(zip(X, Y)):
                xt = np.zeros_like(x)
                xt[:, :] = (np.arange(len(x), dtype=np.float32) / max(len(x) - 1, 1))[:, None]
                f1s['real'].append(f1_of(top_mask(score(models[(arm, s)], x), y, args.keep), y))
                f1s['zeros'].append(f1_of(top_mask(score(models[(arm, s)], np.zeros_like(x)), y, args.keep), y))
                xp = x[rng.permutation(len(x))]
                f1s['shuffled'].append(f1_of(top_mask(score(models[(arm, s)], xp), y, args.keep), y))
                f1s['time_only'].append(f1_of(top_mask(score(models[(arm, s)], xt), y, args.keep), y))
            for k in variants:
                variants[k].append(float(np.mean(f1s[k])))
        destr[arm] = {k: round(float(np.mean(v)), 4) for k, v in variants.items()}
    res['content_destruction'] = destr

    # ---- 3. mask exchange + oracles (seed 20261006) ----------------------
    s0 = args.seeds[0]
    ex = {'slots_in': [], 'slots_out': [], 'label_in': [], 'label_out': []}
    for i, y in enumerate(Y):
        md = masks[('exactdp', s0)][i]
        mm = masks[('mse', s0)][i]
        ex['slots_in'].append(int((md & ~mm).sum()))
        ex['slots_out'].append(int((mm & ~md).sum()))
        ex['label_in'].append(float(y[md & ~mm].sum()))
        ex['label_out'].append(float(y[mm & ~md].sum()))
    exchange = {'mean_slots_in': round(float(np.mean(ex['slots_in'])), 3),
                'mean_slots_out': round(float(np.mean(ex['slots_out'])), 3),
                'mean_label_mass_in': round(float(np.mean(ex['label_in'])), 3),
                'mean_label_mass_out': round(float(np.mean(ex['label_out'])), 3),
                'net_label_mass': round(float(np.mean(ex['label_in']) - np.mean(ex['label_out'])), 4),
                'reading': 'net label mass > 0 at equal budget = the exact '
                           'quantity Q3.4 says must be positive for DP to beat '
                           'MSE at the cut'}
    # oracles on the same pool
    act_or, prefix_or = [], []
    for y in Y:
        G = float(y.sum())
        k = max(1, int(round(args.keep * len(y))))
        act_or.append(float(2 * min(k, G) / (k + G)))          # equal-cost ceiling
        order = np.argsort(-y)                                  # perfect ranking prefix
        prefix_or.append(max(f1_of(top_mask(order.astype(float), y, args.keep), y),
                             f1_of(top_mask(-order.astype(float), y, args.keep), y)))
    exchange['fixed_budget_action_oracle_mean'] = round(float(np.mean(act_or)), 4)
    exchange['fixed_ranking_prefix_oracle_mean'] = round(float(np.mean(prefix_or)), 4)
    res['mask_exchange_and_oracles'] = exchange

    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
