"""R6 T-CONTEXT step 3: the 2x2 arm matrix + position baseline (R6 Q2).

Arms (preregistered, 3 seeds each, source-level split 210 train / 90 dev):
  Slicing arm:
    anchor : the event's ANCHOR variant only  (historical protocol shape:
             event starts at t=1 s inside a 12 s fragment)
    multi  : FRONT + MID + BACK variants of the same events (the event
             appears at all three relative positions in training support)
  Head arm:
    tcn    : the healthy TCN (d_in 768, ch 128, dils 1,2,4) - the deployed
             temporal architecture family
    dense  : dense multi-scale head (FlashVTG-inspired: 128 d, scales
             1/2/4, skip connections, per-frame logits)

Supervision: per-keyframe BCE, y = 1 iff the keyframe's absolute time is
inside any merged GT interval of the source (other real annotations kept).
Primary read: dev-source original-frame temporal macro F with the
keep-0.80 mask per variant (the posbl convention), equal weight per
variant, source-cluster paired CI.
Baselines: slotprior (per-slot positive rate of the TRAINING pool; no
model).  Gate: multi must beat anchor AND slotprior by >= +0.007, both
paired CI lower > 0; positive-video noninferiority <= 0.005.
Diagnostics: per-slot positive-rate tables (the equivariance table).

CPU only.  Output: r6_temporal_context_matrix.json
"""
import argparse, glob, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--plan', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_t_context_plan.json'))
ap.add_argument('--feats', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/t_context_feats'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_temporal_context_matrix.json'))
args = ap.parse_args()


def merge(ivs):
    ivs = sorted(ivs)
    out = []
    for a, b in ivs:
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


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


class DenseMultiScale(torch.nn.Module):
    """FlashVTG-inspired dense head: 128 d, scales 1/2/4, skip connections."""

    def __init__(self, d_in=768, ch=128):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.s1 = torch.nn.Conv1d(ch, ch, 3, padding=1)
        self.s2 = torch.nn.Conv1d(ch, ch, 3, padding=2, dilation=2)
        self.s4 = torch.nn.Conv1d(ch, ch, 3, padding=4, dilation=4)
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv1d(3 * ch, ch, 1), torch.nn.GELU(),
            torch.nn.Conv1d(ch, ch, 3, padding=1))
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.proj(x.transpose(1, 2)))
        hs = [torch.nn.functional.gelu(c(h)) for c in (self.s1, self.s2, self.s4)]
        h = self.fuse(torch.cat(hs, 1)) + h
        return self.out(h).squeeze(1)


def main():
    plan = json.loads(args.plan.read_text())
    sel = json.loads(args.selections.read_text())
    gt = {}
    for rec in plan['sources']:
        ivs = []
        for recs in sel[rec['src']].values():
            for r in recs:
                if float(r['t1']) > float(r['t0']):
                    ivs.append((float(r['t0']), float(r['t1'])))
        gt[rec['src']] = merge(ivs)

    # ---- load variants ----
    X, Y, SRC, ARM, EV = [], [], [], [], []
    pos_rate = np.zeros(8); pos_tot = 0
    for rec in plan['sources']:
        for e in rec['events']:
            vd = f'{e["a"]:.2f}_{e["b"]:.2f}'
            for wname, v in e['variants'].items():
                p = args.feats / rec['src'] / f'{vd}_{wname}.npz'
                if not p.exists():
                    continue
                d = np.load(p)
                if 'mean' not in d:
                    continue
                x = d['mean'].astype(np.float32)
                if x.shape[0] != 8:
                    continue
                stamps = np.array(v['frames'], np.float32)
                y = np.zeros(8, np.float32)
                for a, b in gt[rec['src']]:
                    y[(stamps >= a) & (stamps <= b)] = 1.0
                if not (0 < y.sum() < 8):
                    continue
                X.append(x); Y.append(y); SRC.append(rec['src'])
                ARM.append('anchor' if wname == 'anchor' else 'multi')
                EV.append(wname)
                pos_rate += y; pos_tot += 1
    X = np.stack(X); Y = np.stack(Y); SRC = np.array(SRC); ARM = np.array(ARM)
    print(f'variants loaded: {len(X)} (anchor {(ARM == "anchor").sum()}, '
          f'multi {(ARM == "multi").sum()})', flush=True)

    train_srcs = {r['src'] for r in plan['sources'] if r['split'] == 'train'}
    dv_idx = np.array([i for i, s in enumerate(SRC) if s not in train_srcs])
    tr_idx = np.array([i for i, s in enumerate(SRC) if s in train_srcs])
    prior = (pos_rate / max(pos_tot, 1)).astype(np.float32)
    print(f'slot prior (train pool): {np.round(prior, 3).tolist()}', flush=True)

    def f_keep08(pred, y):
        k = max(1, int(round(0.8 * len(y))))
        idx = set(np.argsort(-pred)[:k].tolist())
        tp = float(sum(y[i] for i in idx))
        return 2 * tp / (k + float(y.sum()))

    def run(arm_set, head_cls, seed):
        tr = [i for i in tr_idx if ARM[i] in arm_set]
        torch.manual_seed(seed)
        model = head_cls()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        rng = np.random.RandomState(seed)
        for st in range(args.steps):
            idx = rng.choice(len(tr), args.batch)
            xb = torch.from_numpy(X[tr[idx]]).transpose(1, 2)
            yb = torch.from_numpy(Y[tr[idx]])
            opt.zero_grad()
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        per_var = {}
        with torch.no_grad():
            for i in dv_idx:
                if ARM[i] not in arm_set:
                    continue
                pred = model(torch.from_numpy(X[i])[None].transpose(1, 2))[0].numpy()
                per_var[i] = f_keep08(pred, Y[i])
        return per_var

    def macro(per_var):
        # equal weight per variant, grouped report by source and by positive
        vals = np.array(list(per_var.values()))
        pos = [i for i in per_var if Y[i].sum() >= 4]
        return (float(vals.mean()),
                float(np.mean([per_var[i] for i in pos])) if pos else 0.0)

    def paired_ci(a, b):
        keys = sorted(set(a) & set(b))
        delta = [a[k] - b[k] for k in keys]
        srcs = [SRC[k] for k in keys]
        vs = sorted(set(srcs))
        idxm = {v: [i for i, s in enumerate(srcs) if s == v] for v in vs}
        rng = np.random.RandomState(99)
        ms = []
        for _ in range(2000):
            pick = rng.choice(len(vs), len(vs), replace=True)
            ms.append(float(np.mean([delta[i] for p in pick for i in idxm[vs[p]]])))
        lo, hi = np.percentile(ms, [2.5, 97.5])
        return round(float(np.mean(delta)), 5), [round(float(lo), 5), round(float(hi), 5)]

    res = {'protocol': 'R6 T-CONTEXT 2x2 + slotprior; keep-0.80 original-frame '
                       'F per variant, equal weight; gate: multi >= +0.007 vs '
                       'anchor AND slotprior, both CI lower > 0; positive-video '
                       'noninferiority <= 0.005',
           'slot_prior': np.round(prior, 3).tolist(),
           'n_variants': {'anchor': int((ARM == 'anchor').sum()),
                          'multi': int((ARM == 'multi').sum())},
           'arms': {}}

    # slotprior baseline (no model)
    sp = {i: f_keep08(prior, Y[i]) for i in dv_idx}
    m_sp, pos_sp = macro(sp)
    res['arms']['slotprior'] = {'f': round(m_sp, 5), 'f_positive_variants': round(pos_sp, 5)}

    detail = {'slotprior': sp}
    for arm_set, aname in ((('anchor',), 'anchor'), (('multi',), 'multi')):
        for head_cls, hname in ((TCN, 'tcn'), (DenseMultiScale, 'dense')):
            per_seed = []
            for sd in args.seeds:
                pv = run(arm_set, head_cls, sd)
                per_seed.append(pv)
            keys = sorted(set.intersection(*[set(p) for p in per_seed]))
            mean_pv = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
            m, pm = macro(mean_pv)
            detail[f'{aname}+{hname}'] = mean_pv
            res['arms'][f'{aname}+{hname}'] = {
                'f': round(m, 5), 'f_positive_variants': round(pm, 5),
                'per_seed_f': [round(float(np.mean(list(p.values()))), 5) for p in per_seed]}
            print(f'{aname}+{hname}: F {m:.5f} (pos {pm:.5f})', flush=True)

    # gates: paired CIs + positive-variant noninferiority
    res['gate'] = {}
    for hname in ('tcn', 'dense'):
        a, mlt = f'anchor+{hname}', f'multi+{hname}'
        dm_vs_anchor, ci_vs_anchor = paired_ci(detail[mlt], detail[a])
        dm_vs_sp, ci_vs_sp = paired_ci(detail[mlt], detail['slotprior'])
        # positive-variant noninferiority: variants whose GT coverage >= 4/8
        pos_keys = [k for k in detail[mlt] if Y[k].sum() >= 4]
        pos_deg = float(np.mean([detail[a][k] for k in pos_keys if k in detail[a]] or [0])) \
            - float(np.mean([detail[mlt][k] for k in pos_keys]))
        res['gate'][f'{hname}'] = {
            'multi_vs_anchor': {'delta': dm_vs_anchor, 'ci95': ci_vs_anchor},
            'multi_vs_slotprior': {'delta': dm_vs_sp, 'ci95': ci_vs_sp},
            'positive_variant_degradation': round(-pos_deg, 5),
            'gate_pass': bool(dm_vs_anchor >= 0.007 and ci_vs_anchor[0] > 0 and
                              dm_vs_sp >= 0.007 and ci_vs_sp[0] > 0 and
                              pos_deg >= -0.005)}
        print(f'gate {hname}:', res['gate'][hname], flush=True)
    res['_detail'] = {k: {str(i): round(v, 5) for i, v in d.items()}
                      for k, d in detail.items()}
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res['arms'], indent=1))
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
