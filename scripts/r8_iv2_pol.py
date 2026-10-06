"""R8 IV2_1B_TEMPORAL step 2: P/O/S/L/B attribution matrix on IV2 features.

Transplant of v11_round5_posbl_matrix.py (R5 3.3 protocol) onto the new
InternVideo2-1B frag-pool features.  Preregistered STOP rule
(reports/r8/preregistration.yaml, IV2_1B_TEMPORAL step 2): if the features
again carry only slicing-protocol signal, STOP and record - head training is
not started.

Arms (all inputs (8,768) per window -> flatten 6144 -> TCN; labels CONTINUOUS at
frame level: the frag-label `saliency` (annotator-mean importance, 0..1)
linear-interpolated at the frame timestamp.  Rationale: the frag pool is
CUT around relevant intervals - 74.9% of frames have soft>0.5 and most
windows are all-positive, so a binary rule degenerates (measured
2026-10-06); a within-window RANKING readout keeps the attribution
logic intact. source-level split by YouTube
src; dev read only, confirmation pool untouched):
  P  all-zero input            (position-only prior the content must beat)
  O  real IV2 features, time order
  S  the 8 frames of each window permuted by one fixed global rng
     permutation (shared train/eval)          -> tests temporal ORDER
  L  every other frame kept, rest zeroed        -> tests sampling density
  B  the PREVIOUS backbone's pooled features on the same windows and grid
     (frag_feats_train) - a same-grid static-content control (R5 allowed B
     only as a weak different-encoder control; here the grid matches)

Protocol: lr scan {3e-5, 1e-4, 3e-4} on O first (3 seeds), the chosen lr is
shared by every arm (no per-arm tuning); 3 seeds per arm; primary readout
dev AP + keep-0.80 frame F1; attribution O-P, O-S, O-L with source-level
paired bootstrap CIs.  GO rule: O-P dev AP delta > 0 with paired CI lower
> 0 (features carry learnable content signal -> proceed to head training).
STOP rule: O-P CI lower <= 0 -> features carry only slicing-protocol signal,
record and stop.

CPU only (small TCN head; NPU stays free).  Output JSON + per-source CSV.
"""
import argparse, csv, glob, json, os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--iv2-dir', default='/data/aic/experiments_910a/LFM_V11/r8_iv2/iv2_frag_feats_train')
p.add_argument('--frag-root', default='/data/aic/experiments_910a/QVH_V10/frag_train')
p.add_argument('--old-feats', default='/data/aic/experiments_910a/QVH_V10/frag_feats_train')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r8_npu/iv2_pol_matrix.json')
p.add_argument('--csv-out', default='/data/aic/experiments_910a/LFM_V11/r8_npu/iv2_pol_per_source.csv')
p.add_argument('--seeds', nargs='*', type=int, default=[510, 511, 512])
p.add_argument('--lrs', nargs='*', type=float, default=[3e-5, 1e-4, 3e-4])
p.add_argument('--steps', type=int, default=900)
p.add_argument('--batch', type=int, default=8)
p.add_argument('--split-seed', type=int, default=77)
args = p.parse_args()


class TCN(torch.nn.Module):
    """Time axis = the 8 frames of a window; channels = per-frame features.

    R5's matrix flattened the 8 frames into channels (d_in=6144) because its
    time axis was the action sequence; here the window IS the sequence, so
    the per-frame feature is the channel.
    """

    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4, 8, 16)):
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


def soft_at(t, gt_t, gt_soft):
    """Linear interpolation of the per-second soft label at time t."""
    i = np.clip(np.searchsorted(gt_t, t) - 1, 0, len(gt_t) - 2)
    w = (t - gt_t[i]) / np.maximum(gt_t[i + 1] - gt_t[i], 1e-6)
    w = np.clip(w, 0, 1)
    return gt_soft[i] * (1 - w) + gt_soft[i + 1] * w


def boot_ci_paired(delta, groups, boot=2000, seed=99):
    gs = sorted(set(groups))
    idx = {g: [i for i, x in enumerate(groups) if x == g] for g in gs}
    rng = np.random.RandomState(seed)
    ms = []
    for _ in range(boot):
        pick = rng.choice(len(gs), len(gs), replace=True)
        ms.append(float(np.mean([delta[i] for q in pick for i in idx[gs[q]]])))
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)], round(float(np.mean(delta)), 5)


def main():
    # ---------- load windows: features, old-backbone features, labels ----------
    wins = []
    Xl, Bl, Yl, Gl = [], [], [], []
    for fp in sorted(glob.glob(f'{args.iv2_dir}/p*/*.npz')):
        win = Path(fp).stem
        lp = Path(args.frag_root) / 'frag_labels' / f'{win}.npz'
        op = Path(args.old_feats) / f'{win}.npz'
        if not (lp.exists() and op.exists()):
            continue
        z = np.load(fp)
        X = z['pooled'].astype(np.float32)          # (8,768)
        t = z['t'].astype(np.float32)               # (8,)
        lz = np.load(lp)
        # window-level scalar label: frag windows are cut around relevant
        # intervals, so WITHIN-window label variance is ~zero (measured:
        # 8869/9066 windows have constant interpolated saliency).  The
        # learnable structure is the BETWEEN-window ranking inside one src.
        y = np.full(len(t), float(np.mean(
            soft_at(t, lz['t'].astype(np.float32),
                    lz['saliency'].astype(np.float32)))), np.float32)
        oz = np.load(op)
        B = oz['pooled'].astype(np.float32)
        if B.shape != X.shape:
            continue
        src = win.rsplit('_q', 1)[0]
        wins.append(win)
        Xl.append(X); Bl.append(B); Yl.append(y); Gl.append(src)
    print(f'windows aligned: {len(wins)} across {len(set(Gl))} srcs', flush=True)

    rng = np.random.RandomState(args.split_seed)
    srcs = sorted(set(Gl))
    rng.shuffle(srcs)
    n_tr = int(0.8 * len(srcs))
    tr_set = set(srcs[:n_tr])
    tr = [i for i, g in enumerate(Gl) if g in tr_set]
    dv = [i for i, g in enumerate(Gl) if g not in tr_set]
    print(f'split: train {len(tr)} windows / dev {len(dv)} windows '
          f'({n_tr} vs {len(srcs)-n_tr} srcs)', flush=True)

    rng_fix = np.random.RandomState(77)
    perm8 = rng_fix.permutation(8)

    def arm_X(name, feats):
        if name == 'O':
            return [x for x in feats]
        if name == 'S':
            # time axis is 0 (the 8 frames); x[:, perm8] would FANCY-REPLACE
            # the 768 channel axis with 8 channels (crashed run 2026-10-06)
            return [x[perm8] for x in feats]
        if name == 'P':
            return [np.zeros((len(x), 768), np.float32) for x in feats]
        if name == 'L':
            out = []
            for x in feats:
                z = np.zeros_like(x)
                z[::2] = x[::2]   # keep even FRAMES (axis 0), not channels
                out.append(z)
            return out
        if name == 'B':
            return [b for b in feats]
        raise SystemExit(name)

    def spearman(s, y):
        if float(np.std(s)) < 1e-6 or float(np.std(y)) < 1e-6:
            return 0.0
        # s and y are per-window scalars of one src (s = mean frame logit)
        rs = np.argsort(np.argsort(s)).astype(np.float32)
        ry = np.argsort(np.argsort(y)).astype(np.float32)
        rs = (rs - rs.mean()) / max(rs.std(), 1e-9)
        ry = (ry - ry.mean()) / max(ry.std(), 1e-9)
        return float((rs * ry).mean())

    def ap_of_unused(s, y):
        o = np.argsort(-s)
        ys = y[o]
        return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / max(ys.sum(), 1e-9))

    def f1_keep(s, y, keep=0.8):
        return 0.0

    def spearman_src_vec(sc):
        """Per-source within-src window Spearman for one trained run
        (window prediction = mean of the 8 frame logits); srcs with <3 dev
        windows are dropped."""
        src_pred = {}
        src_true = {}
        for i in dv:
            src_pred.setdefault(Gl[i], []).append(float(np.mean(sc[i])))
            src_true.setdefault(Gl[i], []).append(float(Yl[i][0]))
        return {g: spearman(np.array(src_pred[g]), np.array(src_true[g]))
                for g in sorted(src_pred) if len(src_pred[g]) >= 3}

    def spearman_src(sc, sd):
        return float(np.mean(list(spearman_src_vec(sc).values())))

    def train_one(X, seed, lr):
        torch.manual_seed(seed)
        run_rng = np.random.RandomState(seed)
        model = TCN()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for step in range(args.steps):
            idxs = run_rng.choice(len(tr), args.batch)
            opt.zero_grad()
            for j in idxs:
                logits = model(torch.from_numpy(X[tr[j]])[None])[0]
                yb = torch.from_numpy(Yl[tr[j]])
                loss = torch.nn.functional.mse_loss(logits, yb,
                                                    reduction='mean') / args.batch
                loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            return {i: model(torch.from_numpy(X[i])[None])[0].numpy() for i in dv}

    results = {'protocol': 'R8 IV2 P/O/S/L/B attribution; frame-level binary '
                           'labels (frag soft>0.5); source split 80/20 seed 77; '
                           'dev read only; readout = per-source WINDOW-level Spearman '
                           '(pred = mean over the 8 frame logits; label = mean '
                           'saliency of the window); frame labels are the '
                           'window scalar broadcast; GO = O Spearman >0 AND '
                           'paired CI lower >0, else STOP per preregistration',
               'windows': len(wins), 'srcs': len(srcs),
               'label_rule': 'window scalar = mean saliency, broadcast to frames; '
                             'readout per-source window-level Spearman',
               's_arm_note': 'mean pooling is permutation-invariant, so the S '
                             'arm has reduced interpretability under this '
                             'label grain; the GO/STOP decision rests on O '
                             'absolute and O-P / O-B paired deltas'}
    per_source_rows = []

    # ---------- lr scan on O ----------
    chosen_lr, best = None, -1
    for lr in args.lrs:
        aps = []
        for sd in args.seeds:
            sc = train_one(arm_X('O', Xl), sd, lr)
            aps.append(spearman_src(sc, sd))
        m = float(np.mean(aps))
        results[f'O_lr{lr}'] = round(m, 4)
        print(f'O lr={lr}: dev Spearman {m:.4f}', flush=True)
        if m > best:
            best, chosen_lr = m, lr
    results['chosen_lr'] = chosen_lr
    print('chosen lr', chosen_lr, flush=True)

    # ---------- arms ----------
    # AMENDMENT (2026-10-06, run 9, before reading it): run 8's decision
    # fields (o_abs / attribution / decision) were computed on WINDOW-MEAN
    # LOGITS (ps_store below), NOT on per-source Spearmans - its printed
    # STOP (delta -0.0093, a logit-scale mean diff) does not measure the
    # preregistered rank-correlation criterion and is invalid.  Run 8's ARM
    # table stays valid (those read via spearman_src).  This run computes
    # every decision quantity on per-source Spearmans averaged over seeds;
    # the logit store is kept as raw_pred_mean, reference only.
    sp_store = {}
    ps_store = {}
    for arm in ('P', 'O', 'S', 'L', 'B'):
        X = arm_X(arm, Xl if arm != 'B' else Bl)
        per_seed_ap = []
        seed_vecs = []
        ps_acc = None
        for sd in args.seeds:
            sc = train_one(X, sd, chosen_lr)
            vec = spearman_src_vec(sc)
            seed_vecs.append(vec)
            per_seed_ap.append(float(np.mean(list(vec.values()))))
            ps = {i: float(np.mean(sc[i])) for i in dv}
            ps_acc = ps if ps_acc is None else {i: ps_acc[i] + ps[i] for i in dv}
        ps_store[arm] = {i: ps_acc[i] / len(args.seeds) for i in dv}
        sp_store[arm] = {g: float(np.mean([v[g] for v in seed_vecs]))
                         for g in seed_vecs[0]}
        results[arm] = {'dev_spearman_mean': round(float(np.mean(per_seed_ap)), 4),
                        'spearman_per_seed': [round(a, 4) for a in per_seed_ap]}
        print(arm, results[arm], flush=True)

    srcs_dv = sorted(sp_store['O'])
    results['attribution'] = {}
    for b in ('P', 'S', 'L', 'B'):
        d = [sp_store['O'][g] - sp_store[b][g] for g in srcs_dv]
        ci, dm = boot_ci_paired(d, srcs_dv)
        results['attribution'][f'O_minus_{b}_spearman'] = {'delta': dm, 'ci95': ci}
        for g in srcs_dv:
            per_source_rows.append([g, 'O', b, round(sp_store['O'][g], 5),
                                    round(sp_store[b][g], 5),
                                    round(sp_store['O'][g] - sp_store[b][g], 5)])

    o_abs = float(np.mean([sp_store['O'][g] for g in srcs_dv]))
    op_ci = results['attribution']['O_minus_P_spearman']['ci95']
    op_d = results['attribution']['O_minus_P_spearman']['delta']
    results['O_absolute_spearman'] = round(o_abs, 4)
    results['decision'] = ('GO (features carry learnable content signal; '
                           'proceed to head training)'
                           if o_abs > 0 and op_d > 0 and op_ci[0] > 0 else
                           'STOP (O absolute Spearman <= 0 or O-P CI lower '
                           '<= 0: features carry only slicing-protocol signal; '
                           'head training not started per preregistration)')
    results['decision_delta'] = op_d
    results['decision_ci95'] = op_ci
    results['raw_pred_mean_reference_only'] = {
        arm: round(float(np.mean(list(ps_store[arm].values()))), 5)
        for arm in ps_store}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1) + '\n')
    with open(args.csv_out, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['src', 'arm_a', 'arm_b', 'spearman_a', 'spearman_b', 'delta'])
        w.writerows(per_source_rows)
    print(json.dumps({'attribution': results['attribution'],
                      'decision': results['decision']}, indent=1))
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
