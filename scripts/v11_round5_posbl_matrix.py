"""Round-5 B5: P/O/S/B/L attribution matrix on native tubelet features.

Arms (R5 section 3.3, preregistered):
  P  position-only     : input = zeros [n_act, 8, 768] (TCN reads only its
                         own position structure) - the prior the content
                         arms must beat
  O  ordered native    : real [n_act, 8, 768] tubelet anchors, in time order
  S  shuffled tubelets : the 8 tubelets of each window permuted by a FIXED
                         rng per window (train and eval share the shuffling)
  B  static content    : LFM SigLIP per-frame pooled features [8, 768]
                         (the existing 8x1s-keyframe contract; static,
                         order-invariant aggregation by construction)
  L  1 Hz sparse       : native tubelet anchors downsampled to 4 of 8
                         (every other tubelet -> ~1 Hz effective) and
                         zero-padded back to length 8, so the head sees
                         the same shape with half the temporal detail

Labels: original-action binary labels - a tubelet is positive iff its
center time falls inside a GT interval (projected via selections).

Protocol (R5 3.4):
  * lr scan {3e-5, 1e-4, 3e-4} for the O arm first, 900 steps, batch 8;
    the chosen lr is shared by all arms (no per-arm tuning);
  * train on the 128 train sources, read on the 128 dev sources;
  * DEV READ ONLY - never a promotion gate; the fresh confirm set is NOT
    touched here;
  * primary readout: AP and keep-0.80 original-action F1, per arm;
  * attribution rules: O must beat P; O > S/B supports temporal ORDER;
    O > L supports dense sampling.  O > P only => multi-frame content.

Output: /data/aic/experiments_910a/LFM_V11/round5_posbl_matrix.json
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/R5_NATIVE_FEATS_V1'))
ap.add_argument('--sources-file', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_native_dev_sources.json'))
ap.add_argument('--bfeat-dir', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--seeds', nargs='*', type=int, default=[510, 511, 512])
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lrs', nargs='*', type=float, default=[3e-5, 1e-4, 3e-4])
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_posbl_matrix.json'))
args = ap.parse_args()


class TCN(torch.nn.Module):
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


def load_labels(sel):
    out = {}
    for src, users in sel.items():
        ivs = []
        for recs in users.values():
            for r in recs:
                if float(r['t1']) > float(r['t0']):
                    ivs.append((float(r['t0']), float(r['t1'])))
        if ivs:
            out[src] = ivs
    return out


def load_native(feat_dir, srcs, labels):
    """Per source: X [n_act, 8, 768], centers [n_act, 8], y [n_act, 8]."""
    X, C, Y, keep = [], [], [], []
    for s in srcs:
        p = feat_dir / f'{s}.npz'
        if not p.exists():
            continue
        d = np.load(p)
        meta = json.loads(str(d['meta']))
        centers = np.array([m['tubelet_centers_s'] for m in meta], np.float32)
        y = np.zeros_like(centers)
        for a in range(len(centers)):
            for t in range(8):
                ct = centers[a, t]
                for lo, hi in labels.get(s, []):
                    if lo <= ct <= hi:
                        y[a, t] = 1.0
                        break
        if not (0 < y.sum() < y.size):
            continue
        X.append(d['feats'].astype(np.float32))
        C.append(centers)
        Y.append(y)
        keep.append(s)
    return X, C, Y, keep


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def f1_keep(s, y, keep=0.8):
    n = len(y)
    k = max(1, int(round(keep * n)))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def train_arm(X, Y, tr, dv, arm, seed, lr, steps, batch, rng):
    torch.manual_seed(seed)
    model = TCN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        i = rng.choice(len(tr), batch)
        xb = torch.stack([X[tr[j]] for j in i])
        yb = torch.from_numpy(np.concatenate([Y[tr[j]] for j in i])).float()
        logits = model(xb)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yb, reduction='mean')
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    model.eval()
    with torch.no_grad():
        sc = {i: model(X[i:i + 1])[0].numpy().astype(np.float32)
              for i in dv}
    return model, sc


def main():
    sel = json.loads(args.selections.read_text())
    labels = load_labels(sel)
    src_sel = json.loads(args.sources_file.read_text())
    tr_srcs = sorted(set(src_sel['train_sources']))
    dv_srcs = sorted(set(src_sel['eval_sources']))
    Xl, Cl, Yl, keep = load_native(args.feat_dir, tr_srcs + dv_srcs, labels)
    keep_set = set(keep)
    tr = [i for i, s in enumerate(keep) if s in set(tr_srcs)]
    dv = [i for i, s in enumerate(keep) if s in set(dv_srcs)]
    print(f'native pools: train {len(tr)}, dev {len(dv)} sources', flush=True)

    # arm inputs
    X = {}
    X['O'] = [x.copy() for x in Xl]
    rng_fix = np.random.RandomState(77)
    X['S'] = [x[:, rng_fix.permutation(8)].copy() for x in Xl]
    X['P'] = [np.zeros_like(x) for x in Xl]
    X['L'] = []
    for x in Xl:
        z = np.zeros_like(x)
        z[:, ::2] = x[:, ::2]                    # 4 of 8 tubelets (~1 Hz)
        X['L'].append(z)
    # B arm: LFM static features on the SAME sources where available is a
    # different fragmenting; R5 allows a static-CONTENT control with a
    # different encoder only as a WEAK control - run it only if the pooled
    # LFM features cover the same sources, else skip with a note.
    B_cov = None
    if args.bfeat_dir.exists():
        idx = args.bfeat_dir / 'index.jsonl'
        if idx.exists():
            vids = {}
            for l in idx.read_text().splitlines():
                r = json.loads(l)
                vids.setdefault(r['src'], []).append(r['video_id'])
            B_cov = sum(1 for s in keep if s in vids) / max(len(keep), 1)
    print(f'B-arm LFM coverage on native sources: {B_cov}', flush=True)
    if B_cov is not None and B_cov > 0.5:
        bf = args.bfeat_dir / 'pool_feats'
        X['B'] = []
        for s in keep:
            fs = []
            for vid in sorted(set(vids.get(s, []))):
                p = bf / f'{vid}.npz'
                if p.exists():
                    fs.append(np.load(p)['mean'].astype(np.float32))
            if fs:
                X['B'].append(np.concatenate(fs, 0))
            else:
                X['B'].append(np.zeros((8, 768), np.float32))
    else:
        X['B'] = None

    results = {}
    # lr scan on O, dev read picks the lr shared by all arms
    chosen_lr = None
    for lr in args.lrs:
        aps = []
        for sd in args.seeds:
            rng = np.random.RandomState(sd)
            _, sc = train_arm(X['O'], Yl, tr, dv, 'O', sd, lr, args.steps,
                              args.batch, rng)
            aps.append(float(np.mean([ap_of(sc[i], Yl[i]) for i in dv])))
        results[f'O_lr{lr}'] = {'dev_ap_mean': round(float(np.mean(aps)), 4),
                                'per_seed': [round(a, 4) for a in aps]}
        print(f'O lr={lr}: dev AP {results[f"O_lr{lr}"]["dev_ap_mean"]}', flush=True)
    chosen_lr = max(args.lrs,
                    key=lambda l: results[f'O_lr{l}']['dev_ap_mean'])
    results['chosen_lr'] = chosen_lr

    for arm in ('P', 'O', 'S', 'L'):
        if X.get(arm) is None:
            continue
        aps, f1s = [], []
        for sd in args.seeds:
            rng = np.random.RandomState(sd)
            _, sc = train_arm(X[arm], Yl, tr, dv, arm, sd, chosen_lr,
                              args.steps, args.batch, rng)
            aps.append(float(np.mean([ap_of(sc[i], Yl[i]) for i in dv])))
            f1s.append(float(np.mean([f1_keep(sc[i], Yl[i]) for i in dv])))
        results[arm] = {'dev_ap_mean': round(float(np.mean(aps)), 4),
                        'dev_f1_mean': round(float(np.mean(f1s)), 4),
                        'ap_per_seed': [round(a, 4) for a in aps]}
        print(arm, results[arm], flush=True)
    results['B_note'] = ('static-content control uses LFM 8x1s pooled features '
                         'on matching sources when coverage >0.5; weak control '
                         '(different encoder), attribution per R5 3.3')
    results['attribution'] = {
        'O_minus_P_ap': round(results['O']['dev_ap_mean'] - results['P']['dev_ap_mean'], 4),
        'O_minus_S_ap': round(results['O']['dev_ap_mean'] - results['S']['dev_ap_mean'], 4),
        'O_minus_L_ap': round(results['O']['dev_ap_mean'] - results['L']['dev_ap_mean'], 4)}
    args.out.write_text(json.dumps(results, indent=1) + '\n')
    print(json.dumps(results['attribution'], indent=1))
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
