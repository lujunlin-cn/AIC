"""Round-5 A6 read: anchor vs neutral slicing stress (R5 section 7.2).

Question quantified: how much of the champion's keep-F1 comes from the
ANCHORED slicing protocol's position structure?  Same 460 old-confirm
sources, two 920-fragment slice sets:
  anchor   PHD2_FRAG_CONFIRM_V1 (65 percent GIF-anchored starts)
  neutral  ANCHOR_NEUTRAL_DEV_V1 (100 percent label-agnostic starts)

Conditions (frozen before the read):
  champion   LFM_V10/probe_deploy_head.pt (the VTREPLAY deployment head)
  slotprior  per-slot positive rate fitted on the DEV TRAIN pool ONLY
             (same prior vector for both slice sets - it encodes the
             anchor protocol's position structure by construction)
  zeros      champion head on zero features (pure structural output)

Metrics: keep-0.80 binary-temporal F1 on mixed-only fragments (both sets,
same rule) and on ALL fragments with positive labels reported separately;
AP on mixed.  THIS IS A DEV STRESS READ on demoted audit sources - it
never enters a promotion gate.

Output: /data/aic/experiments_910a/LFM_V11/round5_anchor_neutral_read.json
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--anchor-dir', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_CONFIRM_V1'))
ap.add_argument('--neutral-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/ANCHOR_NEUTRAL_DEV_V1'))
ap.add_argument('--dev-index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--dev-feat', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--head', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_anchor_neutral_read.json'))
args = ap.parse_args()


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


def load(feat_root, index, sel):
    X, Y, SRC, FID = [], [], [], []
    for r in [json.loads(l) for l in index.read_text().splitlines() if l.strip()]:
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
        X.append(Xf); Y.append(y); SRC.append(r['src']); FID.append(r['video_id'])
    return X, Y, SRC, FID


def f1_keep(s, y, keep):
    k = max(1, int(round(keep * len(y))))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum()) if ys.sum() else float('nan')


def main():
    sel = json.loads(args.selections.read_text())
    # slot prior from the DEV TRAIN pool only (never from either slice set)
    ev = set(json.loads(args.eval_sources.read_text()))
    _, Yt, _, SRCt = load(args.dev_feat, args.dev_index, sel)
    pos = np.zeros(8); tot = np.zeros(8)
    n_tr = 0
    for y, s in zip(Yt, SRCt):
        if s in ev or len(y) != 8:
            continue
        n_tr += 1
        for i in range(8):
            pos[i] += y[i]; tot[i] += 1
    prior = (pos / np.maximum(tot, 1)).astype(np.float32)

    ck = torch.load(args.head, map_location='cpu')
    head = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    head.load_state_dict(ck['state_dict']); head.eval()
    torch.set_grad_enabled(False)

    out = {'keep': args.keep, 'slot_prior_devtrain': np.round(prior, 4).tolist(),
           'n_dev_train_frags': n_tr, 'sets': {}}
    for name, d in (('anchor', args.anchor_dir), ('neutral', args.neutral_dir)):
        X, Y, SRC, FID = load(d / 'pool_feats', d / 'index.jsonl', sel)
        n_pos = sum(1 for y in Y if y.sum() > 0)
        mixed = [i for i, y in enumerate(Y) if 0 < y.sum() < len(y)]
        rec = {'n_frags': len(X), 'n_frags_with_pos': n_pos, 'n_mixed': len(mixed),
               'pos_frac': round(n_pos / max(len(X), 1), 4)}
        with torch.no_grad():
            S_champ = [head(torch.from_numpy(X[i])[None])[0].numpy() for i in range(len(X))]
            S_zero = [head(torch.zeros_like(torch.from_numpy(X[i]))[None])[0].numpy()
                      for i in range(len(X))]
        for cond, S in (('champion', S_champ), ('zeros', S_zero), ('slotprior', None)):
            f_mixed = [f1_keep(S[i] if S is not None else prior[:len(Y[i])],
                               Y[i], args.keep) for i in mixed]
            f_pos = [f1_keep(S[i] if S is not None else prior[:len(Y[i])],
                             Y[i], args.keep)
                     for i in range(len(X)) if Y[i].sum() > 0]
            rec[cond] = {
                'f1_mixed': round(float(np.mean(f_mixed)), 4) if f_mixed else None,
                'f1_all_pos': round(float(np.mean(f_pos)), 4) if f_pos else None,
                'ap_mixed': round(float(np.mean([ap_of(S[i], Y[i]) for i in mixed])), 4)
                if (S is not None and mixed) else None}
        out['sets'][name] = rec
        print(name, json.dumps(rec, indent=1), flush=True)

    a, n = out['sets']['anchor'], out['sets']['neutral']
    out['reading'] = {
        'champion_anchor_minus_neutral_f1_mixed':
            (round(a['champion']['f1_mixed'] - n['champion']['f1_mixed'], 4)
             if a['champion']['f1_mixed'] and n['champion']['f1_mixed'] else None),
        'slotprior_anchor_minus_neutral_f1_mixed':
            (round(a['slotprior']['f1_mixed'] - n['slotprior']['f1_mixed'], 4)
             if a['slotprior']['f1_mixed'] and n['slotprior']['f1_mixed'] else None),
        'note': 'large positive gaps mean the measured quality depends on the '
                'anchored slicing protocol; DEV stress read only'}
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['reading'], indent=1))
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
