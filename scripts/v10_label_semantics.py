"""Label-semantics probe: is the OR-union label the noise ceiling?  (P3, sec 15)

Every temporal head so far regresses a binary label built as the UNION of all
GIF selections: a second is positive if ANY user ever made it a highlight.  If
the official GT instead corresponds to individual selection behaviour, the OR
label widens positives, teaches over-selection, and caps ranking AP - the exact
wall every pooling/structure variant just hit (0.677-0.696, nothing escapes).

PHD2 stores per-user selections ({src: {clip: [{t0,t1}, ...]}}), so the same
head can be trained against three label protocols at zero annotation cost:

  or_union      y_t = 1 if any selection covers t          (incumbent)
  vote_fraction y_t = (#selections covering t) / #selections (soft target)
  any_strict    y_t = 1 only where ALL selections agree    (core highlights)

Same split, same head, same steps; the only variable is the label protocol.
Held-out AP against the OR label is still the comparable ranking metric, but
each variant also reports AP against its own protocol - a protocol that only
wins on its own metric has not proven anything.

A vote-fraction win would say the head CAN rank better once the label stops
flattening user disagreement; an any_strict win would say the union's noise is
in rarely-agreed seconds.
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


def ap_of(s, y):
    s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    o = np.argsort(-s); ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
    ap.add_argument('--eval-sources', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--ch', type=int, default=128)
    ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/label_semantics.json'))
    args = ap.parse_args()

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
    eval_set = set(json.loads(args.eval_sources.read_text()))
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]

    data = []
    for r in rows:
        f = args.feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        X = m['mean'].astype(np.float32)
        L = len(X)
        t0 = float(r['t0'])
        # per-interval coverage, counted so per-user multiplicity survives
        ints = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
                if b_ > a_:
                    ints.append((a_, b_))
        if not ints:
            continue
        stems = sorted(float(x) for x in m['t'])
        times = np.array([stems[i] - t0 for i in range(L)], np.float32)
        half = ((stems[-1] - stems[0]) / max(len(stems) - 1, 1)) * 0.5 if len(stems) > 1 else 0.5
        cover = np.zeros(L, np.float32)
        for a_, b_ in ints:
            cover[(times + half >= a_) & (times - half < b_)] += 1.0
        y_or = (cover > 0).astype(np.float32)
        y_vote = cover / len(ints)
        y_strict = (cover >= len(ints)).astype(np.float32)
        if y_or.sum() == 0 or y_or.sum() == L:
            continue
        data.append({'src': r['src'], 'X': X, 'y_or': y_or, 'y_vote': y_vote,
                     'y_strict': y_strict})

    tr = [d for d in data if d['src'] not in eval_set]
    va = [d for d in data if d['src'] in eval_set]
    print(f'fragments train={len(tr)} heldout={len(va)} '
          f'sources={len(set(d["src"] for d in data))}', flush=True)

    Xt = torch.from_numpy(np.stack([d['X'] for d in tr]))
    Xv = torch.from_numpy(np.stack([d['X'] for d in va]))
    out = {'n_train': len(tr), 'n_heldout': len(va), 'protocols': {}}
    for proto, key in (('or_union', 'y_or'), ('vote_fraction', 'y_vote'), ('any_strict', 'y_strict')):
        torch.manual_seed(0); np.random.seed(0)
        Yt = torch.from_numpy(np.stack([d[key] for d in tr]))
        model = TCN(ch=args.ch, dils=(1, 2, 4)).float()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        for step in range(1, args.steps + 1):
            model.train()
            i = torch.randint(0, len(Xt), (args.batch,))
            loss = torch.nn.functional.mse_loss(torch.sigmoid(model(Xt[i])), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Xv)).numpy()
        res = {'ap_vs_own_label': round(float(np.nanmean([ap_of(sv[j], va[j][key]) for j in range(len(va))])), 4),
               'ap_vs_or_label': round(float(np.nanmean([ap_of(sv[j], va[j]['y_or']) for j in range(len(va))])), 4),
               'ap_vs_vote_label': round(float(np.nanmean([ap_of(sv[j], va[j]['y_vote']) for j in range(len(va))])), 4)}
        out['protocols'][proto] = res
        print(f'{proto:14s} {res}', flush=True)

    args.out.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()