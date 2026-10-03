"""Temporal head on PHD2 fragments: pooling x structure matrix.

The keep-mask ceiling is frame-ranking AP, and the head that ships today reads
a mean-pooled 768-d vector through a 3-block dilated TCN (receptive field 15
frames).  Two independent causes for the ranking plateau are tested here:

  pooling   mean / attn / topk / mean+attn concat - the mean cannot tell a
            frame whose subject fills the view from one where it covers 5 %,
            which is exactly the distinction a keep decision turns on
  structure  single-scale dilated TCN vs multi-scale (parallel branches), whose
            receptive field must cover a whole official clip (p50 14 s) rather
            than a fragment (8 s)

Both axes are evaluated against the SAME binary in-interval label on the SAME
fragment set, with source-level splits, so the comparison is paired: any
difference is attributable to the factor, not to a different pool.

Labels are PHD2 GIF intervals (the official objective's own family) rather than
QV saliency, because the QV head was measured cross-domain and did not survive
the transfer; this keeps the supervision in-domain and varies only the pooling
and the receptive field.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch


def load_head_feats(feat_root, index, sel, keep_src, pooling):
    """Build (X, y, src) for one pooling variant over PHD2 fragments.

    Layout: ONE npz per fragment at <feat_root>/<frag_id>.npz with keys
    mean/attn/topk [8,768] and t [8] (v10_temporal_attnpool.py).  The first
    version of this loader expected a directory per fragment and matched
    nothing, which is why the first unattended run reported "no features".
    """
    rows = [json.loads(l) for l in Path(index).read_text().splitlines() if l.strip()]
    X, Y, SRC = [], [], []
    for r in rows:
        src = r.get('src')
        f = Path(feat_root) / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        keys = set(m.keys())
        if pooling == 'mean+attn':
            if not {'mean', 'attn'} <= keys:
                continue
            Xf = np.concatenate([m['mean'].astype(np.float32), m['attn'].astype(np.float32)], 1)
        elif pooling == 'mean+attn+topk':
            if not {'mean', 'attn', 'topk'} <= keys:
                continue
            Xf = np.concatenate([m['mean'].astype(np.float32), m['attn'].astype(np.float32),
                                 m['topk'].astype(np.float32)], 1)
        else:
            if pooling not in keys:
                continue
            Xf = m[pooling].astype(np.float32)
        L = len(Xf)
        t0 = float(r['t0'])
        ivs = []
        for recs in sel.get(src, {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']), float(rec['t1'])
                if b_ > a_:
                    ivs.append((a_ - t0, b_ - t0))
        stamps = sorted(float(x) for x in m['t'])
        times = np.array([stamps[i] - t0 for i in range(L)], np.float32)
        y = np.zeros(L, np.float32)
        half = ((stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1)) * 0.5 if len(stamps) > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if y.sum() == 0 or y.sum() == L:
            continue
        X.append(Xf); Y.append(y); SRC.append(src)
    return X, Y, SRC


class SingleScale(torch.nn.Module):
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


class MultiScale(torch.nn.Module):
    """Parallel dilated branches summed back onto the residual path, so a long
    highlight and a short one are both representable at their own time scale."""

    def __init__(self, d_in, ch=128, dils=(1, 2, 4, 8, 16)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d),
                                torch.nn.GELU()) for d in dils])
        self.merge = torch.nn.Conv1d(ch, ch, 1)
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        g = self.merge(sum(b(h) for b in self.branches))
        return self.out(torch.nn.functional.gelu(g + h)).squeeze(1)


def ap_of(s, y):
    s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    o = np.argsort(-s); ys = y[o]; cum = np.cumsum(ys)
    return float((cum / np.arange(1, len(ys) + 1) * ys).sum() / y.sum())


def recall_at(s, y, keep):
    s = np.asarray(s); y = np.asarray(y)
    if y.sum() == 0:
        return float('nan')
    k = max(1, int(round(keep * len(s))))
    return float(y[np.argsort(-s)[:k]].sum() / y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-root', type=Path, required=True)
    ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    ap.add_argument('--sources', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/val_sources.json'))
    ap.add_argument('--poolings', nargs='*', default=['mean', 'attn', 'topk', 'mean+attn'])
    ap.add_argument('--structs', nargs='*', default=['single', 'multi'])
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1])
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--ch', type=int, default=128)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--boot', type=int, default=4000)
    args = ap.parse_args()

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
    keep_src = set(json.loads(args.sources.read_text())) if args.sources.exists() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {'variants': {}, 'n_val_sources': 0}

    results['splits_note'] = ('train on the complement of --sources, report AP on '
                              'the held-out sources only, FIXED step count with no '
                              'checkpoint selection - the first version trained and '
                              'evaluated on the same fragments and its AP was training AP')
    for pooling in args.poolings:
        X, Y, SRC = load_head_feats(args.feat_root, args.index, sel, None, pooling)
        if not X:
            print(f'{pooling}: no features', flush=True); continue
        d_in = X[0].shape[1]
        eval_set = set(json.loads(args.sources.read_text())) if args.sources.exists() else set()
        tr = [(x, y) for x, y, s in zip(X, Y, SRC) if s not in eval_set]
        va = [(x, y) for x, y, s in zip(X, Y, SRC) if s in eval_set]
        if not tr or not va:
            print(f'{pooling}: split empty tr={len(tr)} va={len(va)}', flush=True); continue
        pos = float(np.mean([y.mean() for _, y in tr]))
        print(f'== pooling={pooling} d_in={d_in} train={len(tr)} heldout={len(va)} '
              f'srcs_heldout={len(set(s for s in SRC if s in eval_set))} pos={pos:.3f}', flush=True)
        results['n_val_sources'] = len(set(s for s in SRC if s in eval_set))

        Xt = torch.from_numpy(np.stack([x for x, _ in tr]))
        Yt = torch.from_numpy(np.stack([y for _, y in tr]))
        Xv = torch.from_numpy(np.stack([x for x, _ in va]))
        Yv = [y for _, y in va]

        for struct in args.structs:
            for seed in args.seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                Cls = SingleScale if struct == 'single' else MultiScale
                dils = (1, 2, 4) if struct == 'single' else (1, 2, 4, 8, 16)
                model = Cls(d_in, ch=args.ch, dils=dils).float()
                n_par = sum(p.numel() for p in model.parameters())
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
                for step in range(1, args.steps + 1):
                    model.train()
                    i = torch.randint(0, len(Xt), (args.batch,))
                    loss = torch.nn.functional.mse_loss(torch.sigmoid(model(Xt[i])), Yt[i])
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                # held-out AP at the final step; NO checkpoint selection anywhere
                model.eval()
                with torch.no_grad():
                    sv = torch.sigmoid(model(Xv)).numpy()
                ap = float(np.mean([ap_of(sv[j], Yv[j]) for j in range(len(Yv))]))
                rc = float(np.mean([recall_at(sv[j], Yv[j], 0.5) for j in range(len(Yv))]))
                key = f'{pooling}|{struct}|s{seed}'
                results['variants'][key] = {
                    'pooling': pooling, 'struct': struct, 'seed': seed,
                    'd_in': int(d_in), 'params': int(n_par),
                    'receptive_field': int(1 + 2 * sum(dils)),
                    'heldout_ap': round(ap, 4), 'heldout_recall_at_50': round(rc, 4)}
                if seed == args.seeds[0]:
                    torch.save({'state_dict': model.state_dict(), 'ch': args.ch, 'dils': list(dils),
                                'struct': struct, 'pooling': pooling, 'd_in': int(d_in),
                                'params': n_par, 'heldout_ap': ap},
                               args.output_dir / f'head_{pooling.replace("+","_")}_{struct}_s{seed}.pt')
                print(f'  -> {key}: heldout_ap={ap:.4f} r@50={rc:.4f}', flush=True)

    (args.output_dir / 'pool_structure_matrix.json').write_text(json.dumps(results, indent=1))
    # rank by mean AP across seeds, pooling then structure
    agg = {}
    for k, v in results['variants'].items():
        agg.setdefault((v['pooling'], v['struct']), []).append(v['heldout_ap'])
    print('\n=== ranking (mean AP over seeds) ===', flush=True)
    for (p_, s_), aps in sorted(agg.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print(f'  {p_:18s} {s_:7s} heldout_AP={np.mean(aps):.4f} (n={len(aps)})', flush=True)


if __name__ == '__main__':
    main()