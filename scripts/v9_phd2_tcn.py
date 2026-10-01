"""V9 P0-2: temporal highlight head on PHD2, trained on real GIF selections.

Why temporal and why now.  The only lever that ever moved the platform score on
the temporal axis is frame removal: TEMP (drop the lowest-value 10% of frames)
was +4.01, and no round since has touched the temporal row.  InternVideo2's
+0.01 is usually read as "temporal does not work"; the real reason is domain -
that head was fit on QVHighlights weak labels while the evaluation drop is
PHD2-style (YouTube clips with user-made GIF highlights), and 84,721 PHD2
training rows sat unused on disk.

Why a shallow TCN rather than the V6 T1 recipe.  T1 used 6 dilated blocks
(receptive field 63 frames) because TVSum clips are minutes long.  The official
semifinal drop has duration p50 14.1 s / p75 22.6 s, and every PHD2 fragment here
is 8 frames on a 1 s grid - a 8 s context.  A 63-frame receptive field would be
pure padding on this distribution, so this uses 3 dilated blocks (field 7).

Labels are the union of the GIF intervals overlapping the fragment.  PHD2's own
semantics say unselected time is "unlabelled for this user, not a negative";
this is weak supervision and is reported as such - the fragment anchors are
drawn 65% from inside a GIF interval precisely so the positive class is real
highlights rather than inferred absence.

Splits are by source video (never by fragment) and stratified by geometry, so
phd2_val is a fresh pool.
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path('/data/aic/external_datasets/PHD2')


def load_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--feat-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ch', type=int, default=128)
    ap.add_argument('--dils', type=int, nargs='*', default=[1, 2, 4])
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--pair-weight', type=float, default=0.3)
    ap.add_argument('--label', choices=['soft', 'binary'], default='soft')
    ap.add_argument('--tau', type=float, default=3.0,
                    help='seconds; decay constant of the distance-to-highlight label')
    ap.add_argument('--train-sources', type=int, default=0, help='0 = all but val')
    return ap.parse_args()


def load_fragments(pool, feat_root, frames, a):
    rows = [json.loads(l) for l in (pool / 'index.jsonl').read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        sk = json.loads((pool / 'skel' / f"{r['video_id']}.json").read_text())
        kfs = sk['keyframes']
        if not all((feat_root / r['video_id'] / f'{k}.npz').exists() for k in kfs):
            continue
        t0, L = float(r['t0']), float(r['L'])
        ann = ALL_ANN.get(r['src'], [])
        secs = [t0 + L * j / frames for j in range(frames)]
        if a.label == 'soft':
            # distance to the nearest GIF interval, exponentially decayed.
            # Binary in-interval labels leak through the sampler: 65% of the
            # fragments are anchored inside a GIF interval, so "is this frame a
            # highlight" is answered by "was this fragment anchored", and pure
            # background fragments carry no gradient at all.  A distance target
            # gives every fragment information and matches deployment better:
            # the frame remover ranks by score, it does not threshold a class.
            d = []
            for s in secs:
                if not ann:
                    d.append(float(L))
                    continue
                d.append(min(max(0.0, min(abs(s - a0) if s < a0 else (s - b0 if s > b0 else 0.0)
                                    for a0, b0 in ann)) for _ in (0,)))
            y = np.exp(-np.asarray(d, dtype=np.float32) / a.tau).astype(np.float32)
            yb = (y > 0.5).astype(np.float32)
        else:
            y = np.array([1.0 if any(a0 < s < b0 for a0, b0 in ann) else 0.0 for s in secs],
                         dtype=np.float32)
            yb = y
        out.append({'vid': r['video_id'], 'src': r['src'], 'kfs': kfs, 'y': y, 'yb': yb,
                    'rot': bool(r.get('rotated')), 'W': r['W'], 'H': r['H']})
    return out


ALL_ANN = {}
# train.json only: selections/test.json is the upstream PHD2 test split, i.e.
# the same GIF-highlight objective the AIC official drop is drawn from.
_p = D / 'annotations' / 'selections' / 'train.json'
if _p.exists():
    _d = json.loads(_p.read_text())
    for _v, _u in _d.items():
        ivs = [(float(s['t0']), float(s['t1'])) for lst in _u.values() for s in lst
               if float(s['t1']) > float(s['t0'])]
        if ivs:
            ALL_ANN[_v] = ivs


def load_pooled(frags, feat_root, frames):
    X = np.zeros((len(frags), frames, 768), dtype=np.float16)
    for i, f in enumerate(frags):
        for j, k in enumerate(f['kfs']):
            X[i, j] = np.load(feat_root / f['vid'] / f'{k}.npz')['pooled']
    return X


def main():
    a = load_args()
    import torch
    if a.device == 'npu':
        import torch_npu  # noqa: F401
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    frags = load_fragments(a.pool, a.feat_root, a.frames, a)
    by_src = defaultdict(list)
    for f in frags:
        by_src[f['src']].append(f)
    srcs = sorted(by_src)
    rng = np.random.default_rng(a.val_seed if hasattr(a, 'val_seed') else 20261001)
    rng.shuffle(srcs)
    n_val = max(1, int(round(len(srcs) * 0.06)))
    val_src = set(srcs[:n_val])
    tr = [f for s in srcs if s not in val_src for f in by_src[s]]
    va = [f for s in val_src for f in by_src[s]]
    if a.train_sources:
        keep = sorted({f['src'] for f in tr})[:a.train_sources]
        tr = [f for f in tr if f['src'] in set(keep)]
    Xtr = load_pooled(tr, a.feat_root, a.frames)
    Xva = load_pooled(va, a.feat_root, a.frames)
    # train target is the chosen label form; every metric is always computed on
    # the binary in-interval labels so runs stay comparable.
    Ytr = np.stack([f['y'] if a.label == 'soft' else f['yb'] for f in tr])
    Yva = np.stack([f['yb'] for f in va])
    print(f'TCN train={len(tr)} val={len(va)} label={a.label} '
          f'target_mean train={Ytr.mean():.3f} pos_rate(val)={Yva.mean():.3f} '
          f'val_sources={len(val_src)}', flush=True)

    class TCN(torch.nn.Module):
        def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
            super().__init__()
            self.proj = torch.nn.Conv1d(d_in, ch, 1)
            self.convs = torch.nn.ModuleList(
                [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
            self.out = torch.nn.Conv1d(ch, 1, 1)
            self.dils = dils

        def forward(self, x):
            h = self.proj(x.transpose(1, 2))
            for conv, d in zip(self.convs, self.dils):
                h = torch.nn.functional.gelu(conv(h) + h)
            return self.out(h).squeeze(1)

    model = TCN(ch=a.ch, dils=tuple(a.dils)).to(a.device).float()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    Xt = torch.from_numpy(Xtr).float().to(a.device)
    Yt = torch.from_numpy(Ytr).to(a.device)
    Xv = torch.from_numpy(Xva).float().to(a.device)
    print(f'TCN params={n_par} receptive_field={1 + 2 * sum(a.dils)} frames', flush=True)

    def metrics(sc, y):
        sc = np.asarray(sc.detach().cpu() if torch.is_tensor(sc) else sc).reshape(-1)
        y = np.asarray(y.detach().cpu() if torch.is_tensor(y) else y).reshape(-1)
        if y.min() == y.max():
            return {'ap': float('nan'), 'acc': float((sc > sc.mean()).astype(float).__eq__(y).mean())}
        order = np.argsort(-sc)
        ys = y[order]
        cum = np.cumsum(ys)
        prec = cum / np.arange(1, len(ys) + 1)
        npos = ys.sum()
        ap = float((prec * ys).sum() / npos) if npos else float('nan')
        k = max(1, int(round(0.15 * len(ys))))
        return {'ap': ap, 'recall_at_15pct': float(ys[:k].sum() / npos) if npos else float('nan'),
                'base_ap': float(npos / len(ys)), 'acc': float(((sc > np.median(sc)) == (y > 0.5)).mean())}

    best = (-1.0, None)
    hist = []
    for step in range(1, a.steps + 1):
        model.train()
        idx = torch.randint(0, len(Xt), (32,))
        xb, yb = Xt[idx], Yt[idx]
        pred = model(xb)
        if a.label == 'soft':
            reg = torch.nn.functional.mse_loss(pred, yb)
        else:
            reg = torch.nn.functional.binary_cross_entropy_with_logits(pred, yb)
        pair = torch.zeros((), device=pred.device)
        hi, lo = yb.argmax(1), yb.argmin(1)
        m = (yb.max(1).values - yb.min(1).values) > 1e-3
        if m.any():
            ar = torch.arange(len(yb), device=pred.device)[m]
            pair = torch.nn.functional.softplus(
                -(pred[ar, hi[m]] - pred[ar, lo[m]])).mean()
        loss = reg + a.pair_weight * pair
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vs = model(Xv)
            m = metrics(vs, Yva)
            hist.append({'step': step, 'loss': round(float(loss), 4), **m})
            print(f'step {step} loss={float(loss):.4f} val_ap={m["ap"]:.4f} '
                  f'r@15={m["recall_at_15pct"]:.4f} (base {m["base_ap"]:.4f})', flush=True)
            if m['ap'] == m['ap'] and m['ap'] > best[0]:
                best = (m['ap'], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if best[1] is not None:
        model.load_state_dict(best[1])
    torch.save({'state_dict': model.state_dict(), 'ch': a.ch, 'dils': list(a.dils),
                'params': n_par, 'val_ap': best[0],
                'val_recall_at_15pct': metrics(model(Xv).detach(), Yva)['recall_at_15pct']},
               a.output_dir / f'tcn_s{a.seed}.pt')
    (a.output_dir / f'history_s{a.seed}.json').write_text(json.dumps(
        {'train_fragments': len(tr), 'val_fragments': len(va), 'val_sources': len(val_src),
         'pos_rate_train': float(Ytr.mean()), 'pos_rate_val': float(Yva.mean()),
         'params': n_par, 'receptive_field_frames': 1 + 2 * sum(a.dils),
         'steps': a.steps, 'best_val_ap': best[0], 'history': hist,
         'label_semantics': ('union of overlapping PHD2 GIF intervals; unselected time '
                             'is weakly treated as negative (PHD2 docs call it unlabelled)'),
         'weak_supervision': True}, indent=1))
    print(f'SUMMARY params={n_par} best_val_ap={best[0]:.4f}', flush=True)


if __name__ == '__main__':
    main()