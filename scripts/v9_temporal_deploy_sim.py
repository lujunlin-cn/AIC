"""Deployment simulation for the PHD2 temporal head.

Frame-level AP is not the quantity the platform sees.  The deployed remover
keeps every frame above a threshold, so what matters is the precision/recall
trade the ranking buys at each keep ratio.  This replays the trained TCN on the
held-out pool and compares three keep policies at several keep ratios:

    all      keep everything (the current production policy)
    model    keep the top (1-k) fraction by TCN score
    random   keep a random (1-k) fraction  - the honest control, since "delete
             the worst k%" only helps if the model knows which frames are worst

Reported as frame-level P / R / F1 against the PHD2 GIF intervals, macro-averaged
over fragments, plus the paired per-fragment delta of F1(model) - F1(random) with
a source-level bootstrap CI.  A positive delta at some k is the precondition for
a temporal package; a delta that straddles zero means the ranking is not yet
worth deleting frames on, whatever AP says.

Caveat carried into the output: PHD2 GIF intervals are weak labels (unselected
time is "unlabelled", not negative), so these numbers bound the mechanism, they
are not a score prediction.
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True)
ap.add_argument('--feat-root', type=Path, required=True)
ap.add_argument('--ckpt', type=Path, required=True)
ap.add_argument('--frames', type=int, default=8)
ap.add_argument('--val-frac', type=float, default=0.06)
ap.add_argument('--seed', type=int, default=20261001)
ap.add_argument('--keeps', type=float, nargs='*', default=[1.0, 0.95, 0.9, 0.8, 0.7, 0.5])
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', type=Path, default=None)
a = ap.parse_args()

import torch
D = Path('/data/aic/external_datasets/PHD2')
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)

rows = [json.loads(l) for l in (a.pool / 'index_v2.jsonl').read_text().splitlines() if l.strip()]
ann_all = {}
for vid, users in json.loads((D / 'annotations' / 'selections' / 'train.json').read_text()).items():
    ivs = [(float(s['t0']), float(s['t1'])) for lst in users.values() for s in lst
           if float(s['t1']) > float(s['t0'])]
    if ivs:
        ann_all[vid] = ivs

frags = []
for r in rows:
    sk = json.loads((a.pool / 'skel' / f"{r['video_id']}.json").read_text())
    if not all((a.feat_root / r['video_id'] / f'{k}.npz').exists() for k in sk['keyframes']):
        continue
    ivs = ann_all.get(r['src'], [])
    t0, L = float(r['t0']), float(r['L'])
    y = np.array([1.0 if any(u < t0 + L * j / a.frames < v for u, v in ivs) else 0.0
                  for j in range(a.frames)], dtype=np.float32)
    if y.sum() == 0 or y.sum() == a.frames:
        continue                      # no temporal signal in this fragment
    frags.append((r['video_id'], sk['keyframes'], y))

by_src = {}
for fid, kfs, y in frags:
    by_src.setdefault(fid.rsplit('_f', 1)[0], []).append((fid, kfs, y))
srcs = sorted(by_src)
rng = np.random.default_rng(a.seed)
rng.shuffle(srcs)
val_src = set(srcs[:max(1, int(round(len(srcs) * a.val_frac)))])
val = [f for s in val_src for f in by_src[s]]
print(f'DEPLOY val fragments={len(val)} sources={len(val_src)} '
      f'pos_rate={np.mean([y.mean() for _, _, y in val]):.3f}', flush=True)

kept, Xs, Ys, ids = [], [], [], []
for fid, kfs, y in val:
    rows_p, ok = [], True
    for k in kfs:
        f = a.feat_root / fid / f'{k}.npz'
        if not f.exists():
            ok = False
            break
        rows_p.append(np.load(f)['pooled'])
    if not ok:
        continue
    kept.append((fid, kfs, y))
    Xs.append(np.stack(rows_p))
    Ys.append(y)
    ids.append(fid.rsplit('_f', 1)[0])
if len(kept) != len(val):
    print(f'  skipped {len(val) - len(kept)} fragments with missing features')
val = kept
X = np.stack(Xs).astype(np.float32)
Y = np.stack(Ys)


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for conv in self.convs:
            h = torch.nn.functional.gelu(conv(h) + h)
        return self.out(h).squeeze(1)


model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
model.load_state_dict(ck['state_dict'])
model.eval()
with torch.no_grad():
    S = model(torch.from_numpy(X)).numpy()


def prf(keep_mask, y):
    tp = float((keep_mask & (y > 0.5)).sum())
    fp = float((keep_mask & (y <= 0.5)).sum())
    fn = float((~keep_mask & (y > 0.5)).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return p, r, (0.0 if p + r == 0 else 2 * p * r / (p + r))


rng2 = np.random.default_rng(a.seed + 1)
rows_out = {}
src_ids = ids
uniq = sorted(set(src_ids))
per_src = {s: [] for s in uniq}
for keep in a.keeps:
    f1_model, f1_rand, f1_all, src_acc = [], [], [], {s: [] for s in uniq}
    for i in range(len(val)):
        y = Y[i]
        n_keep = max(1, int(round(keep * a.frames)))
        order = np.argsort(-S[i])
        m_model = np.zeros(a.frames, dtype=bool)
        m_model[order[:n_keep]] = True
        m_rand = np.zeros(a.frames, dtype=bool)
        m_rand[rng2.permutation(a.frames)[:n_keep]] = True
        m_all = np.ones(a.frames, dtype=bool)
        fm = prf(m_model, y)[2]
        fr = prf(m_rand, y)[2]
        fa = prf(m_all, y)[2]
        f1_model.append(fm)
        f1_rand.append(fr)
        f1_all.append(fa)
        src_acc[src_ids[i]].append((fm, fr, fa))
    d = np.array(f1_model) - np.array(f1_rand)
    bs = np.array([np.mean([x[0] - x[1] for x in src_acc[s]]) for s in uniq])
    idx = rng2.integers(0, len(bs), size=(a.boot, len(bs)))
    boot = bs[idx].mean(1)
    rows_out[f'{keep:.2f}'] = {
        'keep_frac': keep,
        'f1_all_frames': round(float(np.mean(f1_all)), 4),
        'f1_model': round(float(np.mean(f1_model)), 4),
        'f1_random_keep_same_budget': round(float(np.mean(f1_rand)), 4),
        'delta_vs_random': round(float(np.mean(f1_model) - np.mean(f1_rand)), 4),
        'delta_vs_all_frames': round(float(np.mean(f1_model) - np.mean(f1_all)), 4),
        'ci95_source_bootstrap': [round(float(np.percentile(boot, 2.5)), 4),
                                  round(float(np.percentile(boot, 97.5)), 4)],
        'n_sources': len(bs),
    }
    r = rows_out[f'{keep:.2f}']
    print(f'keep={keep:.2f}  F1 all={r["f1_all_frames"]:.4f} model={r["f1_model"]:.4f} '
          f'rand={r["f1_random_keep_same_budget"]:.4f} delta={r["delta_vs_random"]:+.4f} '
          f'CI[{r["ci95_source_bootstrap"][0]:+.4f},{r["ci95_source_bootstrap"][1]:+.4f}]',
          flush=True)

out = {'ckpt': str(a.ckpt), 'val_fragments': len(val), 'val_sources': len(val_src),
       'pos_rate': round(float(Y.mean()), 4), 'labels': 'binary in-interval (PHD2 weak)',
       'policy_grid': rows_out}
if a.out:
    a.out.write_text(json.dumps(out, indent=1))
print('WROTE', a.out)