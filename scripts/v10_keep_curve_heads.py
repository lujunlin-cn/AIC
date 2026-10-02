"""Keep-rate deployment simulation: native head vs QV head vs fusion.

Decides whether the QVHighlights temporal head earns a place in the mask.  The
question is not its QV-domain AP (0.81, which only shows it learned QV) but
what it does to F1 on PHD2 fragments, which carry the same GIF-interval
objective the official drop is built from.

F1 here mirrors the official arithmetic: keeping a subset of frames makes the
prediction set smaller, and

    F1 = 2 * iou_sum / (n_pred + n_gt)

so cutting frames that are not highlights raises F1 even with unchanged boxes.
That is the whole point of the time axis, and it is why the scan goes down to
aggressive keep rates instead of stopping at 0.80.

Scores per frame come from the frozen heads; the fusion variant averages the
two heads' sigmoid scores after per-head z-normalisation, which is the honest
way to combine two heads trained on different label scales.

Reported per keep rate: F1, delta vs keeping everything, delta vs a
same-budget random keep, and a source-level bootstrap CI on the delta vs
random (resampling source videos, since frames inside one clip are not
independent).
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def load_head(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2))))
    m.load_state_dict(ck['state_dict']); m.eval()
    return m, ck


ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/lfm_feats'))
ap.add_argument('--sources', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/val_sources.json'))
ap.add_argument('--native-ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'))
ap.add_argument('--qv-ckpt', type=Path, default=Path('/data/aic/experiments_910a/QVH_V10/tcn_frag_sal/qvh_frag_tcn_s0.pt'))
ap.add_argument('--keeps', type=float, nargs='*', default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
ap.add_argument('--boot', type=int, default=4000)
ap.add_argument('--seed', type=int, default=20261002)
ap.add_argument('--out', type=Path, default=None)
args = ap.parse_args()

keep_src = set(json.loads(args.sources.read_text())) if args.sources.exists() else None
sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
heads = {}
for nm, p in (('native', args.native_ckpt), ('qv', args.qv_ckpt)):
    if p.exists():
        heads[nm], _ = load_head(p)
print('heads loaded:', list(heads), flush=True)


def f1_of(keep_mask, iou, n_gt):
    """Official-shaped F1 over the kept frames of one fragment."""
    n_pred = int(keep_mask.sum())
    if n_pred == 0:
        return 0.0
    return 2.0 * float(iou[keep_mask].sum()) / (n_pred + n_gt)


rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
rng = np.random.default_rng(args.seed)
frag = []   # (src, per-head score arrays, iou, y)
for r in rows:
    src = r.get('src')
    if keep_src is not None and src not in keep_src:
        continue
    kfs = sorted((args.feat_root / r['video_id']).glob('*.npz'), key=lambda p: float(p.stem))
    if len(kfs) < 4:
        continue
    X = np.stack([np.load(k)['pooled'].astype(np.float32) for k in kfs])
    L = len(X)
    t0 = float(r['t0'])
    ivs = []
    for recs in sel.get(src, {}).values():
        for rec in recs:
            a_, b_ = float(rec['t0']), float(rec['t1'])
            if b_ > a_:
                ivs.append((a_ - t0, b_ - t0))
    stems = sorted(float(k.stem) for k in kfs)
    times = np.array([stems[i] - t0 for i in range(L)], np.float32)
    y = np.zeros(L, np.float32)
    # frame CENTRE inside the interval (half-frame tolerance at the edges)
    step_s = (times[-1] - times[0]) / max(len(times) - 1, 1) if len(times) > 1 else 1.0
    half = step_s * 0.5
    for a_, b_ in ivs:
        y[(times + half >= a_) & (times - half < b_)] = 1.0
    if y.sum() == 0 or y.sum() == L:
        continue
    # Box IoU per frame, and the piece that makes the mask matter: only frames
    # that are IN the ground truth contribute to iou_sum.  A kept frame outside
    # the GT costs n_pred without adding any numerator, which is the entire
    # reason a keep mask can raise F1.  Giving every frame the same IoU (the
    # first version of this script) made F1 depend only on how many frames are
    # kept, and every scorer tied at delta 0.
    iou = np.where(y > 0, 0.62, 0.0).astype(np.float32)
    xt = torch.from_numpy(X)[None]
    scores = {}
    for nm, m in heads.items():
        with torch.no_grad():
            scores[nm] = torch.sigmoid(m(xt))[0].numpy()
    if 'native' in scores and 'qv' in scores:
        zn = lambda v: (v - v.mean()) / (v.std() + 1e-6)
        scores['fusion'] = 0.5 * (zn(scores['native']) + zn(scores['qv']))
    frag.append({'src': src, 'y': y, 'iou': iou, 'scores': scores})

if not frag:
    raise SystemExit('no fragments scored')

n_gt_all = [int(f['y'].sum()) for f in frag]
pos_rate = float(np.mean([f['y'].mean() for f in frag]))
print(f'fragments={len(frag)} sources={len(set(f["src"] for f in frag))} pos_rate={pos_rate:.4f}', flush=True)

srcs = sorted({f['src'] for f in frag})
sidx = {s: i for i, s in enumerate(srcs)}
boot_idx = np.random.default_rng(args.seed).integers(0, len(srcs), size=(args.boot, len(srcs)))

out = {'n_fragments': len(frag), 'n_sources': len(srcs), 'pos_rate': round(pos_rate, 4),
       'variants': {}}
for nm in list(heads) + (['fusion'] if len(heads) > 1 else []):
    if nm not in frag[0]['scores']:
        continue
    variants = {}
    for keep in args.keeps:
        per_src = {}
        for f in frag:
            s = f['scores'][nm]
            k = max(1, int(round(keep * len(s))))
            idx = np.argsort(-s)[:k]
            m = np.zeros(len(s), bool); m[idx] = True
            v = f1_of(m, f['iou'], int(f['y'].sum()))
            # same-budget random control
            rm = np.zeros(len(s), bool); rm[rng.permutation(len(s))[:k]] = True
            rv = f1_of(rm, f['iou'], int(f['y'].sum()))
            per_src.setdefault(f['src'], []).append((v, rv))
        arr = np.array([np.mean([a for a, _ in vs]) for vs in per_src.values()])
        rarr = np.array([np.mean([b for _, b in vs]) for vs in per_src.values()])
        d = arr - rarr
        bd = d[boot_idx].mean(1)
        allm = np.ones(len(f['y']), bool)
        variants[f'{keep:.2f}'] = {
            'f1': round(float(arr.mean()), 4),
            'f1_random_same_budget': round(float(rarr.mean()), 4),
            'delta_vs_random': round(float(d.mean()), 4),
            'ci95_vs_random': [round(float(np.percentile(bd, 2.5)), 4),
                               round(float(np.percentile(bd, 97.5)), 4)],
        }
    # keep everything = F1 with all frames, the baseline every delta is against
    variants['all_frames'] = {'f1': round(float(np.mean(
        [f1_of(np.ones(len(f['y']), bool), f['iou'], int(f['y'].sum())) for f in frag])), 4)}
    out['variants'][nm] = variants

print(json.dumps(out, indent=1), flush=True)
if args.out:
    args.out.write_text(json.dumps(out, indent=1))