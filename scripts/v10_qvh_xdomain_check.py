"""Cross-domain check: does the QVHighlights temporal head transfer to PHD2?

The QV head is trained on 9,000 QV fragments (auxiliary domain: YouTube life
clips, 22-annotator saliency).  Whether it can drive the keep mask on the
official drop is not a question the QV val split can answer - QV val shares the
head's domain.  PHD2 fragments carry the same GIF-interval objective the
official drop is built from, so a PHD2 val score is the transfer number that
matters.

This builds the PHD2 temporal evaluation set straight from the fragment index:
for each fragment, the 8 keyframe pooled vectors in order, and the binary
in-interval label on that 1 s grid (a second is positive when it falls inside
any GIF interval).  It then scores with the QV head and with the PHD2-native
head and reports, per source video:

  ap          average precision of the frame scores against in-interval labels
  recall@50   fraction of positive seconds retained when keeping the top half
  prevalence  label base rate, for scale

A head that does not beat prevalence on PHD2 cannot be trusted to shrink
N_pred on the official drop.
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


def ap_of(scores, labels):
    s = np.asarray(scores, np.float64); y = np.asarray(labels, np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    o = np.argsort(-s); ys = y[o]
    cum = np.cumsum(ys)
    return float((cum / np.arange(1, len(ys) + 1) * ys).sum() / y.sum())


def recall_at(scores, labels, keep):
    s = np.asarray(scores); y = np.asarray(labels)
    if y.sum() == 0:
        return float('nan')
    k = max(1, int(round(keep * len(s))))
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / y.sum())


ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/lfm_feats'))
ap.add_argument('--skel-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/skel'))
ap.add_argument('--qv-ckpt', type=Path, default=Path('/data/aic/experiments_910a/QVH_V10/tcn_frag_sal/qvh_frag_tcn_s0.pt'))
ap.add_argument('--native-ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_soft/tcn_s0.pt'))
ap.add_argument('--sources', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/val_sources.json'))
ap.add_argument('--out', type=Path, default=None)
args = ap.parse_args()

keep_src = set(json.loads(args.sources.read_text())) if args.sources.exists() else None
qv, qv_ck = load_head(args.qv_ckpt)
heads = {'qv': qv}
if args.native_ckpt.exists():
    nh, nck = load_head(args.native_ckpt)
    heads['native'] = nh
else:
    nck = None
print('heads:', {k: ck.get('dils') for k, ck in
                (('qv', qv_ck), ('native', nck)) if ck is not None}, flush=True)

rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
sel_path = Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json')
sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
print(f'PHD2 selections: {len(sel)} source videos', flush=True)
per_src = {k: {} for k in heads}
base_rows = []
n_used = 0
for r in rows:
    src = r.get('src')
    if keep_src is not None and src not in keep_src:
        continue
    kfs = sorted((args.feat_root / r['video_id']).glob('*.npz'), key=lambda p: float(p.stem))
    if len(kfs) < 4:
        continue
    X = np.stack([np.load(k)['pooled'].astype(np.float32) for k in kfs])
    L = len(X)
    # in-interval label on the fragment-local 1 s grid
    t0 = float(r['t0']); dur = float(r['L'])
    # PHD2 highlight intervals live in selections/train.json keyed by source
    # video, with t0/t1 in SOURCE-clip seconds.  The fragment index carries no
    # intervals, so they are looked up here and shifted into fragment time.
    ivs = []
    # selections is {src: {clip_key: [{t0,t1,is_last,flag}, ...]}} - flatten
    for recs in sel.get(r['src'], {}).values():
        for rec in recs:
            a_, b_ = float(rec['t0']), float(rec['t1'])
            if b_ > a_:
                ivs.append((a_ - t0, b_ - t0))
    # keyframe file stems are absolute clip times; the fragment starts at t0
    stems = sorted(float(k.stem) for k in kfs)
    times = np.array([(stems[i] - t0) for i in range(L)], np.float32)
    y = np.zeros(L, np.float32)
    for a_, b_ in ivs:
        y[(times >= a_) & (times < b_)] = 1.0
    if y.sum() == 0 or y.sum() == L:
        continue
    xt = torch.from_numpy(X)[None]
    for name, m in heads.items():
        with torch.no_grad():
            s = torch.sigmoid(m(xt))[0].numpy()
        per_src[name].setdefault(src, []).append((ap_of(s, y), recall_at(s, y, 0.5)))
    base_rows.append((ap_of(y + np.random.default_rng(0).normal(0, 1e-6, len(y)).astype(np.float32), y), recall_at(y, y, 0.5)))
    n_used += 1

if n_used == 0:
    print('no fragments matched the source filter with in-interval labels', flush=True)
    raise SystemExit(1)

out = {'n_fragments': n_used, 'n_sources': len(per_src['qv']), 'heads': {}}
for name in heads:
    srcs = per_src[name]
    aps = np.array([np.mean([a for a, _ in v]) for v in srcs.values()])
    rc = np.array([np.mean([r for _, r in v]) for v in srcs.values()])
    out['heads'][name] = {'ap_mean': round(float(aps.mean()), 4),
                          'ap_ci95': [round(float(np.percentile(aps, 2.5)), 4),
                                      round(float(np.percentile(aps, 97.5)), 4)],
                          'recall_at_50': round(float(rc.mean()), 4)}
b = np.array([a for a, _ in base_rows]); br = np.array([r for _, r in base_rows])
out['prevalence_ap'] = round(float(b.mean()), 4)
out['prevalence_recall_at_50'] = round(float(br.mean()), 4)
for name, ck in (('qv', qv_ck), ('native', nck)):
    if ck and 'val_ap' in ck:
        out['heads'][name]['own_val_ap'] = round(float(ck['val_ap']), 4)
print(json.dumps(out, indent=1), flush=True)
if args.out:
    args.out.write_text(json.dumps(out, indent=1))