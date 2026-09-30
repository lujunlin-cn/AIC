"""V8 S-line: multi-geometry candidate-utility head training.

Same head architecture and budget as V7 (MLP 2305-512-128 + 2-layer eager
self-attention, 1,511,425 params) so results are directly comparable.  New in
V8: the training pool can mix
  rv_native  RetargetVid 640x360 landscape, ratios 1-3 / 3-1   (V7 setting)
  rv_rot     same sources rotated 90 cw -> 360x640 vertical, ratios swapped
             (training augmentation; GT rotated exactly, never used for eval)
  live_*     LIVE-YT-VC frozen release, 640x360, 9:16 target on x axis
Candidate windows are still the 129 equally spaced legal max-windows; GT u is
the mean IoU over annotators (6 for RV, 1 for LIVE).

Selection: rv_dev (native) and live_dev; reported as head-argmax IoU and the
delta vs the centre candidate (cross-dataset comparable).  rv_diag and
live_confirmation are never used for train or selection.
"""
import argparse, json, math, random, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
ap.add_argument('--samples-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
ap.add_argument('--sources', nargs='*', default=['rv_native', 'rv_rot', 'live_train'])
ap.add_argument('--feat-roots', nargs='*', default=[
    'rv_native=/data/aic/experiments_910a/LFM_V7/feats',
    'rv_rot=/data/aic/experiments_910a/LFM_V8/rv_feats_rot',
    'live=/data/aic/experiments_910a/LFM_V8/live_feats'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--n-cand', type=int, default=129)
ap.add_argument('--eval-every', type=int, default=100)
ap.add_argument('--device', default='npu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: F401
import numpy as np  # noqa: E402
rng = np.random.default_rng(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

FEATS = {}
for spec in args.feat_roots:
    k, v = spec.split('=', 1)
    FEATS[k] = Path(v)
RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}
NC = args.n_cand
TRAIN_SPLITS = {'rv_native': ['rv_train'], 'rv_rot': ['rv_train'],
                'live_train': ['live_train'], 'live_all': ['live_train']}
DEV_SPLITS = {'rv_native': ['rv_dev'], 'rv_rot': [], 'live_train': [],
              'live_all': ['live_dev']}
if 'live_train' in args.sources:
    args.sources = [s for s in args.sources if s != 'live_train'] + ['live_all']
    args.sources = list(dict.fromkeys(args.sources))


def feat_path(src, vid, kf):
    root = FEATS['live'] if src.startswith('live') else FEATS[src]
    return root / vid / f'{kf}.npz'


rows = [json.loads(l) for l in open(args.manifest)]  # metadata only; feats come from cache


def load_np(tag):
    p = args.samples_dir / f'{tag}.npz'
    if not p.exists():
        return []
    z = np.load(p, allow_pickle=False)
    return [{'feat': z['feat'][i], 'u': z['u'][i], 'vid': str(z['vid'][i]),
             'frame': int(z['frame'][i]), 'ratio': str(z['ratio'][i])} for i in range(len(z['feat']))]


want_train_tags, want_dev_tags = [], []
for s in args.sources:
    want_train_tags += TRAIN_SPLITS[s]
    want_dev_tags += DEV_SPLITS[s]
train = [r for tag in dict.fromkeys(want_train_tags) for r in load_np(tag)]
dev = [r for tag in dict.fromkeys(want_dev_tags) for r in load_np(tag)]
diag = load_np('rv_diag')
print(f'cached pools train={len(train)} dev={len(dev)} diag={len(diag)}', flush=True)


def build_sample(row):
    f = np.load(feat_path(row['src'], row['vid'], row['frame']))
    grid = f['grid'].astype(np.float32)
    fh, fw, D = grid.shape
    W, H = float(row['W']), float(row['H'])
    ratio = RATIOS[row['ratio']]
    w, h, axis = geometry(W, H, ratio)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    win_px = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win_px[j] = [o, 0, o + w, h]
        elif axis == 1:
            win_px[j] = [0, o, w, o + h]
        else:
            win_px[j] = [0, 0, W, H]
    flat = grid.reshape(-1, D)
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(offs), fh * fw), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win_px):
        cx1, cx2 = int(math.floor(x1 / px_per[0])), int(math.ceil(x2 / px_per[0]))
        cy1, cy2 = int(math.floor(y1 / px_per[1])), int(math.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    win = (m @ flat) / ms
    out = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    pos = (offs / span if span > 0 else offs).astype(np.float32)
    feat = np.concatenate([win, out, win - out, pos[:, None]], 1)
    gt = np.array(row['gt'], dtype=np.float32)  # (A,4)
    u = iou(win_px[:, None, :], gt[None, :, :]).mean((1, 2)) if gt.ndim == 2 \
        else iou(win_px[:, None, :], gt[None]).mean(1)
    return {'feat': feat.astype(np.float16), 'u': u.astype(np.float32),
            'vid': row['vid'], 'frame': row['frame'], 'ratio': row['ratio']}


t0 = time.time()
print(f'sample cache load ({time.time() - t0:.1f}s)', flush=True)


class Head(torch.nn.Module):
    """MLP + 2 self-attention layers, eager attention (ascend910 has no SDPA)."""

    def __init__(self, d, nc, ch=128, heads=4, ff=256):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, ch))
        self.h = heads
        self.qkv = torch.nn.ModuleList([torch.nn.Linear(ch, 3 * ch) for _ in range(2)])
        self.proj_o = torch.nn.ModuleList([torch.nn.Linear(ch, ch) for _ in range(2)])
        self.ln1 = torch.nn.ModuleList([torch.nn.LayerNorm(ch) for _ in range(2)])
        self.ln2 = torch.nn.ModuleList([torch.nn.LayerNorm(ch) for _ in range(2)])
        self.ff1 = torch.nn.ModuleList([torch.nn.Linear(ch, ff) for _ in range(2)])
        self.ff2 = torch.nn.ModuleList([torch.nn.Linear(ff, ch) for _ in range(2)])
        self.drop = torch.nn.Dropout(0.1)
        self.out = torch.nn.Linear(ch, 1)

    def _attn(self, x, i):
        B, N, C = x.shape
        q, k, v = self.qkv[i](x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        a = torch.softmax(q @ k.transpose(-1, -2) / (C // self.h) ** 0.5, dim=-1)
        return self.ln1[i](x + self.drop(self.proj_o[i]((a @ v).transpose(1, 2).reshape(B, N, C))))

    def _ff(self, x, i):
        return self.ln2[i](x + self.drop(self.ff2[i](torch.nn.functional.gelu(self.ff1[i](x)))))

    def forward(self, x):  # x (B,NC,d)
        h = self.proj(x)
        for i in range(2):
            h = self._ff(self._attn(h, i), i)
        return self.out(h).squeeze(-1)  # (B,NC)


D = train[0]['feat'].shape[1]
head = Head(D, NC).to(args.device).float()
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)


def batch_to(rs):
    x = torch.from_numpy(np.stack([r['feat'] for r in rs])).float().to(args.device)
    u = torch.from_numpy(np.stack([r['u'] for r in rs])).to(args.device)
    return x, u


def huber_pair(x, u):
    pred = head(x)
    ad = (pred - u).abs()
    hub = torch.where(ad <= 0.25, 0.5 * ad * ad, 0.25 * (ad - 0.125)).mean()
    hi, lo = u.argmax(1), u.argmin(1)
    l_p = torch.nn.functional.softplus(-(pred[torch.arange(len(u)), hi] - pred[torch.arange(len(u)), lo])).mean()
    return hub + 0.3 * l_p, pred


def eval_rows(rs, ret=False):
    head.eval()
    per = []
    with torch.no_grad():
        for i in range(0, len(rs), 32):
            chunk = rs[i:i + 32]
            x, u = batch_to(chunk)
            pred = head(x)
            for c, r in enumerate(chunk):
                j = int(pred[c].argmax())
                per.append({'vid': r['vid'], 'frame': r['frame'], 'ratio': r['ratio'],
                            'iou': float(r['u'][j]), 'best': float(r['u'].max()),
                            'center': float(r['u'][len(r['u']) // 2])})
    head.train()
    if ret:
        return per
    vids = sorted({p['vid'] for p in per})
    d = [np.mean([p['iou'] - p['center'] for p in per if p['vid'] == v]) for v in vids]
    return float(np.mean([p['iou'] for p in per])), float(np.mean(d))


hist, best = [], (-1.0, None)
t0 = time.time()
for step in range(args.steps):
    rs = [train[i] for i in rng.integers(0, len(train), args.batch)]
    x, u = batch_to(rs)
    loss, _ = huber_pair(x, u)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    sched.step()
    if (step + 1) % args.eval_every == 0:
        m = {}
        for name, pool in (('rv_dev', [r for r in dev if r['vid'].isdigit()]),
                           ('live_dev', [r for r in dev if not r['vid'].isdigit()])):
            if pool:
                m[name] = eval_rows(pool)
        score = float(np.mean([v[1] for v in m.values()]))  # mean d_center across dev sets
        hist.append({'step': step + 1, 'loss': float(loss), 'score': score,
                     **{k: {'iou': v[0], 'd_center': v[1]} for k, v in m.items()}})
        print(f'STEP {step + 1} loss={float(loss):.4f} ' +
              ' '.join(f'{k}={v[0]:.4f}/{v[1]:+.4f}' for k, v in m.items()), flush=True)
        if score > best[0]:
            best = (score, {k: t.detach().cpu().clone() for k, t in head.state_dict().items()})

args.output_dir.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': best[1], 'config': {'d': D, 'nc': NC, 'seed': args.seed,
                                              'sources': args.sources, 'steps': args.steps}},
           args.output_dir / 'head_s.pt')
head.load_state_dict(best[1])
for name, pool in (('rv_dev', [r for r in dev if r['vid'].isdigit()]),
                   ('live_dev', [r for r in dev if not r['vid'].isdigit()]),
                   ('rv_diag', diag),
                   ('live_confirm', load_np('live_confirmation'))):
    if not pool:
        continue
    per = eval_rows(pool, ret=True)
    with open(args.output_dir / f'per_{name}.jsonl', 'w') as fo:
        for p in per:
            fo.write(json.dumps(p) + '\n')

summ = {'seed': args.seed, 'sources': args.sources, 'steps': args.steps, 'batch': args.batch,
        'lr': args.lr, 'n_cand': NC, 'train_n': len(train), 'best_score': best[0],
        'dev_hist': hist, 'wall_s': round(time.time() - t0, 1), 'head_params': 1511425}
(args.output_dir / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps({k: summ[k] for k in ('seed', 'sources', 'best_score', 'wall_s')}), flush=True)
