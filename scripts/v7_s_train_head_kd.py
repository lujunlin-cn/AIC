"""V7 S-line: ratio-conditioned candidate-utility head over frozen LFM vision tokens.

For each (keyframe, target ratio): build 129 equally-spaced legal windows along
the free axis (includes the centre), pool window-in / window-out token means
from the cached 2D grid, and regress the 6-annotator mean IoU with a small
MLP + candidate-set transformer, plus a pairwise ranking term.

Splits (exposure ledger v7):
  S_TRAIN  RetargetVid 001-030 + 621-700   (legacy dev/confirm sources demoted to train)
  S_DEV    RetargetVid 601-620             (model selection only, no training)
  DIAG     RetargetVid 031-100             (legacy dev2; NEVER used for train/select here;
                                            paired diagnostic vs the raw-LFM row of the
                                            previous round, first contact for this head)
GT-only by construction: no teacher output enters features, labels or candidates.
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/feats'))
ap.add_argument('--ann', type=Path, default=Path('/data/aic/experiments_910a/LFM450_EVAL_V1/annotations'))
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=1200)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--n-cand', type=int, default=129)
ap.add_argument('--eval-every', type=int, default=60)
ap.add_argument('--device', default='npu')
ap.add_argument('--output-dir', type=Path, required=True)
ap.add_argument('--kd-weight', type=float, default=0.0, help='>0 enables teacher-point KD (C4 arm)')
ap.add_argument('--kd-point-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points'))
ap.add_argument('--kd-sigma-frac', type=float, default=1 / 16.0)
args = ap.parse_args()

import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: F401
    args.device = 'npu'
rng = np.random.default_rng(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

ALL_VIDS = sorted({p.name.split('_')[0] for p in args.cache.glob('*_1-3.npz')})
S_TRAIN = [v for v in ALL_VIDS if v <= '030' or '621' <= v <= '700']
S_DEV = [v for v in ALL_VIDS if '601' <= v <= '620']
DIAG = [v for v in ALL_VIDS if '031' <= v <= '100']
assert len(S_TRAIN) == 110 and len(S_DEV) == 20 and len(DIAG) == 70, (len(S_TRAIN), len(S_DEV), len(DIAG))
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}
NC = args.n_cand


def load_gt(vid, r):
    return np.maximum(np.stack([np.loadtxt(args.ann / f'annotator_{i}' / f'{vid}_{r}.txt',
                                           delimiter=',') for i in range(1, 7)]), 0)


PT_CACHE = {}


def teacher_offset(vid, r, kf):
    '''Free-axis offset of the 32B teacher point for (vid, ratio, keyframe), or None.'''
    key = (vid, r)
    if key not in PT_CACHE:
        f = args.kd_point_dir / f'{vid}.json'
        PT_CACHE[key] = json.loads(f.read_text())['ratios'][r]['points'] if f.exists() else None
    pts = PT_CACHE[key]
    if pts is None or kf >= len(pts) or pts[kf] is None:
        return None
    cx, cy, unc = pts[kf]
    if unc:
        return None
    W, H = 640.0, 360.0
    w, h, axis = geometry(W, H, RATIOS[r])
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    ot = (cx * W - w / 2) if axis == 0 else ((cy * H - h / 2) if axis == 1 else 0.0)
    return min(max(ot, 0.0), span) if span > 0 else None


def build_sample(vid, r, kf):
    """Returns dict with candidate windows, features aggregated from the grid, and labels."""
    f = np.load(args.feat_root / vid / f'{kf}.npz')
    grid = f['grid'].astype(np.float32)  # (fh,fw,768)
    fh, fw, D = grid.shape
    W, H = 640.0, 360.0
    w, h, axis = geometry(W, H, RATIOS[r])
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
    n_tok = fh * fw
    px_per = np.array([W / fw, H / fh])
    m = np.zeros((len(offs), n_tok), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win_px):
        cx1, cx2 = int(math.floor(x1 / px_per[0])), int(math.ceil(x2 / px_per[0]))
        cy1, cy2 = int(math.floor(y1 / px_per[1])), int(math.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        gy, gx = np.mgrid[0:fh, 0:fw]
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    win = (m @ flat) / ms
    out = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    pos = (offs / span) if span > 0 else np.zeros(len(offs))
    feat = np.concatenate([win, out, win - out, pos[:, None].astype(np.float32)], 1)
    gt = load_gt(vid, r)[:, kf]  # (6,4)
    u = iou(win_px[:, None, :], gt[None, :, :]).mean(1)  # (NC,)
    ko = teacher_offset(vid, r, kf) if args.kd_weight > 0 else None
    return {'vid': vid, 'kf': kf, 'r': r, 'feat': feat.astype(np.float16), 'u': u.astype(np.float32),
            'axis': axis, 'win_px': win_px.astype(np.float32),
            'kd': (ko, float((offs[-1] - offs[0]) if len(offs) > 1 else 0.0)) if ko is not None else None}


class Head(torch.nn.Module):
    """MLP + 2 self-attention layers, eager attention (SDPA backend unsupported on ascend910)."""

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


def split_pool(vids):
    samples = []
    for vid in vids:
        kfs = sorted(int(p.stem) for p in (args.feat_root / vid).glob('*.npz'))
        for kf in kfs:
            for r in RATIOS:
                samples.append((vid, kf, r))
    return samples


print('DEBUG building train pool...', flush=True)
t0 = time.time()
train = [build_sample(v, r, kf) for (v, kf, r) in split_pool(S_TRAIN)]
dev = [build_sample(v, r, kf) for (v, kf, r) in split_pool(S_DEV)]
print(f'DEBUG pools train={len(train)} dev={len(dev)} ({time.time() - t0:.0f}s)', flush=True)

D = train[0]['feat'].shape[1]
head = Head(D, NC).to(args.device).float()
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)


def batch_to(rows):
    x = torch.from_numpy(np.stack([r['feat'] for r in rows])).float().to(args.device)
    u = torch.from_numpy(np.stack([r['u'] for r in rows])).to(args.device)
    return x, u


def huber_pair(x, u, rows=None):
    pred = head(x)
    l_h = torch.nn.functional.huber_loss(pred, u, delta=0.25)
    # pairwise on the largest gap within each sample
    hi = u.argmax(1)
    lo = u.argmin(1)
    l_p = torch.nn.functional.softplus(-(pred[torch.arange(len(u)), hi] - pred[torch.arange(len(u)), lo])).mean()
    l_kd = pred.new_zeros(())
    if rows is not None and args.kd_weight > 0:
        offs_all = x.new_tensor(0.0)
        for bi, row in enumerate(rows):
            if row['kd'] is None:
                continue
            ot, span = row['kd']
            if span <= 0:
                continue
            o = torch.linspace(0, span, pred.shape[1], device=pred.device)
            g = torch.softmax(torch.exp(-0.5 * ((o - ot) / (span * args.kd_sigma_frac)) ** 2) / args.kd_sigma_frac, dim=0)
            l_kd = l_kd - (g * torch.log_softmax(pred[bi], dim=0)).sum()
        if torch.is_tensor(l_kd) and l_kd.numel() > 0:
            l_kd = l_kd / max(sum(1 for r_ in rows if r_['kd'] is not None), 1)
    return l_h + 0.3 * l_p + args.kd_weight * l_kd, pred


def eval_iou(rows, ret_rows=False):
    head.eval()
    per = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            x, u = batch_to(chunk)
            pred = head(x)
            for c, r in enumerate(chunk):
                j = int(pred[c].argmax())
                per.append({'vid': r['vid'], 'kf': r['kf'], 'ratio': r['r'], 'iou': float(r['u'][j]),
                            'best': float(r['u'].max()), 'center': float(r['u'][len(r['u']) // 2])})
    head.train()
    if ret_rows:
        return per
    # per-source-video unit (mean of both ratios) paired vs center
    vids = sorted({p['vid'] for p in per})
    d = []
    for v in vids:
        pv = [p for p in per if p['vid'] == v]
        d.append(np.mean([p['iou'] - p['center'] for p in pv]))
    return float(np.mean([p['iou'] for p in per])), float(np.mean(d))


hist, best = [], (-1.0, None)
t0 = time.time()
for step in range(args.steps):
    rows = [train[i] for i in rng.integers(0, len(train), args.batch)]
    x, u = batch_to(rows)
    loss, _ = huber_pair(x, u, rows)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    sched.step()
    if (step + 1) % args.eval_every == 0:
        iou_dev, d_center = eval_iou(dev)
        hist.append({'step': step + 1, 'train_loss': float(loss), 'dev_iou': iou_dev, 'dev_minus_center': d_center})
        print(f'STEP {step + 1} loss={float(loss):.4f} dev_iou={iou_dev:.4f} d_center={d_center:+.4f}', flush=True)
        if iou_dev > best[0]:
            best = (iou_dev, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})

args.output_dir.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': best[1], 'step': hist[-1]['step'] if hist else 0, 'dev_iou': best[0],
            'config': {'d': D, 'nc': NC, 'seed': args.seed}}, args.output_dir / 'head_s.pt')

head.load_state_dict(best[1])
for name, rows in (('s_dev', dev), ('diag_031_100', [build_sample(v, r, kf) for (v, kf, r) in split_pool(DIAG)])):
    per = eval_iou(rows, ret_rows=True)
    with open(args.output_dir / f'per_{name}.jsonl', 'w') as f:
        for p in per:
            f.write(json.dumps(p) + '\n')

summ = {'seed': args.seed, 'steps': args.steps, 'batch': args.batch, 'lr': args.lr, 'n_cand': NC,
        'kd_weight': args.kd_weight,
        'train_n': len(train), 'dev_n': len(dev), 'best_dev_iou': best[0],
        'dev_hist': hist, 'wall_s': round(time.time() - t0, 1),
        'splits': {'S_TRAIN': S_TRAIN[:3] + ['...'] + S_TRAIN[-3:], 'S_DEV': S_DEV, 'DIAG': '031-100 (70)'}}
(args.output_dir / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps({k: summ[k] for k in ('seed', 'best_dev_iou', 'wall_s')}), flush=True)
