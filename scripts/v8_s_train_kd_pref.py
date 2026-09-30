"""V8 S-line KD-v2: teacher candidate-preference distillation as an INDEPENDENT
loss term (not the V7 label-mixture recipe that failed at 0.6465 < 0.6541).

For every training frame with a cached 32B teacher point, build the teacher
preference profile u_t = exp(-0.5((off-ot)/sigma)^2) over the same 129
candidates and add beta * huber(pred, u_t) to the GT loss.  GT terms stay
intact; the teacher acts as auxiliary supervision only.  Teacher use is
offline-training-only - deployment remains the student head.

Motivation (V8 blend fix): corrected frame alignment shows teacher points
beat the student head on DIAG (+0.0124) and the inference-time teacher prior
adds +0.0263 over the head - real complementarity worth distilling.
"""
import argparse, json, math, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--samples-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
ap.add_argument('--teacher-points', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points'))
ap.add_argument('--kd-weight', type=float, default=0.3)
ap.add_argument('--kd-sigma', type=float, default=1.0 / 16)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=1200)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--device', default='npu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: F401
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

NC = 129
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}


class Pool:
    """One cached pool, read through mmap when the .npy sidecars exist.

    Same V8 memory fix as v8_s_train_multidata: the original loader read
    z['feat'][i] per row, and np.load on .npz re-inflates the entire array on
    every key access (~15 GB x 25,878 for rv_train), then kept a second copy of
    the pool alive. mmap sidecars cost megabytes of RSS; the .npz fallback
    inflates once and materialises rows lazily.
    """

    def __init__(self, tag):
        self.tag = tag
        fp = args.samples_dir / f'{tag}_feat.npy'
        if fp.exists():
            self.feat = np.load(fp, mmap_mode='r')
            self.u = np.load(args.samples_dir / f'{tag}_u.npy')
            self.vid = np.load(args.samples_dir / f'{tag}_vid.npy', allow_pickle=True)
            self.frame = np.load(args.samples_dir / f'{tag}_frame.npy', allow_pickle=True)
            self.ratio = np.load(args.samples_dir / f'{tag}_ratio.npy', allow_pickle=True)
            self.n = len(self.u)
            return
        p = args.samples_dir / f'{tag}.npz'
        if not p.exists():
            self.n = 0
            return
        with np.load(p, allow_pickle=False) as z:
            self.feat, self.u = z['feat'], z['u']
            self.vid, self.frame, self.ratio = z['vid'], z['frame'], z['ratio']
        self.n = len(self.u)

    def __len__(self):
        return self.n

    def rows(self, idxs):
        return [{'feat': self.feat[i], 'u': self.u[i], 'vid': str(self.vid[i]),
                 'frame': int(self.frame[i]), 'ratio': str(self.ratio[i])} for i in idxs]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.rows(range(*i.indices(self.n)))
        if isinstance(i, (list, np.ndarray)):
            return self.rows([int(k) for k in i])
        return self.rows([int(i)])[0]


train = Pool('rv_train')
dev = Pool('rv_dev')
diag = Pool('rv_diag')

# teacher preference targets on the candidate grid
tp = {}
for r in train.rows(range(len(train))):
    tp.setdefault((r['vid'], r['ratio']), {})[r['frame']] = None
n_kd = 0
for (vid, rn), frames in sorted(tp.items()):
    pf = args.teacher_points / f'{vid}.json'
    if not pf.exists():
        continue
    t = json.loads(pf.read_text())
    if rn not in t.get('ratios', {}):
        continue
    pts = t['ratios'][rn]['points']
    st = t['ratios'][rn].get('status', ['ok'] * len(pts))
    kfs = t['keyframes']
    from aic.max_window_path import geometry as _g
    W, H = 640.0, 360.0
    rc = RATIOS[rn]
    w, h, axis = _g(W, H, rc)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    offs_n = offs / span if span > 0 else np.zeros(len(offs))
    dim = W if axis == 0 else H
    half = w / 2 if axis == 0 else h / 2
    comp = 0 if axis == 0 else 1
    for ki, kf in enumerate(kfs):
        if kf in frames and ki < len(pts) and st[ki] == 'ok':
            ot_n = float(np.clip((float(pts[ki][comp]) * dim - half) / span if span > 0 else 0.0, 0, 1))
            frames[kf] = np.exp(-0.5 * ((offs_n - ot_n) / args.kd_sigma) ** 2).astype(np.float32)
            n_kd += 1
print(f'KD targets available on {n_kd}/{len(train)} train frames', flush=True)


class Head(torch.nn.Module):
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

    def forward(self, x):
        h = self.proj(x)
        for i in range(2):
            h = self._ff(self._attn(h, i), i)
        return self.out(h).squeeze(-1)


D = train[0]['feat'].shape[1]
head = Head(D, NC).to(args.device).float()
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)


def batch_to(rs):
    x = torch.from_numpy(np.stack([r['feat'] for r in rs])).float().to(args.device)
    u = torch.from_numpy(np.stack([r['u'] for r in rs])).to(args.device)
    return x, u


def huber(pred, u, delta=0.25):
    ad = (pred - u).abs()
    return torch.where(ad <= delta, 0.5 * ad * ad, delta * (ad - delta / 2)).mean()


def loss_fn(x, u, rs):
    pred = head(x)
    l = huber(pred, u)
    hi, lo = u.argmax(1), u.argmin(1)
    l = l + 0.3 * torch.nn.functional.softplus(
        -(pred[torch.arange(len(u)), hi] - pred[torch.arange(len(u)), lo])).mean()
    if args.kd_weight > 0:
        idx = [i for i, r in enumerate(rs) if tp.get((r['vid'], r['ratio']), {}).get(r['frame']) is not None]
        if idx:
            ut = np.stack([tp[(rs[i]['vid'], rs[i]['ratio'])][rs[i]['frame']] for i in idx])
            ut_t = torch.from_numpy(ut).to(args.device)
            l = l + args.kd_weight * huber(pred[idx], ut_t)
    return l, pred


def eval_rows(rows, ret=False):
    head.eval()
    per = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            x, u = batch_to(chunk)
            pred = head(x)
            for c, r in enumerate(chunk):
                j = int(pred[c].argmax())
                per.append({'vid': r['vid'], 'frame': r['frame'], 'ratio': r['ratio'],
                            'iou': float(r['u'][j]), 'center': float(r['u'][len(r['u']) // 2])})
    head.train()
    if ret:
        return per
    vids = sorted({p['vid'] for p in per})
    d = [np.mean([p['iou'] - p['center'] for p in per if p['vid'] == v]) for v in vids]
    return float(np.mean([p['iou'] for p in per])), float(np.mean(d))


hist, best = [], (-1.0, None)
t0 = time.time()
for step in range(args.steps):
    rs = train.rows(rng.integers(0, len(train), args.batch))
    x, u = batch_to(rs)
    loss, _ = loss_fn(x, u, rs)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    sched.step()
    if (step + 1) % 60 == 0:
        iou_dev, d_center = eval_rows(dev)
        hist.append({'step': step + 1, 'loss': float(loss), 'dev_iou': iou_dev, 'dev_d_center': d_center})
        print(f'STEP {step + 1} loss={float(loss):.4f} dev_iou={iou_dev:.4f} d_center={d_center:+.4f}', flush=True)
        if iou_dev > best[0]:
            best = (iou_dev, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})

args.output_dir.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': best[1], 'config': {'d': D, 'nc': NC, 'seed': args.seed,
                                              'kd_weight': args.kd_weight, 'kd_sigma': args.kd_sigma}},
           args.output_dir / 'head_s.pt')
head.load_state_dict(best[1])
confirm = Pool('live_confirmation')
for name, pool in (('rv_dev', dev), ('rv_diag', diag), ('live_confirm', confirm)):
    per = eval_rows(pool, ret=True)
    with open(args.output_dir / f'per_{name}.jsonl', 'w') as fo:
        for p in per:
            fo.write(json.dumps(p) + '\n')
summ = {'seed': args.seed, 'kd_weight': args.kd_weight, 'steps': args.steps, 'train_n': len(train),
        'kd_frames': n_kd, 'best_dev_iou': best[0], 'dev_hist': hist,
        'wall_s': round(time.time() - t0, 1)}
(args.output_dir / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps({k: summ[k] for k in ('seed', 'kd_weight', 'best_dev_iou', 'wall_s')}), flush=True)
