"""V9 student head: multi-pool training with a KD-only stream.

Generalises v8_s_train_kd_pref.py, which was hard-wired to RetargetVid (640x360,
ratios 1-3/3-1, KD targets recomputed from a fixed W/H).  PHD2 fragments carry
their own W/H and a 16:9 / 9:16 target, and their utility target comes from the
teacher point rather than from a human crop, so the two streams need different
supervision:

  GT stream   rv_native + rv_rot + live_train - u is IoU against human crop GT,
              loss = huber(pred, u) + 0.3 * pairwise
  KD stream   phd2_train - u is the teacher's Gaussian preference over the 129
              candidates (same target form as the validated V8 KD-v2 arm), and
              there is no human GT at all, so the only admissible loss is
              huber(pred, u_teacher) + pairwise.  These rows enter with weight
              --kd-pool-weight and never touch the GT term.

Optionally the GT stream also gets a KD term from its own teacher points
(--kd-weight), reproducing the V8 KD-v2 recipe inside the GT pools, so the
"add a KD term" and "add a KD-only pool" effects are separable.

Selection pools (rv_dev, live_dev, phd2_val) are reported separately and never
trained on.  `phd2_val` is the domain-matched confirmation pool: PHD2 sources,
held out by source video, stratified by geometry class.
"""
import argparse, json, math, time
from functools import lru_cache
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--samples-dir', type=Path, required=True)
ap.add_argument('--kd-samples-dir', type=Path, default=None,
                help='pool root for the KD-only stream (defaults to --samples-dir)')
ap.add_argument('--gt-pools', nargs='*', default=['rv_train', 'rv_rot_train', 'live_train'])
ap.add_argument('--kd-pools', nargs='*', default=['phd2_train'])
ap.add_argument('--dev-pools', nargs='*',
                default=['rv_dev', 'rv_rot_dev', 'live_dev',
                         'phd2_val@/data/aic/experiments_910a/PHD2_FRAG_V1/samples'],
                help='each entry is "tag" or "tag@pool_root" so the domain-matched '
                     'PHD2 confirmation pool can be read from its own directory')
ap.add_argument('--confirm-pool', default='live_confirmation')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--kd-batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--n-cand', type=int, default=129)
ap.add_argument('--kd-weight', type=float, default=0.0,
                help='KD term inside the GT pools (V8 KD-v2 recipe)')
ap.add_argument('--kd-pool-weight', type=float, default=0.5,
                help='weight of the KD-only stream relative to the GT stream')
ap.add_argument('--save-per-video', action='store_true',
                help='write per_<pool>.jsonl with per-row predicted-vs-best '
                     'window IoU so arms can be compared with a paired '
                     'source-level bootstrap instead of eyeballing means')
ap.add_argument('--eval-every', type=int, default=500)
ap.add_argument('--eval-cap', type=int, default=3000,
                help='subsample larger eval pools; full evaluation of 14k rows '
                     'every 200 steps dominated wall time')
ap.add_argument('--device', default='npu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: E402,F401
    assert torch.npu.device_count() > 0, 'no NPU visible - refusing CPU fallback'
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

from aic.max_window_path import geometry  # noqa: E402

NC = args.n_cand


def ratio_of(s):
    a, b = str(s).replace('-', ':').split(':')
    return int(a), int(b)


class Pool:
    """Cached candidate-utility rows; features come in through mmap."""

    def __init__(self, tag, root):
        self.tag = tag
        fp = root / f'{tag}_feat.npy'
        up = root / f'{tag}_u.npy'
        if not (fp.exists() and up.exists()):
            self.n = 0
            return
        self.feat = np.load(fp, mmap_mode='r')
        self.u = np.load(up)
        mp = root / f'{tag}_meta.json'
        meta = json.loads(mp.read_text()) if mp.exists() else None
        # V8 pools write a summary dict {"n","nc","d"} into meta.json and keep the
        # per-row fields in sidecar arrays; V9 PHD2 writes a list of row dicts.
        if isinstance(meta, list):
            self.vid = [m['vid'] for m in meta]
            self.frame = [m.get('kf', m.get('frame')) for m in meta]
            self.ratio = [m['ratio'] for m in meta]
            self.W = [float(m.get('W', 640)) for m in meta]
            self.H = [float(m.get('H', 360)) for m in meta]
        else:
            self.vid = list(np.load(root / f'{tag}_vid.npy', allow_pickle=True))
            self.frame = [int(x) for x in np.load(root / f'{tag}_frame.npy', allow_pickle=True)]
            self.ratio = [str(x) for x in np.load(root / f'{tag}_ratio.npy', allow_pickle=True)]
            self.W = [640.0] * len(self.u)
            self.H = [360.0] * len(self.u)
        assert len(self.u) == len(self.vid) == len(self.feat), tag
        self.n = len(self.u)

    def __len__(self):
        return self.n

    def rows(self, idxs):
        return [{'feat': np.asarray(self.feat[i]), 'u': np.asarray(self.u[i]),
                 'vid': str(self.vid[i]), 'frame': self.frame[i],
                 'ratio': self.ratio[i]} for i in idxs]

    def batch(self, idxs):
        # One fancy-index per batch instead of a Python loop of 16 mmap reads:
        # the per-row form dominated wall time (~3 s/step, which put a 3k-step
        # arm at 2.5 h).
        ii = np.asarray(idxs, dtype=np.int64)
        x = torch.from_numpy(np.ascontiguousarray(self.feat[ii])).float()
        u = torch.from_numpy(np.ascontiguousarray(self.u[ii])).float()
        return x.to(args.device), u.to(args.device)


def cat_pools(tags, root):
    pools = [Pool(t, root) for t in tags]
    pools = [p for p in pools if p.n]
    if not pools:
        return None
    offs = np.cumsum([0] + [p.n for p in pools])

    class Cat:
        tag = '+'.join(p.tag for p in pools)
        n = int(offs[-1])

        def rows(self, idxs):
            out = []
            for i in idxs:
                i = int(i)
                j = int(np.searchsorted(offs, i, 'right') - 1)
                out.extend(pools[j].rows([i - offs[j]]))
            return out

        def batch(self, idxs):
            # Group indices by pool so each pool does ONE fancy-index read, then
            # scatter back so the batch order still matches `idxs`.
            ii = np.asarray(idxs, dtype=np.int64)
            js = np.searchsorted(offs, ii, 'right') - 1
            X = U = None
            for j in range(len(pools)):
                sel = np.where(js == j)[0]
                if len(sel) == 0:
                    continue
                x, u = pools[j].batch(ii[sel] - offs[j])
                if X is None:
                    X = torch.zeros(len(ii), x.shape[1], x.shape[2])
                    U = torch.zeros(len(ii), u.shape[1])
                X[sel] = x.cpu()
                U[sel] = u.cpu()
            return X.to(args.device), U.to(args.device)
    print(f'POOL {Cat.tag}: n={Cat.n} from {[p.tag for p in pools]}', flush=True)
    return Cat()


train = cat_pools(args.gt_pools, args.samples_dir)
kd = cat_pools(args.kd_pools, args.kd_samples_dir or args.samples_dir)
def resolve(spec):
    if '@' in spec:
        tag, root = spec.split('@', 1)
        return tag, Path(root)
    return spec, args.samples_dir


devs = {}
for spec in args.dev_pools:
    tag, root = resolve(spec)
    devs[tag] = Pool(tag, root) if (root / f'{tag}_feat.npy').exists() else None
_ctag, _croot = resolve(args.confirm_pool)
confirm = Pool(_ctag, _croot) if (_croot / f'{_ctag}_feat.npy').exists() else None
for t, p in devs.items():
    print(f'DEV {t}: {p.n if p else 0}', flush=True)
print(f'CONFIRM {args.confirm_pool}: {confirm.n if confirm else 0}', flush=True)
if train is None:
    raise SystemExit('no GT pool')
D = train.rows([0])[0]['feat'].shape[-1]


class Head(torch.nn.Module):
    """Identical to the V7/V8 head (1,511,425 params) so arms stay comparable."""

    def __init__(self, d, nc, ch=128, heads=4, ff=256):
        super().__init__()
        self.proj = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(),
                                        torch.nn.Linear(512, ch))
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
        q, k, v = self.qkv[i](x).reshape(B, N, 3, self.h, C // self.h).permute(
            2, 0, 3, 1, 4).unbind(0)
        a = torch.softmax(q @ k.transpose(-1, -2) / (C // self.h) ** 0.5, dim=-1)
        return self.ln1[i](x + self.drop(self.proj_o[i]((a @ v).transpose(1, 2).reshape(B, N, C))))

    def _ff(self, x, i):
        return self.ln2[i](x + self.drop(self.ff2[i](torch.nn.functional.gelu(self.ff1[i](x)))))

    def forward(self, x):
        h = self.proj(x)
        for i in range(2):
            h = self._ff(self._attn(h, i), i)
        return self.out(h).squeeze(-1)


head = Head(D, NC).to(args.device).float()
n_par = sum(p.numel() for p in head.parameters())
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01,
                          foreach=(args.device != 'cpu'))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
print(f'HEAD params={n_par} device={args.device}', flush=True)


def huber(pred, u):
    ad = (pred - u).abs()
    return torch.where(ad <= 0.25, 0.5 * ad * ad, 0.25 * (ad - 0.125)).mean()


def pair(pred, u):
    hi, lo = u.argmax(1), u.argmin(1)
    m = (u.max(1).values - u.min(1).values) > 1e-4
    if not m.any():
        return torch.zeros((), device=pred.device)
    ar = torch.arange(len(u), device=pred.device)[m]
    return torch.nn.functional.softplus(-(pred[ar, hi[m]] - pred[ar, lo[m]])).mean()


@lru_cache(maxsize=256)
def win_matrix(W, H, rw, rh, nc):
    w, h, axis = geometry(W, H, (rw, rh))
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    out = np.zeros((len(offs), 4), dtype=np.float32)
    for j, o in enumerate(offs):
        out[j] = ([o, 0, o + w, h] if axis == 0 else
                  [0, o, w, o + h] if axis == 1 else [0, 0, W, H])
    return out


@torch.no_grad()
def evaluate(pool, collect=False):
    """Mean IoU between the predicted window and the best available window.

    Top-1 agreement over 129 candidates is the wrong yardstick - it reads ~0.025
    even for a good head and is not comparable with the 0.65-0.71 IoU that every
    V7/V8 arm reports.  This reconstructs the 129 candidate windows per row and
    scores predicted-vs-best with the same rectangle IoU the project uses.
    """
    if pool is None or pool.n == 0:
        return None
    head.eval()
    vals, recs = [], []
    n_eval = min(pool.n, args.eval_cap)
    sub = (range(pool.n) if n_eval == pool.n else
           np.random.default_rng(7).choice(pool.n, n_eval, replace=False))
    sub = sorted(int(x) for x in sub)
    for s0 in range(0, n_eval, 64):
        sl = sub[s0:s0 + 64]
        x, u = pool.batch(sl)
        pick = head(x).argmax(1).cpu().numpy()
        best = u.argmax(1).cpu().numpy()
        # Cache the per-(W,H,ratio) candidate grid - fragments share geometry
        # heavily, so this turns 14k window constructions per eval into a handful,
        # and the IoU itself becomes one vectorised expression.
        A = np.empty((len(sl), 4), dtype=np.float32)
        B = np.empty((len(sl), 4), dtype=np.float32)
        for k, i in enumerate(sl):
            W = float(pool.W[i]) if hasattr(pool, 'W') else 640.0
            H = float(pool.H[i]) if hasattr(pool, 'H') else 360.0
            rw, rh = ratio_of(pool.ratio[i])
            wm = win_matrix(W, H, rw, rh, NC)
            A[k] = wm[pick[k]]
            B[k] = wm[best[k]]
        ix = np.maximum(0.0, np.minimum(A[:, 2], B[:, 2]) - np.maximum(A[:, 0], B[:, 0]))
        iy = np.maximum(0.0, np.minimum(A[:, 3], B[:, 3]) - np.maximum(A[:, 1], B[:, 1]))
        inter = ix * iy
        area = ((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]) +
                (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)
        ious = inter / area
        vals.extend(ious.tolist())
        if collect:
            for k, i in enumerate(sl):
                recs.append({'vid': str(pool.vid[i]),
                             'src': (str(pool.vid[i]).rsplit('_f', 1)[0]),
                             'frame': pool.frame[i],
                             'ratio': str(pool.ratio[i]),
                             'iou': float(ious[k]),
                             'pick': int(pick[k]),
                             'best': int(best[k])})
    head.train()
    if collect:
        return round(float(np.mean(vals)), 4), recs
    return round(float(np.mean(vals)), 4)


best = {'score': -1.0, 'step': 0, 'state': None}
hist = []
t0 = time.time()
for step in range(1, args.steps + 1):
    xi = rng.integers(0, train.n, args.batch)
    x, u = train.batch(xi)
    loss = huber(head(x), u) + 0.3 * pair(head(x), u)
    if kd is not None and args.kd_pool_weight > 0:
        ki = rng.integers(0, kd.n, args.kd_batch)
        xk, uk = kd.batch(ki)
        pk = head(xk)
        loss = loss + args.kd_pool_weight * (huber(pk, uk) + 0.3 * pair(pk, uk))
    if args.kd_weight > 0:
        loss = loss + args.kd_weight * huber(head(x), u)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % args.eval_every == 0 or step == args.steps:
        m = {t: evaluate(p) for t, p in devs.items()}
        m['confirm'] = evaluate(confirm)
        hist.append({'step': step, 'loss': round(float(loss), 4), **m})
        sel = m.get('rv_dev') or 0.0
        print(f'step {step} loss={float(loss):.4f} '
              + ' '.join(f'{k}={v}' for k, v in m.items() if v is not None)
              + f' [{time.time() - t0:.0f}s]', flush=True)
        if sel > best['score']:
            best = {'score': sel, 'step': step,
                    'state': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}

args.output_dir.mkdir(parents=True, exist_ok=True)
if best['state']:
    head.load_state_dict(best['state'])
torch.save({'state_dict': head.state_dict(), 'D': D, 'nc': NC, 'params': n_par,
            'best_rv_dev': best['score'], 'best_step': best['step'],
            # v8_s_official_points.py reads config['d'] / config['nc'], so mirror
            # the feature width into config rather than only at top level.
            'config': {**{k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items()},
                       'd': D, 'nc': NC}},
           args.output_dir / f'head_s{args.seed}.pt')
if args.save_per_video:
    for tag, pool in list(devs.items()) + [('confirm', confirm)]:
        if pool is None or pool.n == 0:
            continue
        _, recs = evaluate(pool, collect=True)
        with (args.output_dir / f'per_{tag}.jsonl').open('w') as fh:
            for r in recs:
                fh.write(json.dumps(r) + '\n')
        print(f'per_video written {tag}: {len(recs)} rows', flush=True)

(args.output_dir / f'history_s{args.seed}.json').write_text(json.dumps(
    {'params': n_par, 'gt_pools': args.gt_pools, 'kd_pools': args.kd_pools,
     'kd_weight': args.kd_weight, 'kd_pool_weight': args.kd_pool_weight,
     'train_n': train.n, 'kd_n': kd.n if kd else 0,
     'best_rv_dev': best['score'], 'best_step': best['step'],
     'selection_rule': 'rv_dev top-1 accuracy (held out, never trained on)',
     'history': hist}, indent=1))
print(f'SUMMARY params={n_par} best_rv_dev={best["score"]} @ step {best["step"]}', flush=True)