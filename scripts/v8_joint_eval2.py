"""V8 synthetic joint evaluation, v2 - OFFICIAL-SHAPE protocol.

Fixes two protocol defects of v8_joint_eval.py (v1):
  1. v1's "kf" strategy submitted only keyframes with frame % 30 == 0, but the
     d1 keyframe grid is shot-adaptive (frames like 073), so kf submitted ~1/30
     of the real keyframes -> its low score was a submission-density ceiling,
     not a hold-vs-interp comparison.
  2. v1 let n_pred vary with strategy; the official metric always submits every
     frame (B0-compatible full-frame JSONL), so n_pred = n_video_frames is
     fixed and only the per-frame window coords differ across strategies.

v2 protocol (per video, per ratio):
  GT frames = top-rho fraction by DHF1K saliency (rho swept, synthetic and
  labelled as a proxy - saliency is not the official highlight annotation)
  GT boxes  = RV 6-annotator mean box at keyframes, LINEARLY INTERPOLATED to
              non-keyframes (crop motion is smooth at 1 s granularity)
  f1_A      = 2*sum(IoU over GT frames) / (N_total + N_gt)   [official shape]
  strategies: hold   keyframe window copied forward (V7 deployment)
              interp keyframe window linearly interpolated (teacher INTERP port)
              dense  head run on every frame (needs --dense-feats, shard-able
                     separately in v8_dense_eval.py)
  f1_B      = keep-mask variant for temporal selection (direction 3): submit
              only the value-head top-m frames, n_pred = |keep|.

DIAG has no shot annotations, so spans are global (no shot reset); the
official-chain interp resets at cuts, this proxy ignores that.
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--head', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/s_head_s1/head_s.pt'))
ap.add_argument('--dense-feats', type=Path, default=None)
ap.add_argument('--value-ckpt', type=Path, default=None)
ap.add_argument('--saliency', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/dhf1k_saliency_value.jsonl'))
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
ap.add_argument('--rhos', type=float, nargs='*', default=[1.0, 0.5, 0.3, 0.15])
ap.add_argument('--keep-fracs', type=float, nargs='*', default=[1.0, 0.75, 0.5, 0.3, 0.15])
ap.add_argument('--kf-windows-cache', type=Path, default=None, help='json to cache/reuse keyframe head outputs')
ap.add_argument('--device', default='cpu')
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

NC = 129
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}


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


def cand_windows(W, H, ratio, n=NC):
    w, h, axis = geometry(W, H, ratio)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, n) if span > 0 else np.zeros(1)
    win = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win[j] = [o, 0, o + w, h]
        elif axis == 1:
            win[j] = [0, o, w, o + h]
        else:
            win[j] = [0, 0, W, H]
    return win


def head_windows(head, grid, W, H, ratio, dev):
    """run the candidate head on one keyframe grid -> (129,4) + scores (129,)"""
    flat = grid.reshape(-1, grid.shape[-1]).astype(np.float32)
    fh, fw, D = grid.shape
    win = cand_windows(W, H, ratio)
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(win), fh * fw), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win):
        cx1, cx2 = int(math.floor(x1 / px_per[0])), int(math.ceil(x2 / px_per[0]))
        cy1, cy2 = int(math.floor(y1 / px_per[1])), int(math.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    winp = (m @ flat) / ms
    outp = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    _, _, axis = geometry(W, H, ratio)
    span = (W - win[0][2]) if axis == 0 else ((H - win[0][3]) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    pos = (offs / span if span > 0 else offs).astype(np.float32)
    feat = torch.from_numpy(np.concatenate([winp, outp, winp - outp, pos[:, None]], 1)).unsqueeze(0).to(dev)
    with torch.no_grad():
        u = head(feat)[0].cpu().numpy()
    return win, u


sal = {json.loads(l)['vid']: json.loads(l) for l in open(args.saliency)}
gt_map = {}
for l in open(args.manifest):
    r = json.loads(l)
    if r['split'] == 'rv_diag':
        gt_map[(r['vid'], r['ratio'], r['frame'])] = np.array(r['gt'], dtype=np.float32).mean(0)

vids = sorted(v for v in sal if '031' <= v <= '100')

# ---- stage 1: keyframe windows (cached across strategies / heads)
cache = {}
if args.kf_windows_cache and args.kf_windows_cache.exists():
    cache = json.loads(args.kf_windows_cache.read_text())

ck = torch.load(args.head, map_location='cpu', weights_only=True)
head = Head(int(ck['config']['d']), NC).to(args.device)
head.load_state_dict(ck['state_dict'])
head.eval()

need = [v for v in vids if v not in cache]
for vi, vid in enumerate(need):
    fdir = Path('/data/aic/experiments_910a/LFM_V7/feats') / vid  # keyframe grids (1s, V7 extractor)
    per_ratio = {}
    for rn, rc in RATIOS.items():
        kfs = sorted(int(p.stem) for p in fdir.glob('*.npz'))
        wins, scores = {}, {}
        for kf in kfs:
            z = np.load(fdir / f'{kf}.npz')
            if 'grid' not in z:
                continue
            win, u = head_windows(head, z['grid'], 640.0, 360.0, rc, args.device)
            wins[kf], scores[kf] = win[int(u.argmax())].tolist(), float(u.max())
        if wins:
            per_ratio[rn] = {'kfs': sorted(wins), 'wins': [wins[k] for k in sorted(wins)],
                             'scores': [scores[k] for k in sorted(wins)]}
    cache[vid] = per_ratio
    if vi % 10 == 0:
        print(f'kf-head {vid} ({vi}/{len(need)})', flush=True)
if args.kf_windows_cache:
    args.kf_windows_cache.parent.mkdir(parents=True, exist_ok=True)
    args.kf_windows_cache.write_text(json.dumps(cache))


def expand_windows(kfs, wins, n_frame, mode):
    """per-frame windows for hold / interp on the global keyframe spine"""
    outw = np.zeros((n_frame, 4))
    for f in range(n_frame):
        if f <= kfs[0]:
            outw[f] = wins[0]
        elif f >= kfs[-1]:
            outw[f] = wins[-1]
        else:
            i = int(np.searchsorted(kfs, f))
            k0, k1 = kfs[i - 1], kfs[i]
            a, b = np.asarray(wins[i - 1]), np.asarray(wins[i])
            outw[f] = a if mode == 'hold' else a + (b - a) * (f - k0) / (k1 - k0)
    return outw


def gt_interp(kfs, gts, n_frame):
    outg = np.zeros((n_frame, 4))
    for f in range(n_frame):
        if f <= kfs[0]:
            outg[f] = gts[0]
        elif f >= kfs[-1]:
            outg[f] = gts[-1]
        else:
            i = int(np.searchsorted(kfs, f))
            a, b = gts[i - 1], gts[i]
            outg[f] = a + (b - a) * (f - kfs[i - 1]) / (kfs[i] - kfs[i - 1])
    return outg


# ---- stage 2: hold/interp f1_A (official shape)
res = {}
for vid in vids:
    vals = np.array(sal[vid]['values'])
    n_frame = len(vals)
    per_ratio = {}
    for rn, entry in cache.get(vid, {}).items():
        kfs = entry['kfs']
        wins = np.asarray(entry['wins'])
        gts = np.stack([gt_map[(vid, rn, k)] for k in kfs])
        g_all = gt_interp(kfs, gts, n_frame)
        iou_of = {}
        for mode in ('hold', 'interp'):
            w = expand_windows(kfs, wins, n_frame, mode)
            iou_of[mode] = iou(w, g_all).astype(np.float64)  # per-frame IoU
        per_ratio[rn] = {'n_frame': n_frame, 'iou': iou_of}
    res[vid] = per_ratio


def f1_a(iou_row, gt_frames, n_total):
    s = float(sum(iou_row[f] for f in gt_frames))
    return 2 * s / (n_total + len(gt_frames)) if (n_total + len(gt_frames)) else 1.0


out = {'f1_A': {}, 'note': 'v2 official-shape proxy: full-frame submission, n_pred=N_total; '
                           'GT boxes interpolated between keyframes; hold/interp on global spine'}
for rho in args.rhos:
    acc = {'hold': [], 'interp': []}
    for vid in vids:
        vals = np.array(sal[vid]['values'])
        n_gt = max(1, int(round(len(vals) * rho)))
        gt_frames = set(np.argsort(-vals)[:n_gt].tolist())
        for rn, d in res[vid].items():
            for mode in acc:
                acc[mode].append(f1_a(d['iou'][mode], gt_frames, d['n_frame']))
    out['f1_A'][str(rho)] = {m: float(np.mean(v)) for m, v in acc.items()}
    print(rho, out['f1_A'][str(rho)], flush=True)

# ---- stage 3: dense is computed by scripts/v8_dense_eval.py (shard-able across
# devices) with the same f1_a(); merged afterwards.

# ---- stage 4: f1_B keep-mask (temporal selection on the keyframe grid).
class Block(torch.nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.f = torch.nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.g = torch.nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.out = torch.nn.Conv1d(ch, ch, 1)
        self.ln = torch.nn.LayerNorm(ch)

    def forward(self, x):
        a = torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))
        return self.ln((self.out(a) + x).transpose(1, 2)).transpose(1, 2)


class TCN(torch.nn.Module):
    def __init__(self, d=768, ch=128):
        super().__init__()
        self.inp = torch.nn.Conv1d(d, ch, 1)
        self.blocks = torch.nn.ModuleList([Block(ch, 2 ** i) for i in range(6)])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2).squeeze(-1)


if args.value_ckpt:
    vck = torch.load(args.value_ckpt, map_location='cpu', weights_only=True)
    vh = TCN(d=vck['config']['d'], ch=vck['config']['ch'])
    vh.load_state_dict(vck['state_dict'])
    vh.eval()
    out['f1_B'] = {'keep_fracs': args.keep_fracs, 'rhos': {},
                   'note': 'same denominator as f1_A (full-frame submission, n_pred=N_total). '
                           'A frame is re-predicted by the head only if it is in the value-head '
                           'top-m keyframes; other frames hold the last kept window. Measures the '
                           'f1 cost of sparse observation, not a change of submission density.'}
    for rho in args.rhos:
        acc = {str(k): [] for k in args.keep_fracs}
        for vid in vids:
            vals = np.array(sal[vid]['values'])
            n_frame = len(vals)
            n_gt = max(1, int(round(n_frame * rho)))
            gt_frames = set(np.argsort(-vals)[:n_gt].tolist())
            fdir = Path('/data/aic/experiments_910a/LFM_V7/feats') / vid
            kf_seq = sorted(int(p.stem) for p in fdir.glob('*.npz'))
            kf_seq = [k for k in kf_seq if k < n_frame]
            if len(kf_seq) < 4:
                continue
            x = np.stack([np.load(fdir / f'{k}.npz')['pooled'].astype(np.float32) for k in kf_seq])
            with torch.no_grad():
                vs = vh(torch.from_numpy(x[None]))[0].numpy()
            for keep in args.keep_fracs:
                m = max(1, int(round(len(kf_seq) * keep)))
                keep_set = {kf_seq[i] for i in np.argsort(-vs)[:m]}
                for rn, d in res[vid].items():
                    kfsr = cache[vid][rn]['kfs']
                    g_all = gt_interp(kfsr, np.stack([gt_map[(vid, rn, k)] for k in kfsr]), n_frame)
                    w = expand_windows(kfsr, np.asarray(cache[vid][rn]['wins']), n_frame, 'hold')
                    # sparse observation: only kept keyframes refresh the window,
                    # every other frame keeps the last kept one
                    ks = sorted(keep_set)
                    if not ks:
                        continue
                    w_sparse = w.copy()
                    prev = w[0]
                    ki = 0
                    for f in range(n_frame):
                        while ki < len(ks) and ks[ki] <= f:
                            prev = w[ks[ki]]
                            ki += 1
                        w_sparse[f] = prev
                    iou_row = iou(w_sparse, g_all).astype(np.float64)
                    acc[str(keep)].append(f1_a(iou_row, gt_frames, n_frame))
        out['f1_B']['rhos'][str(rho)] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
        print('f1_B', rho, out['f1_B']['rhos'][str(rho)], flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=1) + '\n')
print('WROTE', args.output)
