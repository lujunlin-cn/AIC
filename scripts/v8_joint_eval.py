"""V8: synthetic joint evaluation on the RetargetVid DIAG sources (031-100).

The official joint metric is f1 = 2*sum_iou / (n_pred + n_gt) per video
(exact frame match, macro average).  DIAG has no official-style temporal GT,
so we *synthesize* it honestly and label it as a proxy:
  GT frames  = top-rho fraction of frames by DHF1K human saliency value
               (source-level separated: never trained on, never selected on)
  GT boxes   = RV 6-annotator mean crop at those frames
  GT density = rho swept over {1.0, 0.5, 0.3, 0.15} for sensitivity
Strategies compared under identical spatial head:
  kf      1s keyframe grid only (V7 deployment)
  interp  keyframe windows linearly interpolated to every frame (teacher
          INTERP mechanism ported to the student chain; no new inference)
  dense   head run on every frame (dense observation)
All numbers are proxy scores, not official or predicted official scores.
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--head', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/s_head_s1/head_s.pt'))
ap.add_argument('--feats', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/diag_feats_dense'))
ap.add_argument('--saliency', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/dhf1k_saliency_value.jsonl'))
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
ap.add_argument('--rhos', type=float, nargs='*', default=[1.0, 0.5, 0.3, 0.15])
ap.add_argument('--kf-step', type=int, default=30)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

NC = 129


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
    return win, offs, axis, span, w, h


def feat_u(head, grid, W, H, ratio):
    flat = grid.reshape(-1, grid.shape[-1]).astype(np.float32)
    fh, fw, D = grid.shape
    win, offs, axis, span, w, h = cand_windows(W, H, ratio)
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(offs), fh * fw), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win):
        cx1, cx2 = int(math.floor(x1 / px_per[0])), int(math.ceil(x2 / px_per[0]))
        cy1, cy2 = int(math.floor(y1 / px_per[1])), int(math.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    winp = (m @ flat) / ms
    outp = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    pos = (offs / span if span > 0 else offs).astype(np.float32)
    feat = torch.from_numpy(np.concatenate([winp, outp, winp - outp, pos[:, None]], 1)).unsqueeze(0)
    with torch.no_grad():
        u = head(feat)[0].numpy()
    return u, win


sal = {json.loads(l)['vid']: json.loads(l) for l in open(args.saliency)}
gt_map = {}
for l in open(args.manifest):
    r = json.loads(l)
    if r['split'] == 'rv_diag':
        gt_map[(r['vid'], r['ratio'], r['frame'])] = np.array(r['gt'], dtype=np.float32)

ck = torch.load(args.head, map_location='cpu', weights_only=True)
head = Head(int(ck['config']['d']), NC)
head.load_state_dict(ck['state_dict'])
head.eval()

RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}
vids = sorted(v for v in sal if '031' <= v <= '100')
print(f'{len(vids)} diag videos', flush=True)

# per (vid, ratio): per-frame predicted windows for each strategy + IoU vs RV GT
res = {}
for vi, vid in enumerate(vids):
    vals = np.array(sal[vid]['values'])
    n_frame = len(vals)
    fdir = args.feats / vid
    per_ratio = {}
    for rn, rc in RATIOS.items():
        frames = sorted(int(p.stem) for p in fdir.glob('*.npz'))
        if len(frames) < n_frame * 0.9:
            continue
        kfs = list(range(0, max(frames) + 1, args.kf_step))
        kfs = [k for k in kfs if k in frames]
        pred_iou = np.full(n_frame, np.nan, dtype=np.float64)
        kf_windows = {}
        for kf in kfs:
            z = np.load(fdir / f'{kf}.npz')
            u, win = feat_u(head, z['grid'], 640.0, 360.0, rc)
            j = int(u.argmax())
            kf_windows[kf] = win[j]
            gt = gt_map.get((vid, rn, kf))
            if gt is not None:
                pred_iou[kf] = float(iou(win[j][None], gt.mean(0)[None]).mean())
        # interpolate window coords between keyframes -> per-frame windows
        for f in range(n_frame):
            if pred_iou[f] == pred_iou[f]:
                continue
            if f not in frames:
                continue
            lo = max(k for k in kfs if k <= f) if any(k <= f for k in kfs) else kfs[0]
            hi = min(k for k in kfs if k >= f) if any(k >= f for k in kfs) else kfs[-1]
            if hi == lo:
                w = kf_windows[lo]
            else:
                a, b = kf_windows[lo], kf_windows[hi]
                w = a + (b - a) * (f - lo) / (hi - lo)
            gt = gt_map.get((vid, rn, f))
            if gt is not None:
                pred_iou[f] = float(iou(w[None], gt.mean(0)[None]).mean())
        per_ratio[rn] = pred_iou
    res[vid] = per_ratio
    if vi % 10 == 0:
        print(f'{vid} done ({vi}/{len(vids)})', flush=True)


def f1_for(pred_iou, sub_frames, gt_frames):
    """f1 over one (vid,ratio): predicted frame set = sub_frames, iou looked up."""
    iou_sum = float(np.nansum([pred_iou[f] for f in gt_frames if f in sub_frames]))
    n_pred = len(sub_frames)
    n_gt = len(gt_frames)
    return 2 * iou_sum / (n_pred + n_gt) if (n_pred + n_gt) else 1.0


out = {'rhos': {}, 'note': 'synthetic proxy on DIAG; saliency-top frames as temporal GT; '
                           'kf = 1s grid, interp = linear window interpolation to every frame'}
for rho in args.rhos:
    per_vid = {'kf': [], 'interp': []}
    for vid in vids:
        vals = np.array(sal[vid]['values'])
        n_frame = len(vals)
        n_gt = max(1, int(round(n_frame * rho)))
        gt_frames = set(np.argsort(-vals)[:n_gt].tolist())
        for rn, pred_iou in res[vid].items():
            frames = np.flatnonzero(~np.isnan(pred_iou)).tolist()
            kf_set = [f for f in frames if f % args.kf_step == 0]
            per_vid['kf'].append(f1_for(pred_iou, kf_set, gt_frames))
            per_vid['interp'].append(f1_for(pred_iou, frames, gt_frames))
    out['rhos'][str(rho)] = {k: float(np.mean(v)) for k, v in per_vid.items()}
    print(rho, {k: round(float(np.mean(v)), 4) for k, v in per_vid.items()}, flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=1) + '\n')
print('WROTE', args.output)
