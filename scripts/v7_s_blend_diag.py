"""V7 S-line diagnostic: teacher-prior blend at inference (no training, no packaging).

DIAG 031-100 only.  u_final = u_head + lam * exp(-0.5((off - off_t)/sig)^2), sig=1/16.
Measures complementarity between cached 32B teacher points and the student head on
untouched sources.  Deployment stays student-only (blending would pull the 32B
teacher into the deployed parameter count) -> report-only diagnostic.
"""
import argparse, json, math
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--head', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/s_head_s1/head_s.pt'))
ap.add_argument('--feats', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/feats'))
ap.add_argument('--teacher', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points'))
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--ann', type=Path, default=Path('/data/aic/experiments_910a/LFM450_EVAL_V1/annotations'))
ap.add_argument('--lams', type=float, nargs='*', default=[0.2, 0.4, 0.8, 1.0])
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import torch  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

NC = 129
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}


class Head(torch.nn.Module):
    """Copy of the deployed head (keep in sync with v7_s_train_head.py)."""

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


def load_gt(ann, vid, r):
    return np.maximum(np.stack([np.loadtxt(ann / f'annotator_{k}' / f'{vid}_{r}.txt', delimiter=',')
                                for k in range(1, 7)]), 0)


def load_gt(ann, vid, r):
    return np.maximum(np.stack([np.loadtxt(ann / f'annotator_{k}' / f'{vid}_{r}.txt', delimiter=',')
                                for k in range(1, 7)]), 0)


def feat_and_u(head, grid, W, H, rc):
    """Rebuild the 129-candidate features from the raw grid and run the head."""
    flat = grid.reshape(-1, grid.shape[-1]).astype(np.float32)
    n_tok = flat.shape[0]
    w, h, axis = geometry(float(W), float(H), rc)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    fh, fw = grid.shape[0], grid.shape[1]
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(offs), n_tok), dtype=np.float32)
    for j, o in enumerate(offs):
        if axis == 0:
            x1, y1, x2, y2 = o, 0, o + w, h
        elif axis == 1:
            x1, y1, x2, y2 = 0, o, w, o + h
        else:
            x1, y1, x2, y2 = 0, 0, W, H
        cx1 = int(math.floor(x1 / px_per[0]))
        cx2 = max(int(math.ceil(x2 / px_per[0])), cx1 + 1)
        cy1 = int(math.floor(y1 / px_per[1]))
        cy2 = max(int(math.ceil(y2 / px_per[1])), cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = np.maximum(m.sum(1, keepdims=True), 1)
    with torch.no_grad():
        win = torch.from_numpy((m @ flat) / ms)
        outm = torch.from_numpy(((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1))
        pos = torch.from_numpy((offs / span if span > 0 else offs)[:, None].astype(np.float32))
        feat = torch.cat([win, outm, win - outm, pos], 1).unsqueeze(0)
        u = head(feat)[0].numpy()
    win_px = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win_px[j] = [o, 0, o + w, h]
        elif axis == 1:
            win_px[j] = [0, o, w, o + h]
        else:
            win_px[j] = [0, 0, W, H]
    return u, offs, win_px, axis, span, w, h, W, H


def main():
    ck = torch.load(args.head, map_location='cpu', weights_only=True)
    head = Head(int(ck['config']['d']), NC)
    head.load_state_dict(ck['state_dict'])
    head.eval()
    rows = []
    vids = sorted(p.stem for p in args.teacher.glob('*.json'))
    vids = [v for v in vids if '031' <= v <= '100']
    for vid in vids:
        t = json.loads((args.teacher / f'{vid}.json').read_text())
        kfs = t['keyframes']
        zc = np.load(args.cache / f'{vid}_1-3.npz')
        W, H = float(zc['W']), float(zc['H'])
        fdir = args.feats / vid
        per_ratio = {}
        for rn, rc in RATIOS.items():
            if rn not in t.get('ratios', {}):
                continue
            pts = t['ratios'][rn]['points']
            st = t['ratios'][rn].get('status', ['ok'] * len(pts))
            gt = load_gt(args.ann, vid, rn)
            per_ratio[rn] = []
            for ki, kf in enumerate(kfs):
                f = fdir / f'{kf}.npz'
                if ki >= len(pts) or st[ki] != 'ok' or not f.exists():
                    per_ratio[rn].append(None)
                    continue
                z = np.load(f)
                u, offs, win_px, axis, span, w, h, Wf, Hf = feat_and_u(head, z['grid'], W, H, rc)
                ious = iou(win_px[:, None, :], gt[:, ki][None, :, :]).mean(1) if gt.shape[1] > ki else None
                if ious is None:
                    per_ratio[rn].append(None)
                    continue
                offs_n = offs / span if span > 0 else np.zeros(len(offs))
                if span > 0:
                    comp = 0 if axis == 0 else 1
                    dim = Wf if axis == 0 else Hf
                    half = w / 2 if axis == 0 else h / 2
                    ot_n = (float(pts[ki][comp]) * dim - half) / span
                else:
                    ot_n = 0.0
                per_ratio[rn].append({'u': u.astype(np.float32), 'offs_n': offs_n.astype(np.float32),
                                      'ot_n': float(np.clip(ot_n, 0, 1)), 'ious': ious.astype(np.float32),
                                      'head_iou': float(ious[int(u.argmax())]),
                                      'center_iou': float(ious[NC // 2]),
                                      'full': span <= 0})
        rows.append({'vid': vid, 'per': per_ratio})
        print(vid, {k: len([x for x in v if x]) for k, v in per_ratio.items()}, flush=True)

    # per-(vid,ratio) then per-vid paired metrics
    def unit(lam):
        res = {}
        for rr in rows:
            for rn, frames in rr['per'].items():
                good = [f for f in frames if f]
                if not good:
                    continue
                hm, tm, bm, cm = [], [], [], []
                for f in good:
                    if f['full']:
                        hm.append(f['head_iou']); tm.append(f['head_iou']); bm.append(f['head_iou']); cm.append(f['center_iou'])
                        continue
                    prior = np.exp(-0.5 * ((f['offs_n'] - f['ot_n']) / 0.0625) ** 2)
                    jb = int((f['u'] + lam * prior).argmax())
                    jt = int(np.abs(f['offs_n'] - f['ot_n']).argmin())
                    hm.append(f['ious'][int(f['u'].argmax())]); tm.append(f['ious'][jt]); bm.append(f['ious'][jb]); cm.append(f['center_iou'])
                res[(rr['vid'], rn)] = (np.mean(hm), np.mean(tm), np.mean(bm), np.mean(cm))
        return res

    svs = lambda res: sorted({v for v, _ in res})
    def pair(res, ia, ib):
        vs = svs(res)
        d = np.array([np.mean([res[(v, rn)][ia] - res[(v, rn)][ib] for rn in RATIOS if (v, rn) in res])
                      for v in vs])
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        return {'n_src': len(d), 'delta': float(d.mean()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                'wins': int((d > 1e-9).sum()), 'losses': int((d < -1e-9).sum()),
                'mean_a': float(np.mean([np.mean([res[(v, rn)][ia] for rn in RATIOS if (v, rn) in res]) for v in vs])),
                'mean_b': float(np.mean([np.mean([res[(v, rn)][ib] for rn in RATIOS if (v, rn) in res]) for v in vs]))}

    rng = np.random.default_rng(0)
    out = {'n_frames': sum(len([f for f in fr if f]) for rr in rows for fr in rr['per'].values()), 'lams': {}}
    for lam in args.lams:
        res = unit(lam)
        out['lams'][str(lam)] = {'means': {n: float(np.mean([np.mean([res[(v, rn)][i] for rn in RATIOS if (v, rn) in res]) for v in svs(res)]))
                                           for n, i in (('head', 0), ('teacher', 1), ('blend', 2), ('center', 3))},
                                 'blend-head': pair(res, 2, 0), 'blend-teacher': pair(res, 2, 1)}
    out['teacher-head'] = pair(unit(0.0), 1, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
