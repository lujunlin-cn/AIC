"""R8 A_PROTO: protocol-sensitivity audit on RetargetVid per-annotator labels.

Preregistered (reports/r8/preregistration.yaml): diagnostic only, no gate,
no training, no synthetic raters.  Uses the rv_dev pool (manifest gt is
(6,4) - six per-frame human boxes) and the FROZEN B3 checkpoint.

Reads (video-macro over rv_dev sources, B3 argmax over the 129 legal
windows, boxes rebuilt from W/H/ratio exactly as in training):
  U_util   mean over frames of u_util[j*],  u_util[j] = mean_a IoU(win_j, Y_a)
  U_coord  mean over frames of IoU(win_j*, mean-box(Y_1..Y_A))
  O_util   mean over frames of max_j u_util[j]              (utility oracle)
  O_coord  mean over frames of max_j IoU(win_j, mean-box)   (aggregate oracle)
  LOO_k    per annotator k: same read under u_{-k} = mean_{a!=k}
Sensitivity = U_util - U_coord, O_util - O_coord: how much the official
protocol fork (IoU-averaged vs coordinate-aggregated GT) would move the
numbers on data where both semantics are computable.

CPU only.  Output: JSON on stdout (caller redirects).
"""
import argparse, json, os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--samples', default='/data/aic/experiments_910a/LFM_V8/samples')
ap.add_argument('--manifest', default='/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl')
ap.add_argument('--b3-ckpt', default='/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt')
ap.add_argument('--batch', type=int, default=32)
args = ap.parse_args()

NC = 129
RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}


class Head(torch.nn.Module):
    """Identical to v8_s_train_multidata.Head (B3's architecture)."""

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


def candidate_boxes(W, H, ratio_wh, nc=NC):
    w, h, axis = geometry(W, H, ratio_wh)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    boxes = np.zeros((len(offs), 4), np.float32)
    for j, o in enumerate(offs):
        if axis == 0:
            boxes[j] = [o, 0, o + w, h]
        elif axis == 1:
            boxes[j] = [0, o, w, o + h]
        else:
            boxes[j] = [0, 0, W, H]
    return boxes


def macro(vals, vids):
    per = defaultdict(list)
    for x, v in zip(vals, vids):
        per[v].append(float(x))
    return round(float(np.mean([np.mean(per[v]) for v in sorted(per)])), 5)


def main():
    feats = np.load(f'{args.samples}/rv_dev_feat.npy', mmap_mode='r')
    vids = np.load(f'{args.samples}/rv_dev_vid.npy', allow_pickle=True)
    frames = np.load(f'{args.samples}/rv_dev_frame.npy', allow_pickle=True)
    man = {}
    for l in open(args.manifest):
        r = json.loads(l)
        if r.get('src') == 'rv_native':
            man[(r['vid'], int(r['frame']))] = r
    head = Head(feats.shape[-1], NC).float()
    head.load_state_dict(torch.load(args.b3_ckpt, map_location='cpu', weights_only=False)['state_dict'])
    head.eval()
    n = len(vids)
    recs = []   # (vid, u_util_j*, u_cm_j*, o_util, o_coord, [loo_k...])
    gt_dims = set()
    with torch.no_grad():
        for i in range(0, n, args.batch):
            sl = slice(i, min(i + args.batch, n))
            xs = torch.from_numpy(np.stack([np.asarray(feats[j], np.float32)
                                            for j in range(*sl.indices(n))]))
            ps = head(xs).numpy()
            for b, j in enumerate(range(i, min(i + args.batch, n))):
                m = man.get((str(vids[j]), int(frames[j])))
                if m is None:
                    continue
                gt = np.array(m['gt'], np.float32)
                gt_dims.add((gt.ndim, gt.shape[0] if gt.ndim == 2 else 1))
                if gt.ndim != 2 or gt.shape[0] < 2:
                    continue
                boxes = candidate_boxes(float(m['W']), float(m['H']), RATIOS[m['ratio']])
                u_util = iou(boxes[:, None, :], gt[None, :, :]).mean(1)
                gt_cm = gt.mean(0)
                u_cm = iou(boxes, gt_cm[None]).reshape(-1)
                jstar = int(np.argmax(ps[b]))
                loo = []
                for k in range(gt.shape[0]):
                    u_loo = np.delete(gt, k, 0)
                    loo.append(float(iou(boxes[:, None, :], u_loo[None, :, :]).mean(1)[jstar]))
                recs.append((str(vids[j]), float(u_util[jstar]), float(u_cm[jstar]),
                             float(u_util.max()), float(u_cm.max()), loo))
    v_list = [r[0] for r in recs]
    u_util_j = [r[1] for r in recs]
    u_coord_j = [r[2] for r in recs]
    o_util = [r[3] for r in recs]
    o_coord = [r[4] for r in recs]
    loo_vals = [x for r in recs for x in r[5]]
    out = {
        'protocol': 'R8 A_PROTO protocol-sensitivity audit on RV 6-annotator '
                    'labels; frozen B3 argmax-129; diagnostic, no gate',
        'rows_used': len(recs), 'gt_shapes_seen': sorted(gt_dims),
        'U_util_meanboxannotators': macro(u_util_j, v_list),
        'U_coord_coordinate_mean_box': macro(u_coord_j, v_list),
        'O_util_oracle': macro(o_util, v_list),
        'O_coord_oracle': macro(o_coord, v_list),
        'sensitivity_U_util_minus_U_coord': round(float(np.mean(u_util_j) - np.mean(u_coord_j)), 5),
        'sensitivity_O_util_minus_O_coord': round(float(np.mean(o_util) - np.mean(o_coord)), 5),
        'loo_read_mean': round(float(np.mean(loo_vals)), 5),
        'note': 'if sensitivity ~ 0, the two aggregation semantics do not move '
                'this table - the protocol fork matters less than the label '
                'noise itself; if large, official-protocol identity matters '
                'and stays unknowable without official labels'}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
