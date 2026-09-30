"""V8 dense-observation strategy for the synthetic joint f1 (official shape).

For every DIAG frame (all frames, since rho=1.0 needs them all) run the
candidate head directly on the dense grid -> per-frame argmax window -> IoU
against the GT-interpolated RV box.  f1_A is then the official-shape
2*sum(IoU over GT frames) / (N_total + N_gt), identical to
scripts/v8_joint_eval2.py:f1_a - the 'dense' row of the strategy table.

Speed: candidate masks for a fixed (ratio, grid shape) are cached on the NPU;
each frame is one bmm-pool + a 1.5M head forward (~15 ms/frame/card).
Shard across 4 cards with --shard/--nshards; merge with --merge.
"""
import argparse, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--head', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/s_head_s1/head_s.pt'))
ap.add_argument('--dense-feats', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/diag_feats_dense'))
ap.add_argument('--saliency', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/dhf1k_saliency_value.jsonl'))
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=4)
ap.add_argument('--merge', action='store_true', help='merge shard outputs and emit f1_A table')
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import numpy as np  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from scripts.benchmark_spatial import iou  # noqa: E402

RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}
NC = 129

sal = {json.loads(l)['vid']: json.loads(l) for l in open(args.saliency)}
gt_map = {}
for l in open(args.manifest):
    r = json.loads(l)
    if r['split'] == 'rv_diag':
        gt_map[(r['vid'], r['ratio'], r['frame'])] = np.array(r['gt'], dtype=np.float32).mean(0)
vids = sorted(v for v in sal if '031' <= v <= '100')

if args.merge:
    rows = []
    for s in range(args.nshards):
        p = args.output.parent / f'{args.output.stem}_s{s}.jsonl'
        rows += [json.loads(l) for l in open(p)]
    out = {'rhos': {}}
    for rho in (1.0, 0.5, 0.3, 0.15):
        acc = []
        for r in rows:
            vals = np.array(sal[r['vid']]['values'])
            n_frame = len(vals)
            n_gt = max(1, int(round(n_frame * rho)))
            gt_frames = set(np.argsort(-vals)[:n_gt].tolist())
            for rn in RATIOS:  # rows are nested {ratio: {frame: iou}}
                iou_row = r['iou'].get(rn, {})
                s_ = float(sum(iou_row[str(f)] for f in gt_frames if str(f) in iou_row))
                acc.append(2 * s_ / (n_frame + n_gt) if (n_frame + n_gt) else 1.0)
        out['rhos'][str(rho)] = float(np.mean(acc))
        print('dense f1_A', rho, round(out['rhos'][str(rho)], 4), flush=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    print('WROTE', args.output)
    sys.exit(0)

import torch  # noqa: E402
import torch_npu  # noqa: F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from aic.max_window_path import geometry  # noqa: E402


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


ck = torch.load(args.head, map_location='cpu', weights_only=True)
head = Head(int(ck['config']['d']), NC).float().to('npu')
head.load_state_dict(ck['state_dict'])
head.eval()

mask_cache = {}


def masks_for(rn, fh, fw):
    key = (rn, fh, fw)
    if key in mask_cache:
        return mask_cache[key]
    W = H = None
    w, h, axis = geometry(640.0, 360.0, RATIOS[rn])
    span = (640.0 - w) if axis == 0 else ((360.0 - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    win = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win[j] = [o, 0, o + w, h]
        elif axis == 1:
            win[j] = [0, o, w, o + h]
        else:
            win[j] = [0, 0, 640.0, 360.0]
    px_per = np.array([640.0 / fw, 360.0 / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(win), fh * fw), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win):
        cx1, cx2 = int(math.floor(x1 / px_per[0])), int(math.ceil(x2 / px_per[0]))
        cy1, cy2 = int(math.floor(y1 / px_per[1])), int(math.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    pos = (offs / span if span > 0 else offs).astype(np.float32)
    mt = torch.from_numpy(m).to('npu')
    pt = torch.from_numpy(pos[:, None]).to('npu')
    mask_cache[key] = (mt, pt)
    return mt, pt


outf = args.output.parent / f'{args.output.stem}_s{args.shard}.jsonl'
done = set()
if outf.exists():
    for l in open(outf):
        try:
            done.add(json.loads(l)["vid"])
        except Exception:
            pass
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard and v not in done]
print(f'DEBUG shard {args.shard}/{args.nshards}: {len(mine)} vids ({len(done)} already done)', flush=True)
t0 = __import__('time').time()
n_f = 0
with open(outf, 'a') as fo:
    for vi, vid in enumerate(mine):
        vals = np.array(sal[vid]['values'])
        n_frame = len(vals)
        fdir = args.dense_feats / vid
        if not fdir.exists():
            continue
        # pre-extract annotated keyframes/GT per ratio once (GT interpolation below)
        kfs_by = {}
        for rn in RATIOS:
            kfs = sorted(k for (v_, r_, k) in gt_map if v_ == vid and r_ == rn)
            kfs_by[rn] = (kfs, np.stack([gt_map[(vid, rn, k)] for k in kfs]) if kfs else None)
        iou_rows = {}
        for rn in RATIOS:
            iou_rows[rn] = {}
            kfs, gts = kfs_by[rn]
            if not kfs:
                continue
            for f in range(n_frame):
                p = fdir / f'{f}.npz'
                if not p.exists():
                    continue
                z = np.load(p)
                grid = z['grid'].astype(np.float32)
                fh, fw, _ = grid.shape
                mt, pt = masks_for(rn, fh, fw)
                flat = torch.from_numpy(grid.reshape(-1, grid.shape[-1])).to('npu')
                ms = mt.sum(1, keepdims=True)
                winp = (mt @ flat) / ms
                outp = ((1 - mt) @ flat) / (1 - mt).sum(1, keepdims=True).clamp(min=1)
                feat = torch.cat([winp, outp, winp - outp, pt], 1).unsqueeze(0)
                with torch.no_grad():
                    u = head(feat)[0].cpu().numpy()
                w, h, axis = geometry(640.0, 360.0, RATIOS[rn])
                span = (640.0 - w) if axis == 0 else ((360.0 - h) if axis == 1 else 0.0)
                offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
                j = int(u.argmax())
                o = offs[j]
                win_j = ([o, 0, o + w, h] if axis == 0 else
                         ([0, o, w, o + h] if axis == 1 else [0, 0, 640.0, 360.0]))
                gt = gt_map.get((vid, rn, f))
                if gt is None:
                    i = int(np.searchsorted(kfs, f))
                    lo, hi = kfs[max(i - 1, 0)], kfs[min(i, len(kfs) - 1)]
                    g0, g1 = gts[max(i - 1, 0)], gts[min(i, len(kfs) - 1)]
                    gt = g0 if hi == lo else g0 + (g1 - g0) * (f - lo) / (hi - lo)
                iou_rows[rn][f] = float(iou(np.asarray(win_j)[None], gt[None])[0])
                n_f += 1
            iou_rows[rn] = {str(k): v for k, v in iou_rows[rn].items()}
        fo.write(json.dumps({'vid': vid, 'iou': iou_rows}) + '\n')
        if vi % 5 == 0:
            print(f'DEBUG {vid} ({vi}/{len(mine)}) frames={n_f} {__import__("time").time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
