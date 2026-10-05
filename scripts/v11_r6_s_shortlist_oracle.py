"""R6 S-REREAD step 1: candidate shortlist + shortlist-oracle gate (R6 Q3).

Shortlist design (preregistered in reports/r6/preregistration.yaml):
  per frame: B3 top-3 candidates (by B3_s0 head score) UNION the 9
  equidistant grid points {0,16,32,...,128} of the 129-candidate slider,
  deduplicated -> <= 12 candidates per frame.

Gate: shortlist oracle regret vs the full 129-candidate oracle must be
<= 0.01 mean IoU (source-cluster CI on the dev pool).  If the gate fails,
EXTEND THE SHORTLIST FIRST - never attribute recall loss to the new
scorer (R6 Q3).

Pools: this script runs on the OLD dev124 sample cache (dev-exposed;
DEV ONLY - the val178 confirmation pool is never touched here).
A preregistered 24-video subset (seed 20261006) is reported alongside
the full dev124 read.

Also exports the crop-export manifest for the throughput check:
100 (frame, candidate) crops from the 24-subset, pixel boxes scaled from
manifest W,H into keyframe pixel space (T5 keyframes are 640 px wide).

CPU only.  Output: r6_spatial_shortlist_oracle.json (+ crop manifest csv)
"""
import argparse, csv, hashlib, json, os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--samples', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/samples/live_dev.npz'))
ap.add_argument('--manifest', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
ap.add_argument('--head', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt'))
ap.add_argument('--n-cand', type=int, default=129)
ap.add_argument('--grid-step', type=int, default=4,
                help='g33 = step 4 (AMENDED: v1 step 16 failed the regret gate)')
ap.add_argument('--topk', type=int, default=3)
ap.add_argument('--subset-n', type=int, default=24)
ap.add_argument('--subset-seed', type=int, default=20261006)
ap.add_argument('--n-crops', type=int, default=100)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_spatial_shortlist_oracle.json'))
args = ap.parse_args()

sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402

RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}
GRID = list(range(0, 129, args.grid_step))            # amended: step 4 (g33)


class Head(torch.nn.Module):
    """Copied verbatim from v8_s_train_multidata.py."""

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


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def boot_ci_vid(delta, vids, boot=2000, seed=99):
    vs = sorted(set(vids))
    idx = {v: [i for i, x in enumerate(vids) if x == v] for v in vs}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(boot):
        pick = rng.choice(len(vs), len(vs), replace=True)
        means.append(float(np.mean([delta[i] for p in pick for i in idx[vs[p]]])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)], round(float(np.mean(delta)), 5)


def win_boxes(W, H, ratio, nc):
    w, h, axis = geometry(W, H, ratio)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    boxes = []
    for o in offs:
        if axis == 0:
            boxes.append([o, 0, o + w, h])
        elif axis == 1:
            boxes.append([0, o, w, o + h])
        else:
            boxes.append([0, 0, W, H])
    return np.array(boxes, np.float32), axis


def main():
    d = np.load(args.samples, allow_pickle=True)
    feat, u = d['feat'], d['u'].astype(np.float32)
    vids_np = d['vid'].astype(str)
    frames_np = d['frame'].astype(int)
    ratios_np = d['ratio'].astype(str)
    N, NC = u.shape
    print(f'pool: {N} frames x {NC} candidates', flush=True)

    ck = torch.load(args.head, map_location='cpu', weights_only=False)
    head = Head(feat.shape[-1], NC, ch=ck.get('ch', 128)).float()
    head.load_state_dict(ck['state_dict']); head.eval()
    torch.set_grad_enabled(False)
    scores = np.concatenate([head(torch.from_numpy(feat[i:i + 64]).float()).numpy()
                             for i in range(0, N, 64)], 0)          # (N,NC)

    short_idx = np.array(sorted(set(GRID)))                        # grid points
    reg_full, regs = [], []
    n_short = []
    for i in range(N):
        top3 = np.argsort(-scores[i])[:args.topk]
        sl = sorted(set(top3.tolist()) | set(GRID))
        n_short.append(len(sl))
        reg_full.append(float(u[i].max() - u[i][sl].max()))
        regs.append(sl)
    reg_full = np.array(reg_full)
    vids_arr = vids_np.tolist()
    print(f'shortlist sizes: min {min(n_short)} p50 {int(np.median(n_short))} max {max(n_short)}',
          flush=True)

    rng = np.random.RandomState(args.subset_seed)
    all_vids = sorted(set(vids_arr))
    subset = set(rng.choice(all_vids, min(args.subset_n, len(all_vids)), replace=False).tolist())
    out = {'protocol': 'R6 Q3 shortlist = B3_top3 U 9-grid, dedup; regret gate '
                       '<= 0.01 mean IoU; pools: old dev124 (DEV ONLY, val178 '
                       'untouched); 24-video preregistered subset seed '
                       f'{args.subset_seed}',
           'head_sha16': sha16(args.head),
           'grid': GRID, 'grid_step': args.grid_step, 'topk': args.topk}
    for name, sel_vids in (('subset24', subset), ('dev124_full', set(all_vids))):
        mask = np.array([v in sel_vids for v in vids_arr])
        (ci, mean) = boot_ci_vid(reg_full[mask].tolist(), [vids_arr[i] for i in np.where(mask)[0]])
        out[name] = {'n_frames': int(mask.sum()), 'n_vids': len(sel_vids),
                     'regret_mean': mean, 'regret_ci95': ci,
                     'regret_p90': round(float(np.percentile(reg_full[mask], 90)), 5),
                     'gate_pass_le_001': bool(mean <= 0.01)}
        print(name, out[name], flush=True)

    # crop-export manifest for the throughput check (subset24 frames only)
    man = {}
    for l in args.manifest.read_text().splitlines():
        r = json.loads(l)
        if r.get('split') in ('live_dev', 'live_confirmation'):
            man[(r['vid'], r['frame'], r['ratio'])] = r
    crops, seen = [], set()
    for i in range(N):
        if vids_arr[i] not in subset:
            continue
        key = (vids_arr[i], int(frames_np[i]), str(ratios_np[i]))
        r = man.get(key)
        if r is None:
            continue
        boxes, _ = win_boxes(r['W'], r['H'], RATIOS[r['ratio']], NC)
        top3 = np.argsort(-scores[i])[:args.topk].tolist()
        sl = sorted(set(top3) | set(GRID))
        for c in sl:
            crops.append({'vid': vids_arr[i], 'frame': int(frames_np[i]),
                          'ratio': key[2], 'cand': c,
                          'box': [round(float(v), 2) for v in boxes[c]]})
        if len(crops) >= args.n_crops:
            break
    csvp = args.out.with_name(args.out.stem + '_crops.csv')
    with open(csvp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['vid', 'frame', 'ratio', 'cand', 'x1', 'y1', 'x2', 'y2'])
        for c in crops[:args.n_crops]:
            w.writerow([c['vid'], c['frame'], c['ratio'], c['cand']] + c['box'])
    out['crop_manifest'] = {'rows': min(len(crops), args.n_crops),
                            'note': 'boxes in manifest W,H pixel space; keyframes '
                                    'are 640px-wide -> scale by img_W/W at crop time',
                            'csv': csvp.as_posix()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
