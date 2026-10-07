"""R11 S1 step 1 (910A host only): dump the frozen B3 score table.

Produces the npz contract consumed by `python -m aic.r11_trajectory`:
    pred (N,NC) f4, u (N,NC) f4, vid/ratio/pool (N,) <U, ord (N,) i8,
    pos (N,NC) f4 (normalised free-axis offsets rebuilt from manifest W/H).

Usage (paths follow the V8 layout):
    python3 scripts/r11_s1_dump_scores.py \
      --ckpt  /data/aic/experiments_910a/LFM_V8/<b3_arm>/best.pt \
      --samples-dir /data/aic/experiments_910a/LFM_V8/samples \
      --manifest    /data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl \
      --tags live_dev live_confirmation rv_dev rv_diag \
      --out   /data/aic/experiments_910a/LFM_V11/r11_s1_score_table.npz

CPU-only by design (~30k rows x 129 candidates, minutes); --device npu is
accepted for parity checks but must NOT run while R10 chains hold NPU memory.
Head definition is copied verbatim from scripts/v8_s_train_multidata.py --
keep the two in sync.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `aic`

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', type=Path, required=True)
ap.add_argument('--samples-dir', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--tags', nargs='+', required=True,
                help='cache tags == manifest split values, e.g. live_dev')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--device', default='cpu', choices=['cpu', 'npu'])
ap.add_argument('--batch', type=int, default=256)
args = ap.parse_args()

import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: F401

from aic.max_window_path import geometry  # noqa: E402

RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}


class Head(torch.nn.Module):
    """Verbatim copy of scripts/v8_s_train_multidata.py::Head (eager attn)."""

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
        return self.out(h).squeeze(-1)


def load_tag(tag):
    """V8 cache: .npy sidecars when complete, legacy single .npz otherwise."""
    side = {k: args.samples_dir / f'{tag}_{k}.npy' for k in ('feat', 'u', 'vid', 'frame', 'ratio')}
    if all(p.exists() for p in side.values()):
        return {'feat': np.load(side['feat'], mmap_mode='r'),
                'u': np.load(side['u']),
                'vid': np.load(side['vid'], allow_pickle=True),
                'frame': np.load(side['frame'], allow_pickle=True),
                'ratio': np.load(side['ratio'], allow_pickle=True)}
    p = args.samples_dir / f'{tag}.npz'
    if not p.exists():
        raise SystemExit(f'tag {tag!r}: incomplete sidecars and no {p.name} under {args.samples_dir}')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in ('feat', 'u', 'vid', 'frame', 'ratio')}


def row_positions(W, H, ratio_tag, nc):
    """Same normalised offsets the head saw as its `pos` feature."""
    w, h, axis = geometry(float(W), float(H), RATIOS[ratio_tag])
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    if span <= 0:
        return np.zeros(nc, dtype=np.float32)
    return (np.linspace(0.0, span, nc) / span).astype(np.float32)


# ---- manifest index: (split, vid, frame) -> (W, H) -------------------------
wh = {}
n_manifest = 0
with open(args.manifest) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_manifest += 1
        if 'W' not in r or 'H' not in r:
            raise SystemExit(f'manifest row lacks W/H; keys={sorted(r)}')
        wh[(str(r.get('split')), str(r['vid']), int(r['frame']))] = (float(r['W']), float(r['H']))
        wh[(str(r['vid']), int(r['frame']))] = (float(r['W']), float(r['H']))  # fallback key
print(f'manifest rows={n_manifest} keys={len(wh)}')

ckpt = torch.load(args.ckpt, map_location='cpu')
cfg = ckpt['config']
head = Head(cfg['d'], cfg['nc']).float()
head.load_state_dict(ckpt['state_dict'])
head.eval()
nc = int(cfg['nc'])
print(f'ckpt {args.ckpt.name}: d={cfg["d"]} nc={nc}')

out = {k: [] for k in ('pred', 'u', 'vid', 'ratio', 'pool', 'ord', 'pos')}
with torch.no_grad():
    for tag in args.tags:
        c = load_tag(tag)
        n = len(c['u'])
        pred = np.zeros((n, nc), dtype=np.float32)
        pos = np.zeros((n, nc), dtype=np.float32)
        for i in range(0, n, args.batch):
            x = torch.from_numpy(np.array(c['feat'][i:i + args.batch])).float()
            pred[i:i + args.batch] = head(x).numpy()
        for i in range(n):
            key = (str(c['vid'][i]), int(c['frame'][i]))
            W, H = wh.get((tag,) + key, wh.get(key))
            pos[i] = row_positions(W, H, str(c['ratio'][i]), nc)
        # ord: 0..T-1 per (vid, ratio) group, ordered by frame number
        groups = {}
        for i in range(n):
            groups.setdefault((str(c['vid'][i]), str(c['ratio'][i])), []).append(i)
        ordv = np.zeros(n, dtype=np.int64)
        for (v, _ra), idxs in groups.items():
            idxs.sort(key=lambda i: int(c['frame'][i]))
            for t, i in enumerate(idxs):
                ordv[i] = t
        out['pred'].append(pred)
        out['u'].append(c['u'].astype(np.float32))
        out['vid'].append(np.array([str(v) for v in c['vid']]))
        out['ratio'].append(np.array([str(r) for r in c['ratio']]))
        out['pool'].append(np.array([tag] * n))
        out['ord'].append(ordv)
        out['pos'].append(pos)
        print(f'{tag}: n={n} pos_missing={int((pos == 0).all(1).sum())}', flush=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out,
         pred=np.concatenate(out['pred']), u=np.concatenate(out['u']),
         vid=np.concatenate(out['vid']), ratio=np.concatenate(out['ratio']),
         pool=np.concatenate(out['pool']), ord=np.concatenate(out['ord']),
         pos=np.concatenate(out['pos']))
print(f'wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)')
