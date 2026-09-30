"""V7 T-line: T1-style TCN retrained on native LFM pooled features (TVSum).

Supervision: per-frame 20-rater mean (proxy-v2) via masked Huber + intra-video
pairwise ranking.  No Mr.HiSum feature-space weights are loaded; the architecture
is inherited from T1 (6 dilated residual blocks, all-resolution) but every weight
is trained from scratch in the LFM domain.

Eval: local dev 16 (splits/local_protocol_v1.json), 16 equal segments, segment
score = model score at the segment centre second; ranking_report + a fresh
seeded RANDOM/UNIFORM baseline per video for paired comparison.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--feat-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/tfeats_tvsum'))
ap.add_argument('--protocol', type=Path, default=Path('splits/local_protocol_v1.json'))
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=1200)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--device', default='npu')
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()

import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: F401
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.temporal_metrics import ranking_report  # noqa: E402

torch.manual_seed(args.seed)
np.random.seed(args.seed)

PROTO = json.loads(args.protocol.read_text())
DEV16 = list(PROTO['dev_video_ids'])
VIDS = sorted(p.stem for p in args.feat_dir.glob('*.npz'))
TRAIN = [v for v in VIDS if v not in DEV16]
print(f'DEBUG train={len(TRAIN)} dev={len([v for v in DEV16 if v in VIDS])}', flush=True)


class TCN(torch.nn.Module):
    """T1-style all-resolution dilated TCN, input projected from 768."""

    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        blocks = []
        for d in dils:
            blocks.append(torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d))
            blocks.append(torch.nn.GELU())
        self.blocks = torch.nn.ModuleList(blocks)
        self.out = torch.nn.Conv1d(ch, 1, 1)
        self.n_blocks = len(dils)

    def forward(self, x):  # x (B, n, d)
        h = self.proj(x.transpose(1, 2))
        for bi in range(self.n_blocks):
            conv, act = self.blocks[2 * bi], self.blocks[2 * bi + 1]
            res = h
            h = act(conv(h) + res)
        return self.out(h).squeeze(1)  # (B, n)


def load(v):
    z = np.load(args.feat_dir / f'{v}.npz')
    return z['pooled'].astype(np.float32), z['label'].astype(np.float32)


train = {v: load(v) for v in TRAIN}
model = TCN().to(args.device).float()
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


def loss_on(v, n_pair=256):
    x, y = train[v]
    m = np.isfinite(y)
    xb = torch.from_numpy(x)[None].to(args.device)
    y1 = torch.from_numpy(np.where(m, y, 0)).to(args.device)
    m1 = torch.from_numpy(m).to(args.device)
    pred = model(xb)[0]  # (n,)
    d = pred[m1] - y1[m1]
    ad = d.abs()
    # hand-written Huber (delta=0.1) from basic ops: F.huber_loss falls back to
    # CPU on the NPU backend and is ~20x slower per step
    hub = torch.where(ad <= 0.1, 0.5 * d * d, 0.1 * (ad - 0.05)).mean()
    idx = np.where(m)[0]
    if len(idx) > 2:
        i = np.random.randint(0, len(idx), n_pair)
        j = np.random.randint(0, len(idx), n_pair)
        keep = idx[i], idx[j]
        dy = y1[list(keep[0])] - y1[list(keep[1])]
        dp = pred[list(keep[0])] - pred[list(keep[1])]
        use = dy.abs() > 0.05
        if use.any():
            rank = torch.nn.functional.softplus(-(dy[use] * dp[use])).mean()
            return hub + 0.5 * rank
    return hub


t0 = time.time()
for step in range(args.steps):
    v = TRAIN[np.random.randint(0, len(TRAIN))]
    loss = loss_on(v)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % 300 == 0:
        print(f'STEP {step + 1} loss={float(loss):.4f}', flush=True)
train_s = time.time() - t0

# ---- dev 16 evaluation: 16 equal segments, score at centre second ----
model.eval()
rows, rng = [], np.random.default_rng(0)
for vid in DEV16:
    if vid not in VIDS:
        continue
    z = np.load(args.feat_dir / f'{vid}.npz')
    lab = np.load(Path('/data/aic/datasets/TVSum/labels_v3') / f'{vid}.proxy-v2.npz', allow_pickle=True)
    fidx, scores = lab['frame_indices'].astype(np.int64), lab['scores'].astype(np.float32)
    nb = int(fidx.max()) + 1
    fps = float(z['fps'])
    x = torch.from_numpy(z['pooled'].astype(np.float32))[None].to(args.device)
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()  # per 1fps sample
    fr_of_sample = z['frame_idx']
    edges = np.linspace(0, nb, 17)
    tgt, sc = [], []
    for k in range(16):
        lo, hi = int(edges[k]), int(edges[k + 1])
        inseg = (fidx >= lo) & (fidx < hi)
        tgt.append(float(scores[inseg].mean()))
        centre_t = (lo + hi) / 2 / fps  # seconds
        centre_f = int(round(centre_t * fps))
        j = int(np.abs(fr_of_sample - centre_f).argmin())
        sc.append(float(pred[j]))
    rr = ranking_report(np.asarray(sc), np.asarray(tgt))
    # paired random / uniform baselines on the same targets
    rrand = [ranking_report(rng.permutation(len(tgt)), np.asarray(tgt))['ndcg_at_15pct'] for _ in range(200)]
    rows.append({'vid': vid, 'ndcg15': rr['ndcg_at_15pct'], 'spearman': rr['spearman'],
                 'top15': rr['top15_mean_relevance'], 'target': tgt, 'score': sc,
                 'random_ndcg15_mean': float(np.mean(rrand)),
                 'uniform_ndcg15': ranking_report(np.ones(16), np.asarray(tgt))['ndcg_at_15pct']})
    print('DEV', vid, f"ndcg15={rr['ndcg_at_15pct']:.3f} rand={np.mean(rrand):.3f}", flush=True)

macro = {k: float(np.mean([r[k] for r in rows])) for k in ('ndcg15', 'spearman', 'top15', 'random_ndcg15_mean', 'uniform_ndcg15')}
args.output_dir.mkdir(parents=True, exist_ok=True)
torch.save({'state_dict': model.state_dict(), 'n_params': n_params, 'seed': args.seed},
           args.output_dir / f'tcn_s{args.seed}.pt')
with open(args.output_dir / f'dev_rows_s{args.seed}.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
summ = {'seed': args.seed, 'n_params': n_params, 'train_vids': len(TRAIN), 'steps': args.steps,
        'macro': macro, 'train_s': round(train_s, 1)}
(args.output_dir / f'summary_s{args.seed}.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
