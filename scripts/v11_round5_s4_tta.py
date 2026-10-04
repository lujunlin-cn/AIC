"""Round-5 S4 (TTA, zero-parameter): score-smoothing over the 129 candidates.

Rationale: adjacent candidates are highly overlapping windows, so the true
utility along the candidate axis should be spatially smooth; a noisy argmax
peak can be corrected by smoothing the score sequence before argmax.
Zero new features, zero new weights - pure post-processing on B3 scores,
evaluated on the SAME frozen pools as the S0 audit (dev = selection,
confirmation = locked-out).

Variants (preregistered):
  raw        argmax as deployed (baseline)
  smooth_k   Gaussian smoothing over the candidate axis, half-width k in
             {1, 2, 3} candidates (normalised, edge-truncated), then argmax
  top2_mean  mean of the top-2 scores' positions as a soft argmax fallback
             (rounded to the nearer candidate)

Readout: mean annotator-IoU per variant, paired delta vs raw with a
source-cluster bootstrap CI (2000 resamples).  Strata: live_dev
(selection), live_confirmation (locked-out).  This is a DEV read; a
positive locked-out delta >= +0.005 with CI above zero earns a candidate-
geometry follow-up (sub-candidate interpolation), not a package.

Output: /data/aic/experiments_910a/LFM_V11/round5_s4_tta.json
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--samples-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
ap.add_argument('--head', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt'))
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_s4_tta.json'))
args = ap.parse_args()

KS = (1, 2, 3)


def gauss_kernel(k):
    xs = np.arange(-k, k + 1, dtype=np.float32)
    w = np.exp(-0.5 * (xs / (k / 1.5)) ** 2)
    return w / w.sum()


def smooth_scores(s, k):
    w = gauss_kernel(k)
    out = np.empty_like(s)
    for j in range(s.shape[1]):
        lo, hi = max(0, j - k), min(s.shape[1], j + k + 1)
        ww = w[(k - (j - lo)):(k + (hi - j))].copy()
        ww = ww / ww.sum()
        out[:, j] = s[:, lo:hi] @ ww
    return out


def main():
    ck = torch.load(args.head, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    d = sd['proj.0.weight'].shape[1]
    ch = sd['proj.0.weight'].shape[0]

    class Head(torch.nn.Module):
        """Verbatim from v8_s_train_multidata.py (MLP + 2 self-attn)."""

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

    model = Head(d, 129)
    model.load_state_dict(sd)
    model.eval()
    torch.set_grad_enabled(False)
    print(f'head loaded, d={d}', flush=True)

    results = {}
    for pool, tag in (('live_dev', 'dev'), ('live_confirmation', 'confirmation')):
        feat = np.load(args.samples_dir / f'{pool}_feat.npy', mmap_mode='r')
        u = np.load(args.samples_dir / f'{pool}_u.npy')
        vids = np.load(args.samples_dir / f'{pool}_vid.npy', allow_pickle=True)
        n = len(u)
        scores = np.empty((n, 129), np.float32)
        B = 4096
        for i in range(0, n, B):
            xb = torch.from_numpy(np.array(feat[i:i + B], dtype=np.float32))
            scores[i:i + B] = model(xb).numpy()
        picks = {'raw': np.argmax(scores, 1)}
        for k in KS:
            picks[f'smooth_{k}'] = np.argmax(smooth_scores(scores, k), 1)
        top2 = np.argsort(-scores, 1)[:, :2]
        w2 = np.take_along_axis(scores, top2, 1)
        # soft argmax over the top-2 neighbourhood, but only when the two
        # scores are close (else the raw winner stands)
        near = (w2[:, 0] - w2[:, 1]) <= 0.05
        cand = (top2[:, 0] + top2[:, 1]) // 2
        picks['top2_mean'] = np.where(near, cand, top2[:, 0])
        ent = {}
        for name, pk in picks.items():
            sel_u = np.take_along_axis(u, pk[:, None], 1)[:, 0]
            ent[name] = sel_u
        res = {'n_frames': n, 'n_videos': int(len(set(vids.tolist())))}
        raw = ent['raw']
        rng = np.random.RandomState(20261005)
        uv = sorted(set(vids.tolist()))
        vidx = {v: np.where(vids == v)[0] for v in uv}
        for name, vals in ent.items():
            res[name] = round(float(np.mean(vals)), 5)
        res['paired_delta_ci'] = {}
        for name in list(ent):
            if name == 'raw':
                continue
            delta = ent[name] - raw
            means = []
            for _ in range(args.boot):
                pick = rng.choice(len(uv), len(uv), replace=True)
                means.append(float(np.mean([delta[i] for p in pick for i in vidx[uv[p]]])))
            lo, hi = np.percentile(means, [2.5, 97.5])
            res['paired_delta_ci'][name] = {
                'mean': round(float(delta.mean()), 5),
                'ci95': [round(float(lo), 5), round(float(hi), 5)]}
        results[tag] = res
        print(tag, json.dumps(res, indent=1), flush=True)

    args.out.write_text(json.dumps(results, indent=1) + '\n')
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
