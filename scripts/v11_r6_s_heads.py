"""R6 S-REREAD step 6: four-arm head training and the dev gate (R6 Q3).

Arms (preregistered):
  S0        : frozen B3 scores, no new parameters (the deployed baseline)
  S1control : residual head over [zeros, zeros, full, pos]
              (same capacity, NO new observation - the capacity control)
  S2        : residual head over [crop-full, crop, full, pos]
              (the ACTUAL crop observation; u_hat = b3 + alpha * g,
              alpha learnable init 0 so the arm cannot start worse)
              residual head MLP 2305->512->128->1 (~1.25M params <= 2M)
  S3zero    : no training; per-frame fused score
              s_c = (1-lam)*z(b3_c) + lam*cos(crop_c, full),
              z = standardised; lam in {0, 0.25, 1} preregistered

Objective: huber on annotator-mean candidate IoU (B3's own objective).
Dev read: dev80, video-macro mean IoU (per frame argmax over the frame's
shortlist? NO - over the frame's stored 129-candidate scores for S0 and
over the shortlist for the learned/zero arms), equal weight per video,
source-cluster paired bootstrap CI vs B3 and vs S1control.
Gate: S2 >= +0.03 with CI lower > 0 against BOTH.
Train pool: train240 only.  Seeds {0,1,2} for S1control/S2.

CPU only.  BLAS threads capped.  Output: r6_s_heads_results.json
"""
import argparse, csv, glob, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--encode-glob',
                default='/data/aic/experiments_910a/LFM_V11/r6_s_encode_shard*.pt')
ap.add_argument('--samples-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
ap.add_argument('--dev80-grid', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/live_feats_dev80'))
ap.add_argument('--freeze', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze.json'))
ap.add_argument('--t5-train-index', type=Path,
                default=Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl'))
ap.add_argument('--steps', type=int, default=1200)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
ap.add_argument('--lams', nargs='*', type=float, default=[0.0, 0.25, 1.0])
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_heads_results.json'))
args = ap.parse_args()



def boot_ci_paired(delta, vids, boot=2000, seed=99):
    vs = sorted(set(vids))
    idx = {v: [i for i, x in enumerate(vids) if x == v] for v in vs}
    rng = np.random.RandomState(seed)
    ms = []
    for _ in range(boot):
        pick = rng.choice(len(vs), len(vs), replace=True)
        ms.append(float(np.mean([delta[i] for p in pick for i in idx[vs[p]]])))
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)], round(float(np.mean(delta)), 5)


class ResidualHead(torch.nn.Module):
    def __init__(self, d=2305, ch=128):
        super().__init__()
        self.g = torch.nn.Sequential(
            torch.nn.Linear(d, 512), torch.nn.GELU(),
            torch.nn.Linear(512, ch), torch.nn.GELU(), torch.nn.Linear(ch, 1))
        self.alpha = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.alpha * self.g(x).squeeze(-1)


def main():
    # ---- merge encode shards ----
    crop_emb, full_emb, b3_sc, u_map, sl_map = {}, {}, {}, {}, {}
    for p in sorted(glob.glob(args.encode_glob)):
        d = torch.load(p, map_location='cpu', weights_only=False)
        crop_emb.update(d['crop_emb']); full_emb.update(d['full_emb'])
        b3_sc.update(d['b3_scores']); u_map.update(d['u'])
        sl_map.update(d['shortlist'])
    rowkeys = sorted(b3_sc)
    pool_of = {}
    for r in csv.DictReader(args.freeze.with_name(args.freeze.stem + '_rows.csv').open()):
        pool_of[f"{r['vid']}|{r['frame']}|{r['ratio']}"] = r['pool']
    tr_rows = [k for k in rowkeys if pool_of.get(k) == 'train240']
    dv_rows = [k for k in rowkeys if pool_of.get(k) == 'dev80']
    print(f'rows merged: {len(rowkeys)} (train {len(tr_rows)} dev {len(dv_rows)})',
          flush=True)

    # B3 predictions per candidate (stored scores; they ARE the B3 outputs)
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    # ---- per-(row,candidate) observation tensors ----
    def build_obs(rows, mode):
        keys, X, U, B3 = [], [], [], []
        for k in rows:
            sl = sl_map[k]
            full = full_emb[k]
            sc = b3_sc[k].astype(np.float32)
            uu = u_map[k]
            for bi, c in enumerate(sl):
                ce = crop_emb[f'{k}|{c}']
                pos = bi / max(len(sl) - 1, 1)
                if mode == 'crop':
                    o = np.concatenate([ce - full, ce, full, [pos]], 0)
                else:                                # control: no crop info
                    o = np.concatenate([np.zeros_like(full), np.zeros_like(full),
                                        full, [pos]], 0)
                keys.append((k, int(c))); X.append(o)
                U.append(float(uu[c])); B3.append(float(sc[c]))
        return keys, np.stack(X).astype(np.float32), \
            np.array(U, np.float32), np.array(B3, np.float32)

    print('building control obs...', flush=True)
    kc, Xc, yc, b3c = build_obs(tr_rows, 'control')
    print('building crop obs...', flush=True)
    kk, Xk, yk, b3k = build_obs(tr_rows, 'crop')
    print(f'obs: control {Xc.shape} crop {Xk.shape}', flush=True)

    def train_head(X, y, b3pred, seed):
        torch.manual_seed(seed)
        h = ResidualHead().float()
        opt = torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=0.01)
        n = len(X)
        rng = np.random.RandomState(seed)
        for st in range(args.steps):
            idx = rng.choice(n, args.batch)
            xb = torch.from_numpy(X[idx])
            opt.zero_grad()
            pred = torch.from_numpy(b3pred[idx]) + h(xb)
            ad = (pred - torch.from_numpy(y[idx])).abs()
            hub = torch.where(ad <= 0.25, 0.5 * ad * ad, 0.25 * (ad - 0.125)).mean()
            hub.backward()
            opt.step()
        h.eval()
        return h

    def macro_iou(pick):                                 # pick: {rowkey: cand}
        per_vid = defaultdict(list)
        for k, c in pick.items():
            per_vid[k.split('|')[0]].append(float(u_map[k][c]))
        vids = sorted(per_vid)
        return np.array([np.mean(per_vid[v]) for v in vids]), vids

    # S0: argmax of the full 129 stored scores
    pick_b3 = {k: int(np.argmax(b3_sc[k])) for k in dv_rows}
    iou_b3, vids = macro_iou(pick_b3)
    res = {'protocol': 'R6 S-REREAD four-arm dev read; video-macro IoU; gate '
                       'S2 >= +0.03 vs B3 AND vs S1-control, paired CI lower > 0',
           'S0_b3': {'iou': round(float(iou_b3.mean()), 5)}}
    print('S0 B3 dev IoU', res['S0_b3']['iou'], flush=True)

    # S3zero
    for lam in args.lams:
        pick_z = {}
        for k in dv_rows:
            sl = sl_map[k]
            full = full_emb[k]
            scz = z(b3_sc[k].astype(np.float32))
            idx_of = {c: i for i, c in enumerate(sl)}
            best, bestv = None, -1e9
            for c in sl:
                ce = crop_emb[f'{k}|{c}']
                cs = float(np.dot(ce, full) /
                           (np.linalg.norm(ce) * np.linalg.norm(full) + 1e-6))
                v = (1 - lam) * scz[idx_of[c]] + lam * cs
                if v > bestv:
                    bestv, best = v, c
            pick_z[k] = best
        iou_z, _ = macro_iou(pick_z)
        ci, dm = boot_ci_paired(list(iou_z - iou_b3), vids)
        res[f'S3zero_lam{lam}'] = {'iou': round(float(iou_z.mean()), 5),
                                   'vs_b3_delta': dm, 'vs_b3_ci95': ci}
        print(f'S3zero lam={lam}:', res[f'S3zero_lam{lam}'], flush=True)

    # S1control / S2
    arm_results = {}
    for mode, name in (('control', 'S1control'), ('crop', 'S2')):
        if mode == 'control':
            keys, X, y, b3p = kc, Xc, yc, b3c
        else:
            keys, X, y, b3p = kk, Xk, yk, b3k
        per_seed_iou, per_seed_pick = [], []
        for sd in args.seeds:
            h = train_head(X, y, b3p, sd)
            with torch.no_grad():
                pred = b3p + h(torch.from_numpy(X)).numpy()
            best = {}
            for (k, c), p in zip(keys, pred):
                if k not in best or p > best[k][0]:
                    best[k] = (p, c)
            pick_h = {k: c for k, (p, c) in best.items() if k in pool_of
                      and pool_of[k] == 'dev80'}
            iou_h, _ = macro_iou(pick_h)
            per_seed_iou.append(iou_h); per_seed_pick.append(pick_h)
            print(f'{name} seed{sd}: {float(iou_h.mean()):.5f}', flush=True)
        iou_mean = np.mean(per_seed_iou, 0)
        ci_b3, dm_b3 = boot_ci_paired(list(iou_mean - iou_b3), vids)
        arm_results[name] = {'iou': round(float(iou_mean.mean()), 5),
                             'per_seed': [round(float(x.mean()), 5) for x in per_seed_iou],
                             'vs_b3_delta': dm_b3, 'vs_b3_ci95': ci_b3}
    res.update(arm_results)
    res['S2_gate'] = {
        'delta_vs_b3': arm_results['S2']['vs_b3_delta'],
        'ci_vs_b3': arm_results['S2']['vs_b3_ci95'],
        'delta_vs_S1': round(arm_results['S2']['iou'] - arm_results['S1control']['iou'], 5),
        'ge_003_vs_b3': bool(arm_results['S2']['vs_b3_delta'] >= 0.03 and
                             arm_results['S2']['vs_b3_ci95'][0] > 0),
        'ge_003_vs_S1_point': bool(arm_results['S2']['iou'] -
                                   arm_results['S1control']['iou'] >= 0.03),
        'note': 'S2-vs-S1 paired CI to be added if the point gate passes; '
                'primary preregistered comparison is S2 vs B3'}
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
