"""Oracle decomposition: where exactly is the score loss?  (GPT-6-PRO P1)

Three questions, kept strictly apart, because each implies a different next
stage:

  Oracle A (fixed-rank best-k)  -- freeze the CURRENT head's ranking; for every
      video choose k by oracle (GT picks the best prefix).  Answers: how much
      would perfect per-video keep decisions add on top of the ranking we
      already have?  If this is small, calibration/threshold work is dead on
      arrival; the ranking itself must improve.

  Oracle B (ranking headroom via best achievable ordering) -- the same frames,
      but the ranking replaced by the GT-perfect one: keep exactly the GT
      prefix.  Together with A this splits "decision gap" from "ranking gap".

  Reachability arithmetic -- the official drop has 159 no-slide videos (37.3%)
      where our box is the full frame by construction; if the GT box there is
      also the full frame, those videos have spatial IoU = 1 and the uniform
      m = 0.62 assumption understates the ceiling.  This script does NOT assume
      an answer: it prints, for a grid of (m_xy, p) values, the implied
      keep-all score, so the observed 0.3475 pins down which (m_xy, p) region
      is consistent with the platform number.

All temporal numbers are BINARY temporal F1 on PHD2 fragments - the honest
proxy, since PHD2 has no per-frame spatial GT.  Constant-IoU numbers are
reported only next to their binary counterpart and labelled as proxies.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def temporal_f1(mask, y):
    """Binary temporal F1: numerator counts kept GT frames (proxy IoU=1),
    denominator is |kept| + |GT|.  This is the F_t of the decomposition, not a
    platform score."""
    keep = int(mask.sum()); gt = int(y.sum())
    if gt == 0:
        return 1.0 if keep == 0 else 0.0
    hit = float(y[mask].sum())
    return 2.0 * hit / (keep + gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
    ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/lfm_feats'))
    ap.add_argument('--native-ckpt', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/tcn_full/tcn_s0.pt'))
    ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/oracle_decomposition.json'))
    args = ap.parse_args()

    ck = torch.load(args.native_ckpt, map_location='cpu', weights_only=False)
    model = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2, 4))))
    model.load_state_dict(ck['state_dict']); model.eval()

    sel = json.loads(Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json').read_text())
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]

    # per SOURCE VIDEO aggregation: an official video corresponds to a source,
    # fragments are just windows into it.  Concatenate all scored frames of a
    # source into one ranked list - the same thing the deployment mask does
    # per official video.
    per_src = {}
    for r in rows:
        d = args.feat_root / r['video_id']
        if not d.exists():
            continue
        fs = sorted(d.glob('*.npz'), key=lambda p: float(p.stem))
        if len(fs) < 4:
            continue
        X = np.stack([np.load(f)['pooled'].astype(np.float32) for f in fs])
        L = len(X)
        t0 = float(r['t0'])
        ivs = [(float(x['t0']) - t0, float(x['t1']) - t0)
               for recs in sel.get(r['src'], {}).values() for x in recs if float(x['t1']) > float(x['t0'])]
        stems = sorted(float(f.stem) for f in fs)
        times = np.array([stems[i] - t0 for i in range(L)], np.float32)
        y = np.zeros(L, np.float32)
        half = ((stems[-1] - stems[0]) / max(len(stems) - 1, 1)) * 0.5 if len(stems) > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if y.sum() == 0 or y.sum() == L:
            continue
        with torch.no_grad():
            s = torch.sigmoid(model(torch.from_numpy(X)[None]))[0].numpy()
        per_src.setdefault(r['src'], []).append((s, y))

    if not per_src:
        raise SystemExit('no sources scored')

    res_a, res_cur, res_all = [], [], []
    detail = []
    for src, parts in per_src.items():
        s = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        n = len(s)
        order = np.argsort(-s)
        # Oracle A: best prefix over the CURRENT ranking
        best_f, best_k = -1.0, n
        for k in range(0, n + 1):
            m = np.zeros(n, bool); m[order[:k]] = True
            f = temporal_f1(m, y)
            if f > best_f:
                best_f, best_k = f, k
        # current deployment: fixed keep 0.8
        k08 = max(1, int(round(0.8 * n)))
        m08 = np.zeros(n, bool); m08[order[:k08]] = True
        # keep-all
        res_a.append(best_f)
        res_cur.append(temporal_f1(m08, y))
        res_all.append(temporal_f1(np.ones(n, bool), y))
        detail.append({'src': src, 'n': int(n), 'p': round(float(y.mean()), 3),
                       'f_oracleA': round(best_f, 4), 'k_oracle': int(best_k),
                       'f_keep08': round(float(temporal_f1(m08, y)), 4),
                       'f_keepall': round(float(temporal_f1(np.ones(n, bool), y)), 4)})

    oA = float(np.mean(res_a)); oCur = float(np.mean(res_cur)); oAll = float(np.mean(res_all))
    out = {
        'n_sources': len(per_src),
        'binary_temporal_F1': {
            'keep_all': round(oAll, 4),
            'fixed_keep_0.8': round(oCur, 4),
            'oracle_A_fixed_rank_best_k': round(oA, 4),
            'oracle_A_gap_vs_keep08': round(oA - oCur, 4),
            'perfect_ranking_ceiling': 1.0,
            'ranking_gap_vs_oracleA': round(1.0 - oA, 4),
        },
        'reading': None,
    }
    # decision rule, per GPT-6-PRO P1/P2/P3
    if oA < 0.72:
        out['reading'] = ('Oracle A is LOW: even perfect per-video keep decisions on the '
                          'current ranking cannot reach the temporal F1 that a 40+ score '
                          'needs. Calibration/threshold work is dead; the ranking itself '
                          'must improve (P3).')
    else:
        out['reading'] = ('Oracle A is HIGH: the current ranking contains enough signal '
                          'and the loss is in the decision (keep choice). Calibration / '
                          'adaptive keep is the right next stage (P2).')

    args.out.write_text(json.dumps(out, indent=1))
    with args.out.with_suffix('.per_video.csv').open('w') as fh:
        fh.write('src,n,p,f_oracleA,k_oracle,f_keep08,f_keepall\n')
        for d in detail:
            fh.write(f"{d['src']},{d['n']},{d['p']},{d['f_oracleA']},{d['k_oracle']},{d['f_keep08']},{d['f_keepall']}\n")

    # ---- reachability grid: which (m_xy, p) are consistent with 0.3475? ----
    # keep-all score = [0.373*1 + 0.627*m_xy] * 2p/(1+p), macro over geometry.
    print(json.dumps(out, indent=1), flush=True)
    print('\n=== reachability grid: keep-all platform score for (m_xy, p) ===', flush=True)
    print('rows m_xy (x/y videos spatial IoU), cols p (highlight frame rate); the', flush=True)
    print('observed 34.75 pins the consistent region; no-slide IoU fixed at 1.0', flush=True)
    header = 'm_xy\\p ' + ' '.join(f'{p:>6}' for p in (0.20, 0.30, 0.39, 0.50))
    print(header, flush=True)
    for m_xy in (0.50, 0.62, 0.70, 0.80):
        row = f'{m_xy:>6} '
        for p in (0.20, 0.30, 0.39, 0.50):
            m_bar = 0.373 * 1.0 + 0.627 * m_xy
            row += f'{100 * m_bar * 2 * p / (1 + p):>6.1f}'
        print(row, flush=True)


if __name__ == '__main__':
    main()