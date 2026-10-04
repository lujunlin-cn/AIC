"""Round-5 S0-ORACLE analysis (R5 section 2.5) over the EXISTING B3
per-frame records - no new inference, pure CPU read of frozen files.

Inputs (frozen, produced by v8_s_train_multidata.py eval passes):
  LFM_V8/arms/B3_s{0,1}/per_live_confirm.jsonl   LOCKED-OUT live sources
                 per_live_dev.jsonl              dev (selection) sources
                 per_rv_diag.jsonl               RetargetVid, EXPOSED diag
  each row: vid, frame, ratio, iou (head argmax), best (candidate oracle),
            center (centre candidate)

Outputs (R5 section 9): spatial_candidate_oracle.csv + a summary block
appended to the round-5 report json:
  - oracle-minus-head gap distribution (mean / median / p25 / p75)
  - stratified by ratio (x-axis 9-16 vs others), by d_center movement bucket,
    by frame index bucket
  - head-minus-center (what the learned scorer buys over the centre pick)
  - practical-gate read: R5 says train S1 only if a scorer gap exists; the
    B3 objective IS already candidate-utility (audit A5), so the remaining
    questions are (a) size of the scorer gap and (b) whether it is
    concentrated in a stratum we can act on.  If oracle-head mean gap is
    below the preregistered 0.02 threshold on the locked-out pool with CI
    excluding 0 in the WRONG direction, pause the S1 family.

CPU only.
Output: /data/aic/experiments_910a/LFM_V11/round5_spatial_oracle.json
        /data/aic/experiments_910a/LFM_V11/spatial_candidate_oracle.csv
"""
import argparse, csv, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--arms-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/arms'))
ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/round5_spatial_oracle.json'))
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()

FILES = {
    'live_confirmation': ('B3_s0', 'per_live_confirm.jsonl'),
    'live_confirmation_s1': ('B3_s1', 'per_live_confirm.jsonl'),
    'live_dev': ('B3_s0', 'per_live_dev.jsonl'),
    'rv_diag_exposed': ('B3_s0', 'per_rv_diag.jsonl'),
}


def boot_ci_vid(rows, col, boot, seed=20261004):
    vids = sorted(set(r['vid'] for r in rows))
    vidx = {v: [i for i, r in enumerate(rows) if r['vid'] == v] for v in vids}
    if col == 'gap':
        vals = np.array([r['best'] - r['iou'] for r in rows], np.float64)
    else:
        vals = np.array([r[col] for r in rows], np.float64)
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(boot):
        pick = rng.choice(len(vids), len(vids), replace=True)
        means.append(float(np.mean([vals[i] for p in pick for i in vidx[vids[p]]])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def strat(rows, keyfn):
    out = {}
    for name, sub in keyfn.items():
        s = [r for r in rows if sub(r)]
        if not s:
            continue
        g = np.array([r['best'] - r['iou'] for r in s], np.float64)
        out[name] = {'n': len(s),
                     'head_iou': round(float(np.mean([r['iou'] for r in s])), 4),
                     'oracle_iou': round(float(np.mean([r['best'] for r in s])), 4),
                     'gap_mean': round(float(g.mean()), 4),
                     'gap_median': round(float(np.median(g)), 4),
                     'gap_p75': round(float(np.percentile(g, 75)), 4)}
    return out


def main():
    report, csv_rows = {}, []
    for pool, (arm, fn) in FILES.items():
        p = args.arms_dir / arm / fn
        if not p.exists():
            report[pool] = {'status': 'MISSING', 'path': str(p)}
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        gap = np.array([r['best'] - r['iou'] for r in rows], np.float64)
        hc = np.array([r['iou'] - r['center'] for r in rows], np.float64)
        entry = {
            'n_frames': len(rows), 'n_videos': len(set(r['vid'] for r in rows)),
            'head_iou': round(float(np.mean([r['iou'] for r in rows])), 4),
            'oracle_iou': round(float(np.mean([r['best'] for r in rows])), 4),
            'center_iou': round(float(np.mean([r['center'] for r in rows])), 4),
            'gap_mean': round(float(gap.mean()), 4),
            'gap_median': round(float(np.median(gap)), 4),
            'gap_p90': round(float(np.percentile(gap, 90)), 4),
            'gap_frac_ge_005': round(float((gap >= 0.05).mean()), 4),
            'head_minus_center': round(float(hc.mean()), 4),
            'gap_ci95_vid': boot_ci_vid(rows, 'gap', args.boot),
            'gap_zero_ci95_vid': boot_ci_vid(rows, 'iou', args.boot),
        }
        # stratifications (R5 2.5 step 4)
        entry['by_ratio'] = strat(rows, {
            k: (lambda r, k=k: r['ratio'] == k)
            for k in sorted(set(r['ratio'] for r in rows))})
        dc = np.array([abs(r['iou'] - r['center']) for r in rows])
        # movement proxy: head's distance from centre in IoU terms
        entry['by_head_moves'] = strat(rows, {
            'head_eq_center': lambda r: abs(r['iou'] - r['center']) < 1e-9,
            'head_ne_center': lambda r: abs(r['iou'] - r['center']) >= 1e-9})
        fr = [r['frame'] for r in rows]
        med_fr = float(np.median(fr))
        entry['by_frame_half'] = strat(rows, {
            'frame_le_median': lambda r: r['frame'] <= med_fr,
            'frame_gt_median': lambda r: r['frame'] > med_fr})
        report[pool] = entry
        for r in rows:
            csv_rows.append([pool, arm, r['vid'], r['frame'], r['ratio'],
                             round(r['iou'], 4), round(r['best'], 4),
                             round(r['center'], 4),
                             round(r['best'] - r['iou'], 4)])
        print(pool, json.dumps({k: v for k, v in entry.items()
                                if not isinstance(v, dict)}, indent=1), flush=True)

    # preregistered practical read (R5 2.5 + A5 audit)
    lc = report.get('live_confirmation', {})
    verdict = 'NOT_RUN'
    if 'gap_mean' in lc:
        ci = lc['gap_ci95_vid']
        if lc['gap_mean'] >= 0.02 and ci[0] > 0:
            verdict = ('SCORER_GAP_PRESENT: locked-out pool shows an exploitable '
                       'scorer gap >=0.02 with CI above zero.  NOTE A5: B3 is '
                       'ALREADY a candidate-utility head (huber on annotator-mean '
                       'IoU, v8_s_train_multidata.py), so re-implementing the '
                       'objective is a duplicate.  The attack is feature/backbone, '
                       'candidate-geometry, TTA (S4), or supervision breadth - '
                       'NOT the loss.  A head that closes half the gap toward '
                       'oracle would add ~'
                       + str(round(0.5 * lc['gap_mean'], 4)) +
                       ' mean IoU on this pool; official-domain transfer is NOT '
                       'implied by this number.')
        elif lc['gap_mean'] < 0.02:
            verdict = ('PAUSED_BUDGET: scorer gap below the preregistered 0.02 '
                       'practical threshold on locked-out sources; residual error '
                       'is dominated by candidate geometry / features, not the '
                       'scorer.  Attack geometry (more/better candidates) or the '
                       'backbone before training any new scorer.')
        else:
            verdict = 'INCONCLUSIVE'
    report['verdict'] = verdict

    out_csv = args.out.parent / 'spatial_candidate_oracle.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pool', 'arm', 'vid', 'frame', 'ratio', 'head_iou',
                    'oracle_iou', 'center_iou', 'gap'])
        w.writerows(csv_rows)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print('verdict:', verdict)
    print('WROTE', args.out, out_csv, flush=True)


if __name__ == '__main__':
    main()
