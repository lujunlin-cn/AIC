"""Aggregate the round-5 P/O/S/L single-task runs into the preregistered read.

Protocol (unchanged from v11_round5_posbl_matrix.py):
  * lr chosen by the O arm's 3-seed mean dev AP (scan values as run);
  * each arm read at the chosen lr, 3 seeds;
  * primary: dev AP and keep-0.80 action-level F1; paired O-P / O-S / O-L
    deltas with source-cluster bootstrap CI;
  * DEV READ ONLY.

Usage: python3 v11_round5_posbl_aggregate.py
Input: /data/aic/experiments_910a/LFM_V11/round5_posbl_tasks/*_task_*.npz
Output: /data/aic/experiments_910a/LFM_V11/round5_posbl_matrix.json
"""
import argparse, glob, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--task-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_posbl_tasks'))
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_posbl_matrix.json'))
args = ap.parse_args()

ARMS = ('O', 'P', 'S', 'L')
SEEDS = (510, 511, 512)


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def f1_keep(s, y, keep):
    n = len(y)
    k = max(1, int(round(keep * n)))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def boot_ci(delta, keep_src, boot, seed=20261005):
    keep_src = np.asarray(keep_src)
    uv = sorted(set(keep_src.tolist()))
    vidx = {v: np.where(keep_src == v)[0] for v in uv}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(boot):
        pick = rng.choice(len(uv), len(uv), replace=True)
        means.append(float(np.mean([delta[i] for p in pick for i in vidx[uv[p]]])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    files = glob.glob(str(args.task_dir / '*_task_*.pt'))
    runs = {}
    for f in files:
        z = torch.load(f, map_location='cpu', weights_only=False)
        arm, seed, lr = str(z['arm']), int(z['seed']), float(z['lr'])
        runs[(arm, seed, lr)] = z
    print(f'loaded {len(files)} task files', flush=True)

    # chosen lr by the O arm's 3-seed mean AP
    lrs = sorted({lr for (a, s, lr) in runs if a == 'O'})
    lr_ap = {}
    for lr in lrs:
        aps = []
        for sd in SEEDS:
            z = runs.get(('O', sd, lr))
            if z is None:
                continue
            dv = z['dv']
            sc = [z['scores'][i] for i in dv]
            ys = [z['labels'][i] for i in dv]
            aps.append(float(np.mean([ap_of(sc[i], ys[i]) for i in range(len(sc))])))
        lr_ap[lr] = round(float(np.mean(aps)), 4) if aps else None
    chosen = max((l for l in lrs if lr_ap[l] is not None), key=lambda l: lr_ap[l])
    print('lr scan:', lr_ap, '-> chosen', chosen, flush=True)

    results = {'lr_scan_dev_ap': {str(k): v for k, v in lr_ap.items()},
               'chosen_lr': chosen, 'arms': {}, 'note': 'DEV READ ONLY'}
    per_arm = {}
    for arm in ARMS:
        aps, f1s, sc_list = [], [], []
        src_of = None
        for sd in SEEDS:
            z = runs.get((arm, sd, chosen))
            if z is None:
                continue
            dv = z['dv']
            sc = [z['scores'][i] for i in dv]
            ys = [z['labels'][i] for i in dv]
            kp = z['keep']
            aps.append(float(np.mean([ap_of(sc[i], ys[i]) for i in range(len(sc))])))
            f1s.append(float(np.mean([f1_keep(sc[i], ys[i], args.keep)
                                      for i in range(len(sc))])))
            sc_list.append(sc)
            src_of = kp
            y_ref = ys
        if not aps:
            results['arms'][arm] = 'MISSING'
            continue
        results['arms'][arm] = {'dev_ap_mean': round(float(np.mean(aps)), 4),
                                'dev_f1_mean': round(float(np.mean(f1s)), 4),
                                'ap_per_seed': [round(a, 4) for a in aps]}
        per_arm[arm] = (sc_list, y_ref, src_of)
        print(arm, results['arms'][arm], flush=True)

    # paired deltas O-P / O-S / O-L (seed-paired, source-cluster CI)
    deltas = {}
    for other in ('P', 'S', 'L'):
        if other not in per_arm or 'O' not in per_arm:
            continue
        d = []
        for si in range(3):
            sc_o, sc_x = per_arm['O'][0][si], per_arm[other][0][si]
            ys = per_arm['O'][1]
            d += [ap_of(sc_o[i], ys[i]) - ap_of(sc_x[i], ys[i])
                  for i in range(len(sc_o))]
        ci = boot_ci(np.array(d), per_arm['O'][2], args.boot)
        deltas[f'O_minus_{other}'] = {'mean': round(float(np.mean(d)), 4), 'ci95': ci}
    results['paired_deltas'] = deltas
    print(json.dumps(deltas, indent=1), flush=True)

    # preregistered attribution lines
    apO = results['arms']['O']['dev_ap_mean']
    attrib = {}
    for other, label in (('P', 'multi-frame content beats position prior'),
                         ('S', 'temporal ORDER beats shuffled'),
                         ('L', 'dense sampling beats 1Hz')):
        if other in results['arms'] and isinstance(results['arms'][other], dict):
            attrib[f'O_gt_{other}'] = bool(apO > results['arms'][other]['dev_ap_mean'])
    results['attribution'] = attrib
    args.out.write_text(json.dumps(results, indent=1) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
