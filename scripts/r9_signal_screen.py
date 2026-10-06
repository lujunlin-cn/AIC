"""R9 signal screen (V10 Q1 protocol, first execution - CPU only).

For each probe family (vis, aud) and each per-slot scalar column, fit on
TRAIN sources only: score_s = w0 * logit(prior_s) + w1 * z(x_{i,s}), with
(w0, w1) on a coarse grid maximising train likelihood - a 2-parameter
family, no head capacity to memorise position.  Evaluate on EVAL:

  F1 at the actual K=6, paired delta vs prior-only (w1=0) with
  source-cluster CI, and boundary-swap net utility (incoming positives -
  outgoing positives vs the prior mask).

Family-level alpha bar (preregistered here before seeing eval numbers):
per family, the BEST column must have CI lower > 0 on the F1 delta AND
net swap utility > 0.  With ~20 columns screened, family-wise error is
controlled by requiring the column to win on BOTH statistics plus the
confirm-split rerun later (V10 Q8 two-stage protocol).
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--probe-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r9_signal_probe'))
ap.add_argument('--manifest-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r9_npu/signal_screen.json'))
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()

K = 6
GRIDS = [(w0, w1) for w0 in (0.0, 0.5, 1.0, 2.0) for w1 in (0.0, 0.25, 0.5, 1.0, 2.0)]


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def f1_at_k(scores, y):
    k = min(K, len(y))
    idx = set(np.argsort(-scores)[:k].tolist())
    hit = sum(y[i] for i in idx)
    g = float(np.sum(y))
    return 0.0 if k + g == 0 else float(2 * hit / (k + g))


def like(scores, y):
    s = np.asarray(scores, np.float64)
    yv = np.asarray(y, np.float64)
    p = 1 / (1 + np.exp(-s))
    return float((yv * np.log(p + 1e-9) + (1 - yv) * np.log(1 - p + 1e-9)).sum())


def cluster_boot_paired(delta, srcs, boot, seed=99):
    us = sorted(set(srcs))
    idx = {s: [i for i, x in enumerate(srcs) if x == s] for s in us}
    rng = np.random.RandomState(seed)
    ms = []
    for _ in range(boot):
        pick = rng.choice(len(us), len(us), replace=True)
        ms.append(float(np.mean([delta[i] for q in pick for i in idx[us[q]]])))
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return [round(float(lo), 5), round(float(hi), 5)]


def main():
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    man = []
    for split in ('train_public', 'eval_public'):
        for l in (args.manifest_dir / f'{split}.jsonl').read_text().splitlines():
            if l.strip():
                man.append(json.loads(l))
    files = {}
    for f in sorted(args.probe_root.glob('p*/*.npz')):
        files[f.stem] = f
    man = [r for r in man if r['fragment_id'] in files]
    ev_src = set(json.loads(args.eval_sources.read_text()))
    X, Y, SRC = [], [], []
    AUD_OK = []
    for r in man:
        m = np.load(files[r['fragment_id']])
        X.append(m)
        Y.append([int(v) for v in json.loads(json.dumps(r))['labels']])
        SRC.append(r['source_id'])
        AUD_OK.append(int(m['has_audio']))
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    # marginal prior from train
    prior = np.zeros(8, np.float64)
    cnt = np.zeros(8, np.float64)
    for i in tr:
        for s, v in enumerate(Y[i]):
            prior[s] += v
            cnt[s] += 1
    prior = prior / np.maximum(cnt, 1)
    lp = logit(prior)
    print(f'pool train {len(tr)} eval {len(ev)} audio_frag_frac '
          f'{np.mean([AUD_OK[i] for i in tr]):.3f} ({time.time()-t0:.0f}s)', flush=True)

    rec = {'protocol': 'score = w0*logit(prior_s) + w1*z(x); 2-param grid on '
                       'train likelihood; alpha bar = F1-delta CI lower > 0 AND '
                       'net swap utility > 0; two-stage confirm still required',
           'audio_source_frac_train': round(float(np.mean([AUD_OK[i] for i in tr])), 4),
           'families': {}, 'status': 'RUNNING'}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    # baseline once
    pf = [f1_at_k(lp[:8], Y[i]) for i in ev]
    base_f = float(np.mean(pf))
    pmask = set(np.argsort(-prior)[:K].tolist())

    for fam, keys in (('vis', ['frame_diff', 'sharpness', 'brightness', 'saturation', 'hist_chi2']),
                      ('aud', ['rms_mean', 'rms_std', 'rms_max', 'rms_flux', 'silence', 'cheer_contrast'])):
        fam_res = {}
        cols = {}
        for i in range(len(X)):
            arr = X[i][fam].astype(np.float32)
            for j, key in enumerate(keys):
                cols.setdefault(j, []).append(arr[:, j])
        for j, key in enumerate(keys):
            xs = cols[j]
            mu = np.mean([xs[i] for i in tr])
            sd = np.std([xs[i] for i in tr]) + 1e-6
            zs = [((xs[i] - mu) / sd).astype(np.float64) for i in range(len(X))]
            # fit grid on train likelihood
            best, best_w = -1e18, (1.0, 0.0)
            for w0, w1 in GRIDS:
                L = sum(like(w0 * lp[:8] + w1 * zs[i], Y[i]) for i in tr)
                if L > best:
                    best, best_w = L, (w0, w1)
            w0, w1 = best_w
            f = [f1_at_k(w0 * lp[:8] + w1 * zs[i], Y[i]) for i in ev]
            d = [f[j] - pf[j] for j in range(len(ev))]
            ci = cluster_boot_paired(d, [SRC[i] for i in ev], args.boot)
            swaps_in = swaps_out = 0
            for j in range(len(ev)):
                m = set(np.argsort(-(w0 * lp[:8] + w1 * zs[ev[j]]))[:K].tolist())
                yv = np.array(Y[ev[j]])
                swaps_in += int(sum(yv[s] for s in m - pmask))
                swaps_out += int(sum(yv[s] for s in pmask - m))
            fam_res[key] = {'w': [w0, w1], 'f1': round(float(np.mean(f)), 4),
                            'delta_vs_prior': round(float(np.mean(d)), 4),
                            'ci95': ci,
                            'net_swaps': int(swaps_in - swaps_out)}
            print(f'{fam}:{key}: w {w0},{w1} F1 {np.mean(f):.4f} d {np.mean(d):+.4f} '
                  f'CI {ci} net {swaps_in - swaps_out}', flush=True)
        best = max(fam_res.values(), key=lambda r: r['delta_vs_prior'])
        fam_res['alpha_qualifies'] = bool(best['ci95'][0] > 0 and best['net_swaps'] > 0)
        rec['families'][fam] = fam_res
        args.out.write_text(json.dumps(rec, indent=1) + '\n')

    rec['baseline_prior_f1'] = round(base_f, 4)
    rec['status'] = 'DONE'
    rec['elapsed_min'] = round((time.time() - t0) / 60, 1)
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
