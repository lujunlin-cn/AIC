"""R9 A1 + A2 discriminating audits (R9 report Q1.6; CPU only, frozen pool).

A1 LABEL AUDIT (guard vs centre-point vs occupancy, same frozen list):
  The frozen labels mark slot i positive when the guard window
  [t_i - h, t_i + h] (h = half the mean sampling interval) INTERSECTS a
  PHD2 selection interval.  Two alternative definitions on the same slots:
    centre    : t_i itself lies inside a selection interval
    occupancy : the fraction of the guard window covered by selection
                intervals is > 0.5
  For EACH definition: fit the marginal slot prior on train, evaluate F1 at
  the actual K=6, and re-score the FROZEN champion head.  Reading: how much
  of the prior/head gap is an artefact of the guard definition.

A2 CONTENT/POSITION DECOMPOSITION (frozen champion features/scores):
  Four score variants on the SAME eval fragments:
    real       : champion scores (baseline)
    tperm      : per-fragment temporal permutation of scores (fixed seed)
    meanvec    : per-fragment constant score = mean of its 8 scores
    zerovec    : all-zero scores (R5 anchor)
  Reading: F1 (actual K), AP, and the Jaccard of top-6 sets vs the prior's
  top-6 - separates "ranking inside top-K" (AP-only) from boundary
  movement (F1-relevant).

No training, no NPU, no official media.  Incremental JSON.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--champion', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r9_npu/a1a2_audit.json'))
args = ap.parse_args()

K = 6          # actual budget: round(0.80 * 8)
N = 8


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def load_pool():
    """Verbatim loader + raw materials for alternative labels."""
    rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
    sel = json.loads(args.selections.read_text())
    X, GY, CY, OY, SRC, FID = [], [], [], [], [], []
    for r in rows:
        f = args.feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        if 'mean' not in m or 't' not in m:
            continue
        Xf = m['mean'].astype(np.float32)
        L = len(Xf)
        t0 = float(r['t0'])
        ivs = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']), float(rec['t1'])
                if b_ > a_:
                    ivs.append((a_ - t0, b_ - t0))
        stamps = sorted(float(x) for x in m['t'])
        times = np.array([stamps[i] - t0 for i in range(L)], np.float32)
        half = ((stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1)) * 0.5 if L > 1 else 0.5
        gy = np.zeros(L, np.float32)
        cy = np.zeros(L, np.float32)
        oy = np.zeros(L, np.float32)
        for i in range(L):
            lo, hi = times[i] - half, times[i] + half
            inter = sum(max(0.0, min(hi, b_) - max(lo, a_)) for a_, b_ in ivs)
            gy[i] = 1.0 if inter > 0 else 0.0                 # guard (frozen)
            cy[i] = 1.0 if any(a_ <= times[i] < b_ for a_, b_ in ivs) else 0.0
            oy[i] = 1.0 if (hi - lo) > 0 and inter / (hi - lo) > 0.5 else 0.0
        keep = 0 < gy.sum() < L                               # frozen mixed-only
        if not keep:
            continue
        X.append(Xf); GY.append(gy); CY.append(cy); OY.append(oy)
        SRC.append(r['src']); FID.append(r['video_id'])
    return X, GY, CY, OY, SRC, FID


def f1_at_k(scores, y):
    k = min(K, len(y))
    idx = set(np.argsort(-scores)[:k].tolist())
    hit = sum(y[i] for i in idx)
    g = float(np.sum(y))
    return 0.0 if k + g == 0 else float(2 * hit / (k + g))


def ap_of(scores, y):
    o = np.argsort(-scores)
    ys = np.asarray(y)[o]
    if ys.sum() == 0:
        return None
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def topk_set(scores):
    return frozenset(np.argsort(-scores)[:min(K, len(scores))].tolist())


def cluster_boot_paired(delta, srcs, boot=2000, seed=99):
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
    rec = {'k_actual': K, 'status': 'RUNNING'}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    X, GY, CY, OY, SRC, FID = load_pool()
    ev_src = set(json.loads(args.eval_sources.read_text()))
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    rec['pool'] = {'train': len(tr), 'eval': len(ev)}
    print(f"pool {rec['pool']} ({time.time()-t0:.0f}s)", flush=True)

    ck = torch.load(args.champion, map_location='cpu', weights_only=False)
    champ = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2))))
    champ.load_state_dict(ck['state_dict'])
    champ.eval()
    with torch.no_grad():
        CS = [champ(torch.from_numpy(X[i])[None])[0].numpy().astype(np.float64)
              for i in range(len(X))]
    print(f'champion forward done ({time.time()-t0:.0f}s)', flush=True)

    # ---------- A1 ----------
    a1 = {}
    for name, Y in (('guard', GY), ('centre', CY), ('occupancy', OY)):
        prior = np.zeros(N, np.float64)
        cnt = np.zeros(N, np.float64)
        for i in tr:
            for s, v in enumerate(Y[i]):
                prior[s] += v
                cnt[s] += 1
        prior = prior / np.maximum(cnt, 1)
        ps = [prior[:len(Y[i])] for i in ev]
        pf = [f1_at_k(ps[j], Y[ev[j]]) for j in range(len(ev))]
        cf = [f1_at_k(CS[ev[j]], Y[ev[j]]) for j in range(len(ev))]
        pac = float(np.mean([ap_of(ps[j], Y[ev[j]]) for j in range(len(ev))
                             if ap_of(ps[j], Y[ev[j]]) is not None]))
        cac = float(np.mean([ap_of(CS[ev[j]], Y[ev[j]]) for j in range(len(ev))
                             if ap_of(CS[ev[j]], Y[ev[j]]) is not None]))
        d = [cf[j] - pf[j] for j in range(len(ev))]
        a1[name] = {
            'g_hist': {str(g): int(sum(1 for i in ev if Y[i].sum() == g))
                       for g in sorted(set(int(Y[i].sum()) for i in ev))},
            'prior_f1': round(float(np.mean(pf)), 4),
            'champ_f1': round(float(np.mean(cf)), 4),
            'champ_minus_prior': round(float(np.mean(d)), 4),
            'ci95': cluster_boot_paired(d, [SRC[i] for i in ev]),
            'prior_ap': round(pac, 4), 'champ_ap': round(cac, 4),
            'champ_topk_equals_prior_topk': round(float(np.mean(
                [topk_set(CS[ev[j]]) == topk_set(ps[j]) for j in range(len(ev))])), 4)}
        print(f"A1 {name}: prior {a1[name]['prior_f1']} champ {a1[name]['champ_f1']}", flush=True)
    rec['A1'] = a1
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    # ---------- A2 ----------
    rng = np.random.RandomState(20261006)
    variants = {}
    for i in ev:
        s = CS[i]
        perm = rng.permutation(len(s))
        variants.setdefault('tperm', []).append(s[perm])
        variants.setdefault('meanvec', []).append(np.full(len(s), s.mean()))
        variants.setdefault('zerovec', []).append(np.zeros(len(s)))
    real_f = [f1_at_k(CS[ev[j]], GY[ev[j]]) for j in range(len(ev))]
    a2 = {'real': {'f1': round(float(np.mean(real_f)), 4),
                   'ap': round(float(np.mean([ap_of(CS[ev[j]], GY[ev[j]])
                                              for j in range(len(ev)) if ap_of(CS[ev[j]], GY[ev[j]]) is not None])), 4)}}
    for name, SS in variants.items():
        f = [f1_at_k(SS[j], GY[ev[j]]) for j in range(len(ev))]
        aps = [ap_of(SS[j], GY[ev[j]]) for j in range(len(ev))]
        aps = [a for a in aps if a is not None]
        a2[name] = {'f1': round(float(np.mean(f)), 4),
                    'ap': round(float(np.mean(aps)), 4) if aps else None}
    # top-6 set overlap with the marginal prior mask (audit_k6)
    prior = np.zeros(N, np.float64); cnt = np.zeros(N, np.float64)
    for i in tr:
        for s, v in enumerate(GY[i]):
            prior[s] += v; cnt[s] += 1
    prior = prior / np.maximum(cnt, 1)
    pset = topk_set(prior)
    for name, SS in list(variants.items()) + [('real', [CS[i] for i in ev])]:
        same = float(np.mean([topk_set(SS[j]) == pset for j in range(len(ev))]))
        a2[name]['topk_equals_marginal_prior'] = round(same, 4)
    rec['A2'] = a2
    rec['status'] = 'DONE'
    rec['elapsed_min'] = round((time.time() - t0) / 60, 1)
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == '__main__':
    main()
