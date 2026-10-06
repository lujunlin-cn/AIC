"""R8 temporal-head preregistered gate (r7 IV2_1B_TEMPORAL step 3).

Two phases (user-approved 2026-10-06: cheap replication first, then the
formal gate):

  --phase pilot   pool stats + champion replication + slotprior + one mse
                  seed, all on the common pool, to prove the pipeline
                  reproduces the R5 anchors before anything is claimed:
                    champion real F1 ~ 0.6588, slotprior ~ 0.6625
                  (round5_content_ablation.json, same pool/readout)

  --phase full    preregistered gate: arms {mse, exactdp} x seeds
                  {20261006..08}, TCN from scratch (r7: old head weights
                  are not a control); ARM-LEVEL 3-seed mean picks the
                  judging arm (pre-recorded selection rule); paired
                  per-fragment delta vs champion with SOURCE-cluster
                  bootstrap; GATE: delta >= +0.007 AND CI lower > 0
                  (r7 preregistration IV2_1B_TEMPORAL).  slotprior is
                  reported as the honest bar the head must also clear.

Common pool (archaeology 2026-10-06, all readouts verbatim from
v11_expected_f_train.py so the caliber cannot drift):
  feats   LFM_V10/pool_feats/{video_id}.npz  'mean' (8,768) + 't'
  index   PHD2_FRAG_V1/index_clean.jsonl
  labels  PHD2 selections intervals projected onto slot guard windows,
          MIXED-ONLY fragments (0 < pos < L)
  split   eval_sources_50.json, source-disjoint
  metric  f1_at_keep(scores, y, 0.80) = 2*hit/(k + n_gt), macro

CPU only.  Incremental JSON after every stage.
"""
import argparse, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--phase', required=True, choices=['pilot', 'full'])
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--champion', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/temporal_gate.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seeds', nargs='*', type=int, default=[20261006, 20261007, 20261008])
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()


class TCN(torch.nn.Module):
    """Verbatim from v11_expected_f_train.py (champion family; its training
    default is dils (1,2,4); champion ckpts carry their own ch/dils)."""

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


def load_fragments(feat_root, index, sel, keep_src):
    """Verbatim loader (mixed-only) from v11_expected_f_train.py."""
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC, FID = [], [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
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
        y = np.zeros(L, np.float32)
        half = ((stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1)) * 0.5 if L > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if not (0 < y.sum() < L):
            continue
        X.append(Xf); Y.append(y); SRC.append(r['src']); FID.append(r['video_id'])
    return X, Y, SRC, FID


def f1_at_keep(scores, y, keep):
    n = len(y)
    k = max(1, int(round(keep * n)))
    idx = set(np.argsort(-scores)[:k].tolist())
    hit = sum(y[i] for i in idx)
    return float(2 * hit / (k + y.sum()))


def ap_of(scores, y):
    o = np.argsort(-scores)
    ys = y[o]
    if ys.sum() == 0:
        return None
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def _expected_f_ref(logits, y):
    """Verbatim expected_f_binary from v11_expected_f_train.py (lines 71-89)."""
    G = float(y.sum())
    if G == 0:
        p = torch.sigmoid(logits)
        return (1 - p).prod()
    p = torch.sigmoid(logits)
    T = logits.shape[0]
    prob = logits.new_ones(1)
    mass = logits.new_zeros(1)
    for t in range(T):
        old_p, old_m = prob, mass
        prob = (1 - p[t]) * torch.nn.functional.pad(old_p, (0, 1)) + \
            p[t] * torch.nn.functional.pad(old_p, (1, 0))
        mass = ((1 - p[t]) * torch.nn.functional.pad(old_m, (0, 1))
                + p[t] * torch.nn.functional.pad(old_m + y[t] * old_p, (1, 0)))
    k = torch.arange(prob.numel(), dtype=logits.dtype, device=logits.device)
    return (2.0 * mass / (k + G)).sum()


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
    sel = json.loads(args.selections.read_text())
    ev_src = set(json.loads(args.eval_sources.read_text()))
    X, Y, SRC, FID = load_fragments(args.feat_root, args.index, sel, ev_src)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    print(f'pool: train {len(tr)} eval {len(ev)} frags '
          f'({time.time()-t0:.0f}s)', flush=True)

    rec = {'phase': args.phase,
           'protocol': 'R8 temporal gate; common pool = pool_feats mean '
                       '(8,768) + PHD2_FRAG_V1 mixed-only + eval_sources_50 '
                       'source split; f1_at_keep 0.80 verbatim from '
                       'v11_expected_f_train',
           'pool': {'train_frags': len(tr), 'eval_frags': len(ev),
                    'eval_srcs': len(set(SRC[i] for i in ev)),
                    'slots_hist': {str(L): sum(1 for i in ev if len(X[i]) == L)
                                   for L in sorted(set(len(X[i]) for i in ev))}},
           'status': 'RUNNING'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    def eval_scores(scores):
        f1s = [f1_at_keep(scores[i], Y[ev[i]], 0.80) for i in range(len(ev))]
        aps = [a for a in (ap_of(scores[i], Y[ev[i]]) for i in range(len(ev))) if a is not None]
        return float(np.mean(f1s)), float(np.mean(aps)), f1s

    # ---- champion replication ----
    ck = torch.load(args.champion, map_location='cpu', weights_only=False)
    champ = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2))))
    champ.load_state_dict(ck['state_dict'])
    champ.eval()
    with torch.no_grad():
        cs = [champ(torch.from_numpy(X[i])[None])[0].numpy().astype(np.float32)
              for i in ev]
    cf1, cap, cf1_list = eval_scores(cs)
    rec['champion'] = {'ckpt': str(args.champion), 'ch': ck.get('ch', 128),
                       'dils': list(ck.get('dils', (1, 2))),
                       'f1_keep080': round(cf1, 4), 'ap': round(cap, 4),
                       'anchor_r5': 0.6588}
    print(f"champion F1 {cf1:.4f} (R5 anchor 0.6588)", flush=True)

    # ---- slotprior (train-fitted position prior) ----
    prior = np.zeros(8, np.float64)
    cnt = np.zeros(8, np.float64)
    for i in tr:
        for s, v in enumerate(Y[i]):
            prior[s] += v
            cnt[s] += 1
    prior = (prior / np.maximum(cnt, 1)).astype(np.float32)
    ps = [prior[:len(Y[i])] for i in ev]
    pf1, pap, _ = eval_scores(ps)
    rec['slotprior'] = {'f1_keep080': round(pf1, 4), 'ap': round(pap, 4),
                        'prior': [round(float(x), 4) for x in prior],
                        'anchor_r5': 0.6625}
    print(f"slotprior F1 {pf1:.4f} (R5 anchor 0.6625)", flush=True)
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    # ---- training arms ----
    arms = ['mse'] if args.phase == 'pilot' else ['mse', 'exactdp']
    seeds = args.seeds[:1] if args.phase == 'pilot' else args.seeds
    rec['arms'] = {}
    for arm in arms:
        for sd in seeds:
            ts = time.time()
            torch.manual_seed(sd)
            rng = np.random.RandomState(sd)
            model = TCN()
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            n = len(tr)
            for step in range(args.steps):
                idx = rng.choice(n, args.batch)
                opt.zero_grad()
                losses = []
                for i in idx:
                    logits = model(torch.from_numpy(X[tr[i]])[None])[0]
                    y = torch.from_numpy(Y[tr[i]])
                    if arm == 'mse':
                        loss = ((torch.sigmoid(logits) - y) ** 2).mean()
                    else:
                        loss = -_expected_f_ref(logits, y)
                    losses.append(loss)
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                ss = [model(torch.from_numpy(X[i])[None])[0].numpy().astype(np.float32)
                      for i in ev]
            f1, apv, f1_list = eval_scores(ss)
            key = f'{arm}_s{sd}'
            rec['arms'][key] = {'f1_keep080': round(f1, 4), 'ap': round(apv, 4),
                                'f1_list': [round(v, 6) for v in f1_list],
                                'train_min': round((time.time() - ts) / 60, 1)}
            args.out.write_text(json.dumps(rec, indent=1) + '\n')
            print(f'{key}: F1 {f1:.4f} AP {apv:.4f}', flush=True)

    if args.phase == 'full':
        # arm-level 3-seed mean picks the judging arm (pre-recorded rule)
        means = {a: float(np.mean([rec['arms'][f'{a}_s{sd}']['f1_keep080']
                                   for sd in seeds])) for a in arms}
        best_arm = max(means, key=means.get)
        best_key = f'{best_arm}_s{seeds[0]}'
        # per-fragment deltas: mean over the arm's seeds (paired with champion)
        nf1 = np.mean([[rec['arms'][f'{best_arm}_s{sd}']['f1_list'][j]
                        for sd in seeds] for j in range(len(ev))], axis=0)
        delta = [nf1[j] - cf1_list[j] for j in range(len(ev))]
        ci = cluster_boot_paired(delta, [SRC[i] for i in ev], args.boot)
        dmean = float(np.mean(delta))
        gate_pass = bool(dmean >= 0.007 and ci[0] > 0)
        rec['gate'] = {'judging_arm': best_arm,
                       'arm_seed_means': {a: round(m, 4) for a, m in means.items()},
                       'delta_vs_champion': round(dmean, 5),
                       'ci95': ci,
                       'gate_threshold': 0.007,
                       'decision': ('PASS' if gate_pass else 'FAIL') +
                                   ' (r7 IV2_1B_TEMPORAL: temporal F +0.007 '
                                   'vs champion, source-cluster CI lower > 0)',
                       'slotprior_bar': round(pf1, 4),
                       'best_arm_clears_slotprior': bool(means[best_arm] > pf1)}
        args.out.write_text(json.dumps(rec, indent=1) + '\n')
        print('GATE', rec['gate']['decision'], flush=True)

    rec['status'] = 'DONE'
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
