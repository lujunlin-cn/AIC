"""R9 head training on the new IV2 Stage1 slot-pool features (R9 report Q3:
"same healthy TCN recipe, no new loss matrix, three seeds, re-initialised
projection; normalisation fitted on train sources only").

Arms (pre-registered before any feature was extracted):
  readouts = mean768 | m1408_l | m1408_m5   (from r9_iv2_slot_feats.py)
  losses   = mse | exactdp  (judging loss = exactdp, same choice rule as the
             R8 temporal gate: arm-level 3-seed mean picks the judging arm)
  seeds    = 20261006..08

Gate (R9 report 3.6, NEW - both must hold, each with source-cluster CI
lower > 0, paired per-fragment):
  delta vs champion (probe_deploy_head on SigLIP pool, frozen f1_list)
  delta vs macro-optimal position prior (audit_k6.per_fragment, frozen)
  each >= +0.007.  This gate is NOT retroactive; the historical gate
  (champion-only) is reported alongside.

Readout protocol decision is DEFERED to this script on purpose: all three
readouts are trained and judged under the identical recipe.
CPU only.  Incremental JSON after every arm.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r9_iv2_slot'))
ap.add_argument('--manifest-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--champion-gate', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/temporal_gate.json'))
ap.add_argument('--audit-k6', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit/audit_k6.per_fragment.jsonl'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r9_npu/slot_head.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seeds', nargs='*', type=int, default=[20261006, 20261007, 20261008])
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()


class TCN(torch.nn.Module):
    """Verbatim from r8_temporal_gate.py (champion family, dils (1,2,4))."""

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


def _expected_f_ref(logits, y):
    """Verbatim expected_f_binary from v11_expected_f_train.py."""
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


def f1_at_keep(scores, y, keep):
    n = len(y)
    k = max(1, int(round(keep * n)))
    idx = set(np.argsort(-scores)[:k].tolist())
    hit = sum(y[i] for i in idx)
    return float(2 * hit / (k + y.sum()))


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


def load_feats(readout):
    files = sorted(args.feat_root.glob('p*/*.npz'))
    seen, X, NMISS = set(), [], []
    for f in files:
        fid = f.stem
        if fid in seen:
            continue
        seen.add(fid)
        m = np.load(f)
        if readout not in m:
            raise SystemExit(f'{f} lacks {readout}')
        X.append(m[readout].astype(np.float32))
        NMISS.append(int(m['nmiss']))
    # frozen manifest order (labels/split live there)
    man = []
    for split in ('train_public', 'eval_public'):
        for l in (args.manifest_dir / f'{split}.jsonl').read_text().splitlines():
            if l.strip():
                man.append(json.loads(l))
    pos = {r['fragment_id']: i for i, r in enumerate(man)}
    order = []
    for r in man:
        fid = r['fragment_id']
        if fid not in pos:
            raise SystemExit(f'manifest frag {fid} has no feature npz')
        order.append(pos[fid])
    X = [X[i] for i in order]
    NMISS = [NMISS[i] for i in order]
    Y = [[int(v) for v in r['labels']] for r in man]
    SRC = [r['source_id'] for r in man]
    FID = [r['fragment_id'] for r in man]
    return X, Y, SRC, FID, NMISS


def main():
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec = {'protocol': 'R9 slot-head; TCN champion family; judging loss '
                       '= exactdp (arm-level 3-seed mean rule, as R8 gate); '
                       'gate = +0.007 vs champion AND vs macro-optimal '
                       'position prior, source-cluster CI lower > 0',
           'seeds': args.seeds, 'arms': {}, 'status': 'RUNNING'}
    args.out.write_text(json.dumps(rec, indent=1) + '\n')

    # frozen references
    gate_prev = json.loads(args.champion_gate.read_text())
    champ_f1 = gate_prev['champion']['f1_list']
    audit = {}
    for l in args.audit_k6.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            audit[r['fragment_id']] = r['macro_prior_f']
    print('frozen refs loaded '
          f'({time.time()-t0:.0f}s)', flush=True)

    summary = {}
    for readout in ('mean768', 'm1408_l', 'm1408_m5'):
        X, Y, SRC, FID, NMISS = load_feats(readout)
        ev_src = set(json.loads(args.eval_sources.read_text()))
        tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
        ev = [i for i in range(len(X)) if SRC[i] in ev_src]
        nmiss_ev = sum(1 for i in ev if NMISS[i] > 0)
        # train-fitted standardisation ONLY
        big = np.stack([X[i] for i in tr])                       # (n_tr, 8, D)
        mu = big.mean(axis=(0, 1)); sd = big.std(axis=(0, 1)) + 1e-6
        Xn = [(x - mu) / sd for x in X]
        D = X[0].shape[1]
        print(f'{readout}: D={D} train {len(tr)} eval {len(ev)} nmiss_ev {nmiss_ev}', flush=True)

        # macro-prior per-fragment F on eval, aligned by fragment id
        mac = np.array([audit[f] for f in (FID[i] for i in ev)], np.float64)
        champ = np.array(champ_f1, np.float64)
        assert len(mac) == len(ev) == len(champ), (len(mac), len(ev), len(champ))

        def eval_f1(model):
            ss = []
            with torch.no_grad():
                for i in ev:
                    lg = model(torch.from_numpy(Xn[i])[None])[0].numpy()
                    ss.append(lg.astype(np.float32))
            return np.array([f1_at_keep(ss[j], Y[ev[j]], 0.80)
                             for j in range(len(ev))], np.float64)

        arm_means = {}
        for loss in ('mse', 'exactdp'):
            for sd_ in args.seeds:
                key = f'{readout}:{loss}:s{sd_}'
                ts = time.time()
                torch.manual_seed(sd_)
                rng = np.random.RandomState(sd_)
                model = TCN(d_in=D)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                for step in range(args.steps):
                    idx = rng.choice(len(tr), args.batch)
                    opt.zero_grad()
                    losses = []
                    for i in idx:
                        logits = model(torch.from_numpy(Xn[tr[i]])[None])[0]
                        y = torch.from_numpy(np.array(Y[tr[i]], np.float32))
                        if loss == 'mse':
                            l_ = ((torch.sigmoid(logits) - y) ** 2).mean()
                        else:
                            l_ = -_expected_f_ref(logits, y)
                        losses.append(l_)
                    loss_ = torch.stack(losses).mean()
                    loss_.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                model.eval()
                f1v = eval_f1(model)
                rec['arms'][key] = {
                    'f1_mean': round(float(f1v.mean()), 4),
                    'f1_list': [round(v, 6) for v in f1v.tolist()],
                    'train_min': round((time.time() - ts) / 60, 1)}
                args.out.write_text(json.dumps(rec, indent=1) + '\n')
                print(f'{key}: F1 {f1v.mean():.4f}', flush=True)

        for loss in ('mse', 'exactdp'):
            arm_means[loss] = float(np.mean(
                [rec['arms'][f'{readout}:{loss}:s{s_}']['f1_mean'] for s_ in args.seeds]))
        judging = max(arm_means, key=arm_means.get)
        nf1 = np.mean([[rec['arms'][f'{readout}:{judging}:s{s_}']['f1_list'][j]
                        for s_ in args.seeds] for j in range(len(ev))], axis=0)
        d_c = (nf1 - champ).tolist()
        d_m = (nf1 - mac).tolist()
        ci_c = cluster_boot_paired(d_c, [SRC[i] for i in ev], args.boot)
        ci_m = cluster_boot_paired(d_m, [SRC[i] for i in ev], args.boot)
        dm_c, dm_m = float(np.mean(d_c)), float(np.mean(d_m))
        gate = bool(dm_c >= 0.007 and ci_c[0] > 0 and dm_m >= 0.007 and ci_m[0] > 0)
        summary[readout] = {
            'judging_loss': judging, 'arm_seed_means': {k: round(v, 4) for k, v in arm_means.items()},
            'delta_vs_champion': round(dm_c, 5), 'ci_champion': ci_c,
            'delta_vs_macroprior': round(dm_m, 5), 'ci_macroprior': ci_m,
            'gate_pass': gate, 'nmiss_eval_frags': nmiss_ev}
        rec['summary'] = summary
        args.out.write_text(json.dumps(rec, indent=1) + '\n')
        print(f"GATE {readout}: {'PASS' if gate else 'FAIL'} {summary[readout]}", flush=True)

    # offline nmiss sensitivity: expose the per-eval-frag missing count
    rec['eval_nmiss'] = {FID[i]: NMISS[i] for i in ev if NMISS[i] > 0}
    rec['status'] = 'DONE'
    rec['elapsed_min'] = round((time.time() - t0) / 60, 1)
    args.out.write_text(json.dumps(rec, indent=1) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
