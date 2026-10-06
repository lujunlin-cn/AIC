"""R9 offline head analysis (post-run, CPU minutes): for EVERY trained arm in
slot_head{,_s2,_qwen}.json recompute eval AP from the stored f1_list runs?
f1_list has no scores, so instead: (1) nmiss sensitivity - recompute the
judging-arm F1 delta vs champion with nmiss>0 eval frags EXCLUDED, from the
stored per-frag f1_list + the recorded eval_nmiss map; (2) champion AP vs
champion f1 reference is already known.  AP of trained heads would need a
re-forward; skip it here (arm logits were not cached) and note that.

Output: one JSON with, per readout per file, the paired delta and CI on the
nmiss-clean subset.  If the FAIL verdicts hold on the clean subset, the
65-missing-frame tail is not driving the result.
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS',):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--npu-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r9_npu'))
ap.add_argument('--out', type=Path, default=Path('/data/aic/experiments_910a/LFM_V11/r9_npu/head_offline.json'))
args = ap.parse_args()


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
    res = {}
    for name in ('slot_head.json', 'slot_head_s2.json', 'slot_head_qwen.json'):
        p = args.npu_dir / name
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        summ = r.get('summary', {})
        if not summ:
            continue
        nmiss = {k: int(v) for k, v in (r.get('eval_nmiss') or {}).items()}
        # champion f1_list: from the gate json (aligned to eval_public order);
        # rebuild marker: slot_head files only store arms + summary, so reload
        # champion from the R8 gate or the audit export - use the gate
        gate = json.loads(Path('/data/aic/experiments_910a/LFM_V11/r8_npu/temporal_gate.json').read_text())
        if 'f1_list' in gate.get('champion', {}):
            champ = np.array(gate['champion']['f1_list'], np.float64)
        else:
            ev_rows = [json.loads(l) for l in
                       Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit/eval_public.jsonl').read_text().splitlines() if l.strip()]
            champ = np.array([
                (lambda s, y: 2 * float(np.asarray(y)[list(set(np.argsort(-np.asarray(s))[:6].tolist()))].sum())
                 / (6 + float(np.sum(y))))(row['scores'], row['labels'])
                for row in ev_rows], np.float64)
        # eval src per index comes from the manifest (frozen order)
        man_ev = [json.loads(l) for l in
                  Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit/eval_public.jsonl').read_text().splitlines() if l.strip()]
        srcs = [r_['source_id'] for r_ in man_ev]
        clean = [i for i in range(len(man_ev)) if nmiss.get(man_ev[i]['fragment_id'], 0) == 0]
        res[name] = {'n_eval': len(man_ev), 'n_clean': len(clean)}
        for readout, v in summ.items():
            seeds = [k.rsplit(':s', 1)[1] for k in r['arms'] if k.startswith(f'{readout}:')]
            judging = v['judging_loss']
            rows = np.array([[r['arms'][f'{readout}:{judging}:s{s_}']['f1_list'][j]
                              for s_ in seeds] for j in range(len(man_ev))], np.float64)
            nf1 = rows.mean(axis=1)
            d_all = (nf1 - champ).mean()
            d_cl = (nf1[clean] - champ[clean]).mean()
            ci_cl = cluster_boot_paired((nf1[clean] - champ[clean]).tolist(),
                                        [srcs[i] for i in clean], 2000)
            res[name][readout] = {'delta_all': round(float(d_all), 5),
                                  'delta_nmiss_clean': round(float(d_cl), 5),
                                  'ci_clean': ci_cl,
                                  'verdict_stable': bool(d_cl < 0 and ci_cl[0] < 0)}
            print(name, readout, f"all {d_all:+.5f} clean {d_cl:+.5f} CI {ci_cl}", flush=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print('WROTE', args.out, f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
