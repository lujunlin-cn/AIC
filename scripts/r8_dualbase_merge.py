"""R8 dual-base verdict: source-cluster bootstrap on the two dual-base runs.

Reads the two per-base JSONs (preds keyed by WINDOW NAME), re-derives the
dev labels and source map from the frag pool (identical loader), and
computes the preregistered verdict:
  pooled Spearman per base; A-B paired delta with SOURCE-cluster bootstrap
  (resample dev sources with replacement, 2000 draws, same draw applied to
  both bases); lower>0 -> old base (A); upper<0 -> IV2 (B); crossing ->
  equivalent, default to the higher pooled mean.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--a-json', required=True)
ap.add_argument('--b-json', required=True)
ap.add_argument('--frag-root', default='/data/aic/experiments_910a/QVH_V10/frag_train')
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', required=True)
args = ap.parse_args()


def soft_at(t, gt_t, gt_soft):
    i = np.clip(np.searchsorted(gt_t, t) - 1, 0, len(gt_t) - 2)
    w = np.clip((t - gt_t[i]) / np.maximum(gt_t[i + 1] - gt_t[i], 1e-6), 0, 1)
    return gt_soft[i] * (1 - w) + gt_soft[i + 1] * w


A = json.loads(Path(args.a_json).read_text())
B = json.loads(Path(args.b_json).read_text())
preds = {}
for tag, R in (('A', A), ('B', B)):
    for e in R['per_seed']:
        preds.setdefault(e['seed'], {})[tag] = e['pred']
seeds = sorted(preds)
winset = set(preds[seeds[0]]['A'])
assert winset == set(preds[seeds[0]]['B']), 'window sets differ across bases'

# labels + sources by window name (loader identical to the training script:
# mean saliency over the window timestamps stored in the feature npz)
G = {win: win.rsplit('_q', 1)[0] for win in winset}
FEAT_A = A['feat']
Y = {}
for win in winset:
    fps = glob.glob(f'{FEAT_A}/{win}.npz') + glob.glob(f'{FEAT_A}/p*/{win}.npz')
    z = np.load(fps[0])
    lz = np.load(Path(args.frag_root) / 'frag_labels' / f'{win}.npz')
    Y[win] = float(np.mean(soft_at(z['t'].astype(np.float32),
                                   lz['t'].astype(np.float32),
                                   lz['saliency'].astype(np.float32))))

wins = sorted(winset)
widx = {w: k for k, w in enumerate(wins)}
srcs = sorted(set(G[w] for w in wins))
src_wins = {g: [w for w in wins if G[w] == g] for g in srcs}

# mean-pooled pred per seed per base
def pooled_spearman(s, y):
    if np.std(s) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    rs = np.argsort(np.argsort(s)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rs = (rs - rs.mean()) / max(rs.std(), 1e-9)
    ry = (ry - ry.mean()) / max(ry.std(), 1e-9)
    return float((rs * ry).mean())


out = {'protocol': 'dual-base verdict; source-cluster bootstrap; preregistered '
                   'rule: lower>0 -> old(A); upper<0 -> IV2(B); crossing -> '
                   'equivalent, default higher pooled mean',
       'windows': len(wins), 'dev_srcs': len(srcs), 'seeds': seeds,
       'checks': {'window_sets_equal': True}}
for tag, R in (('A', A), ('B', B)):
    ps = []
    for sd in seeds:
        s = np.array([preds[sd][tag][w] for w in wins])
        y = np.array([Y[w] for w in wins])
        ps.append(pooled_spearman(s, y))
    out[f'{tag}_pooled_per_seed'] = [round(x, 5) for x in ps]
    out[f'{tag}_pooled_mean'] = round(float(np.mean(ps)), 5)

# paired cluster bootstrap over seed-mean preds
sa = np.mean([[preds[sd]['A'][w] for w in wins] for sd in seeds], axis=0)
sb = np.mean([[preds[sd]['B'][w] for w in wins] for sd in seeds], axis=0)
y = np.array([Y[w] for w in wins])
rng = np.random.RandomState(99)
deltas = []
for _ in range(args.boot):
    pick = rng.choice(len(srcs), len(srcs), replace=True)
    idx = [widx[w] for g in pick for w in src_wins[srcs[g]]]
    ii = np.array(idx)
    da = pooled_spearman(sa[ii], y[ii])
    db = pooled_spearman(sb[ii], y[ii])
    deltas.append(da - db)
lo, hi = np.percentile(deltas, [2.5, 97.5])
out['A_minus_B'] = {'delta': round(float(np.mean(deltas)), 5),
                    'ci95': [round(float(lo), 5), round(float(hi), 5)]}
d, ci = out['A_minus_B']['delta'], out['A_minus_B']['ci95']
if ci[0] > 0:
    verdict = 'OLD_BASE (A)'
elif ci[1] < 0:
    verdict = 'IV2_BASE (B)'
else:
    verdict = ('EQUIVALENT -> default OLD_BASE (A)'
               if out['A_pooled_mean'] >= out['B_pooled_mean']
               else 'EQUIVALENT -> default IV2_BASE (B)')
out['verdict'] = verdict
Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
