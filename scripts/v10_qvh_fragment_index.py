"""Cut QVHighlights 150 s clips into official-shaped 8-14 s fragments.

QV is the auxiliary temporal domain - it contributes dense annotator-mean
saliency (0..1 per second) that PHD2 lacks.  But it arrives as fixed 150 s
clips while the official drop is p50 14.1 s.  Training on 150 s sequences
teaches a highlight-density prior that is wrong for short clips, so we cut
fragments on the SAME recipe as v9_phd2_fragments.py and carry the saliency
sub-range into each fragment's label.

Anchor policy mirrors PHD2: 65 % of fragments are centred on a relevant_window
(the annotated highlight), 35 % land on a random offset - the same positive /
context mix the PHD2 pool used, so a temporal head trained here sees the same
class balance it saw there.

Output per fragment  <out>/index.jsonl row:
    video_id (frag id = <vid>_qNN), src vid, t0, L (seconds within the 150 s
    clip), n_annot_mean (annotator coverage inside the fragment)
And <out>/frag_labels/<frag>.npz:
    t (s within fragment), saliency, soft window label, n_annot
all on the 1 Hz grid, so v10_qvh_tcn.py consumes them identically to the
full-clip labels.
"""
import argparse, json, os, random
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--annotations', type=Path, required=True)
ap.add_argument('--label-root', type=Path, required=True, help='full-clip labels from v10_qvh_labels.py')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--n-fragments', type=int, default=9000)
ap.add_argument('--clip-min', type=float, default=8.0)
ap.add_argument('--clip-max', type=float, default=14.0)
ap.add_argument('--anchor-frac', type=float, default=0.65)
ap.add_argument('--max-per-src', type=int, default=2)
ap.add_argument('--seed', type=int, default=20261002)
args = ap.parse_args()

rng = random.Random(args.seed)
(args.out / 'frag_labels').mkdir(parents=True, exist_ok=True)

rows = {}
for line in args.annotations.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows[r['vid']] = r

vids = sorted(rows)
rng.shuffle(vids)
out_rows = []
per_src = {}
fi = 0
for vid in vids:
    r = rows[vid]
    dur = float(r['duration'])
    wins = [w for w in (r.get('relevant_windows') or []) if len(w) == 2 and w[1] > w[0]]
    lf = args.label_root / f'{vid}.npz'
    if not lf.exists():
        continue
    L = np.load(lf)
    sal_full, soft_full, nann_full = L['saliency'], L['soft'], L['n_annot']
    tgrid = np.arange(len(sal_full), dtype=np.float32) * 1.0 + 0.5

    for _ in range(args.max_per_src):
        if len(out_rows) >= args.n_fragments:
            break
        cl = rng.uniform(args.clip_min, args.clip_max)
        cl = min(cl, dur)
        if rng.random() < args.anchor_frac and wins:
            a_, b_ = wins[rng.randrange(len(wins))]
            c = (a_ + b_) / 2.0
            t0 = min(max(c - rng.uniform(0.0, 0.6) * cl, 0.0), max(dur - cl, 0.0))
        else:
            t0 = rng.uniform(0.0, max(dur - cl, 0.0))
        t1 = t0 + cl
        # slice the 1 Hz labels to [t0, t1)
        sel = (tgrid >= t0) & (tgrid < t1)
        if sel.sum() < 4:
            continue
        frag = f'{vid}_q{fi:05d}'
        fi += 1
        per_src[vid] = per_src.get(vid, 0) + 1
        np.savez_compressed(
            args.out / 'frag_labels' / f'{frag}.npz',
            t=(tgrid[sel] - t0).astype(np.float32),
            saliency=sal_full[sel],
            soft=soft_full[sel],
            n_annot=nann_full[sel])
        out_rows.append({'video_id': frag, 'src': vid, 't0': round(t0, 3),
                         'L': round(cl, 3), 'duration': dur,
                         'n_annot_mean': float(nann_full[sel].mean())})
    if len(out_rows) >= args.n_fragments:
        break

with (args.out / 'index.jsonl').open('w') as fh:
    for r in out_rows:
        fh.write(json.dumps(r) + '\n')

pos = np.mean([np.load(args.out/'frag_labels'/f"{r['video_id']}.npz")['soft'].max() > 0.9
               for r in out_rows[:500]])
print(f'QV_FRAGS n={len(out_rows)} anchored={args.anchor_frac} '
      f'mean_L={np.mean([r["L"] for r in out_rows]):.1f}s '
      f'frag_has_highlight(soft>0.9)~{pos:.2f} -> {args.out}', flush=True)
