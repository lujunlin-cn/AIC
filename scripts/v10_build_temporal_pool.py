"""Build the full PHD2 temporal pool over every teacher-annotated fragment.

The keep-mask ranking head currently trains on the 3,958 fragments whose GIF
intervals land inside an 8-frame window.  The teacher annotated 5,795, so about
a third of the labelled seconds in the pool are invisible to it.  This rebuilds
the pool from the fragment index plus the teacher's point files, keeping any
fragment whose teacher point maps onto the 1 s keyframe grid, and records the
in-interval label from the PHD2 selections (the official objective's family).

Two labels are written per fragment, because they answer different questions:

  in_interval  binary: does this second fall inside a GIF highlight interval?
                This is what the official F1 rewards, and what the mask
                ranking must learn.
  teacher      the normalised distance from the keyframe to the teacher's point
                along the clip, i.e. a soft preference target.  Kept separate,
                not blended in: the A1_raw arm that mixed a KD term into the GT
                stream scored below its no-KD parent on the official drop, so
                the two signals stay in separate experiments.

Split by SOURCE VIDEO with a fixed seed and written to val_sources.json so the
split cannot drift between runs - the same discipline the PHD2 spatial pool uses.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, required=True)
ap.add_argument('--points', type=Path, required=True, help='teacher point dir')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames'))
ap.add_argument('--ext', default='jpg')
ap.add_argument('--frames', type=int, default=0,
                help='0 = use each fragment\'s own keyframe count instead of a fixed 8. '
                     'An official clip is p50 14.1 s on a 1 s grid (11-15 frames); the '
                     'fixed 8-frame window both truncates long clips and labels whole '
                     'fragments positive when the interval covers them, which removes '
                     'exactly the ordering signal a keep mask needs.')
ap.add_argument('--val-frac', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=20261002)
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
args = ap.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
sel = json.loads(args.selections.read_text())

rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
srcs = sorted({r['src'] for r in rows})
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(srcs))
nv = max(1, int(len(srcs) * args.val_frac))
val_srcs = sorted(srcs[i] for i in perm[:nv])
val_set = set(val_srcs)

kept = 0
n_pos = 0
n_teacher = 0
pos_rate = []
rows_out = []
for r in rows:
    vid = r['video_id']
    pf = args.points / f'{vid}.json'
    if not pf.exists():
        continue
    src = r['src']
    t0 = float(r['t0'])
    L = float(r['L'])
    n = args.frames
    if n <= 0:
        # fragment's own keyframe count from the frame tree (the same frames the
        # feature pass pools); falls back to 8 when the tree is not built yet
        fd = args.frames_root / vid
        n = len(list(fd.glob(f'*.{args.ext}'))) if fd.exists() else 8
        if n <= 0:
            continue
    # label on the fragment-local 1 s grid
    ivs = []
    for recs in sel.get(src, {}).values():
        for rec in recs:
            a_, b_ = float(rec['t0']), float(rec['t1'])
            if b_ > a_:
                ivs.append((a_ - t0, b_ - t0))
    # Label on the ACTUAL keyframe timestamps, not the ideal grid: the feature
    # pass pools the frames that exist, so the label has to be indexed by the
    # same times or the two drift by up to half a step.
    kdir = args.frames_root / vid
    if kdir.exists():
        stamps = sorted(float(p.stem) for p in kdir.glob(f'*.{args.ext}'))
    else:
        stamps = [t0 + L * j / n for j in range(n)]
    kt = np.array(stamps, np.float32)
    n = len(kt)
    if n < 4:
        continue
    y = np.zeros(n, np.float32)
    step = float(np.median(np.diff(kt))) if n > 1 else 1.0
    half = step * 0.5
    for a_, b_ in ivs:
        # frame CENTRE inside the interval; a GIF interval clipping a fragment
        # edge must still label that frame
        y[(kt + half >= a_) & (kt - half < b_)] = 1.0
    if y.sum() == 0 or y.sum() == n:
        # all-positive fragments carry no ordering signal (every second is a
        # keep), so they are dropped rather than diluting the ranking loss
        continue
    # teacher soft target: inverse normalised distance to the teacher point
    try:
        d = json.loads(pf.read_text())
        pt = None
        for ratio, blk in (d.get('ratios') or {}).items():
            pts = blk.get('points') or []
            if pts:
                pt = pts[len(pts) // 2]
                break
    except Exception:
        pt = None
    teach = np.full(n, 0.5, np.float32)
    if pt is not None and len(pt) >= 2:
        n_teacher += 1
        c = float(np.clip((float(pt[0]) + float(pt[1])) / 2.0, 0.0, 1.0))
        teach = np.exp(-np.abs(np.linspace(0, 1, n) - c) * 4.0).astype(np.float32)
    np.savez_compressed(args.out / f'{vid}.npz', t=kt, y=y, teacher=teach)
    rows_out.append({'video_id': vid, 'src': src, 'split': 'val' if src in val_set else 'train',
                     'pos': int(y.sum()), 'n': n})
    kept += 1
    pos_rate.append(float(y.mean()))
    if y.sum() > 0:
        n_pos += 1

(args.out / 'val_sources.json').write_text(json.dumps(val_srcs))
with (args.out / 'index.jsonl').open('w') as fh:
    for r in rows_out:
        fh.write(json.dumps(r) + '\n')
pos = np.array(pos_rate) if pos_rate else np.zeros(1)
print(f'TEMPORAL_POOL kept={kept} fragments ({n_pos} with positives) '
      f'sources={len(srcs)} val_sources={len(val_srcs)} '
      f'pos_rate mean={pos.mean():.3f} p50={np.percentile(pos,50):.3f} '
      f'teacher_points_used={n_teacher} -> {args.out}', flush=True)