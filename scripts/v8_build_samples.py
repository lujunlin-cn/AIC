"""V8: pre-build candidate-utility training samples (features + labels) once,
so the arm x seed training grid reads cached arrays instead of re-building.

One .npz per split tag in --output-dir:
  feat (N,NC,2305) fp16, u (N,NC) f32, vid/frame/ratio metadata.
Rows are taken from the manifest, filtered to rows whose feature file exists
(keyframes only, keeping RV strictly comparable with V7).
"""
import argparse, json, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
ap.add_argument('--feat-roots', nargs='*', default=[
    'rv_native=/data/aic/experiments_910a/LFM_V7/feats',
    'rv_rot=/data/aic/experiments_910a/LFM_V8/rv_feats_rot',
    'live=/data/aic/experiments_910a/LFM_V8/live_feats'])
ap.add_argument('--n-cand', type=int, default=129)
ap.add_argument('--workers', type=int, default=24)
ap.add_argument('--tags', nargs='*', default=None, help='only build these split tags')
ap.add_argument('--output-dir', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
args = ap.parse_args()

import numpy as np  # noqa: E402
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

FEATS = {}
for spec in args.feat_roots:
    k, v = spec.split('=', 1)
    FEATS[k] = Path(v)
RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}
NC = args.n_cand


def feat_path(src, vid, kf):
    root = FEATS['live'] if src.startswith('live') else FEATS[src]
    return root / vid / f'{kf}.npz'


rows = [json.loads(l) for l in open(args.manifest)]
rows = [r for r in rows if feat_path(r['src'], r['vid'], r['frame']).exists()]
print(f'{len(rows)} rows with feats', flush=True)


def build(row):
    f = np.load(feat_path(row['src'], row['vid'], row['frame']))
    grid = f['grid'].astype(np.float32)
    fh, fw, D = grid.shape
    W, H = float(row['W']), float(row['H'])
    w, h, axis = geometry(W, H, RATIOS[row['ratio']])
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    win_px = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win_px[j] = [o, 0, o + w, h]
        elif axis == 1:
            win_px[j] = [0, o, w, o + h]
        else:
            win_px[j] = [0, 0, W, H]
    flat = grid.reshape(-1, D)
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    m = np.zeros((len(offs), fh * fw), dtype=np.float32)
    for j, (x1, y1, x2, y2) in enumerate(win_px):
        cx1, cx2 = int(math.floor(x1 / px_per[0])), int(math.ceil(x2 / px_per[0]))
        cy1, cy2 = int(math.floor(y1 / px_per[1])), int(math.ceil(y2 / px_per[1]))
        cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
        m[j] = ((gx >= cx1) & (gx < cx2) & (gy >= cy1) & (gy < cy2)).reshape(-1).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    winp = (m @ flat) / ms
    outp = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    pos = (offs / span if span > 0 else offs).astype(np.float32)
    feat = np.concatenate([winp, outp, winp - outp, pos[:, None]], 1)
    gt = np.array(row['gt'], dtype=np.float32)
    u = iou(win_px[:, None, :], gt[None, :, :]).mean(1)
    return feat.astype(np.float16), u.astype(np.float32)


# V8 fix: group by (split, src), not split alone.  The manifest stores the
# rotated-augmentation rows under the SAME split names as the native ones
# (distinguished only by src='rv_rot'), so grouping by split alone mixed the two
# geometries into one tag - which is why B2 (rv+rotation) silently trained on the
# same data as B1 and produced bit-identical results.
def tag_of(row):
    if row['src'] != 'rv_rot':
        return row['split']
    # split names already start with 'rv_' (rv_train / rv_dev / rv_diag), so
    # prefixing blindly yields 'rv_rot_rv_train' and --tags never matches.
    return 'rv_rot_' + row['split'].split('_', 1)[1] if '_' in row['split'] else 'rv_rot_' + row['split']


groups = {}
for r in rows:
    groups.setdefault(tag_of(r), []).append(r)

args.output_dir.mkdir(parents=True, exist_ok=True)
CHUNK = 512  # rows materialised at once; bounds worker IPC and parent RSS
for tag, rs in sorted(groups.items()):
    if args.tags and tag not in args.tags:
        continue
    if not rs:
        print(tag, 'empty (features not extracted yet), skip', flush=True)
        continue
    # V8 memory fix: write memmap-able .npy sidecars instead of a single .npz.
    # The old path did `res = list(ex.map(build, rs))` and then np.stack, i.e.
    # two full copies of the pool in RAM (rv_train ~30 GB, live_train ~40 GB)
    # on top of 24 workers' IPC buffers.  Now the output array is preallocated as
    # a memmap on disk and filled chunk by chunk, so parent RSS stays ~CHUNK
    # rows and the trainers can mmap it read-only.
    if (args.output_dir / f'{tag}_feat.npy').exists() and (args.output_dir / f'{tag}_u.npy').exists():
        print(tag, 'npy cache exists, skip', flush=True)
        continue
    first = build(rs[0])
    D = first[0].shape[-1]
    feat_mm = np.lib.format.open_memmap(args.output_dir / f'{tag}_feat.npy', mode='w+',
                                         dtype=np.float16, shape=(len(rs), first[0].shape[0], D))
    u_mm = np.lib.format.open_memmap(args.output_dir / f'{tag}_u.npy', mode='w+',
                                     dtype=np.float32, shape=(len(rs), first[0].shape[0]))
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for s in range(0, len(rs), CHUNK):
            block = rs[s:s + CHUNK]
            res = list(ex.map(build, block, chunksize=8))
            feat_mm[s:s + len(res)] = np.stack([x[0] for x in res])
            u_mm[s:s + len(res)] = np.stack([x[1] for x in res])
            done += len(res)
            print(f'{tag} {done}/{len(rs)}', flush=True)
    feat_mm.flush(); u_mm.flush()
    del feat_mm, u_mm
    np.save(args.output_dir / f'{tag}_vid.npy', np.array([r['vid'] for r in rs]))
    np.save(args.output_dir / f'{tag}_frame.npy', np.array([r['frame'] for r in rs]))
    np.save(args.output_dir / f'{tag}_ratio.npy', np.array([r['ratio'] for r in rs]))
    (args.output_dir / f'{tag}_meta.json').write_text(json.dumps({'n': len(rs), 'nc': first[0].shape[0], 'd': D}) + '\n')
    print(tag, len(rs), 'written (npy memmap)', flush=True)
print('ALL DONE', flush=True)
