"""R6 S-REREAD step 3: pool freeze - train240 / dev80 / confirm=val178.

Preregistered (reports/r6/preregistration.yaml, seed 20261006):
  train240 : subset of the release train split (1115 = the B3 training
             ancestry) - u labels come from the B3 sample cache
             (live_train.npz, annotator-mean candidate IoU, SAME objective).
  dev80    : subset of the 156 train_index sources that the B3 NEVER
             trained on and no round ever read (1622 - 1115 - 351) -
             u labels computed here from the sparse annotations with the
             SAME candidate geometry (129 legal max-windows, mean IoU).
  confirm  : val178 (holdout reserve) - NOT built in this step; encoded
             only after the dev gate, one read.

Per video: up to 8 labeled instants (keyframe counters); shortlist =
B3 top-3 + g33 grid (amended, gate PASS).  Writes the freeze manifest:
which video contributes which frames, and (train only) the row indices
into the B3 cache.  dev80 grid features do NOT exist yet - extraction is
the next step; this script only freezes IDs and counts.

CPU only.  Output: r6_s_pool_freeze.json (+ csv)
"""
import argparse, csv, hashlib, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=20261006)
ap.add_argument('--n-train', type=int, default=240)
ap.add_argument('--n-dev', type=int, default=80)
ap.add_argument('--frames-per-vid', type=int, default=8)
ap.add_argument('--release', default='/data/aic/external_datasets/_releases/spatial_crop_v1')
ap.add_argument('--samples-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V8/samples'))
ap.add_argument('--t5-train-index', type=Path,
                default=Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl'))
ap.add_argument('--sparse-root', type=Path,
                default=Path('/data/aic/external_datasets/LIVE_YT_VC/annotations/sparse'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze.json'))
args = ap.parse_args()

sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}
GRID33 = list(range(0, 129, 4))


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def candidate_boxes(W, H, ratio_wh, nc=129):
    w, h, axis = geometry(W, H, ratio_wh)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    boxes = np.zeros((len(offs), 4), np.float32)
    for j, o in enumerate(offs):
        if axis == 0:
            boxes[j] = [o, 0, o + w, h]
        elif axis == 1:
            boxes[j] = [0, o, w, o + h]
        else:
            boxes[j] = [0, 0, W, H]
    return boxes


def main():
    rng = np.random.RandomState(args.seed)
    # --- pools ---
    tr_npz = np.load(args.samples_dir / 'live_train.npz', allow_pickle=True)
    tr_vid = tr_npz['vid'].astype(str)
    train_1115 = sorted(set(tr_vid.tolist()))
    t5 = [json.loads(l) for l in args.t5_train_index.read_text().splitlines()]
    train_index_vids = {r['video_id'] for r in t5}
    exposed351 = set()
    for tag in ('live_dev', 'live_confirmation'):
        d = np.load(args.samples_dir / f'{tag}.npz', allow_pickle=True)
        exposed351 |= set(d['vid'].astype(str).tolist())
    never_used = sorted(train_index_vids - set(train_1115) - exposed351)
    print(f'train1115 {len(train_1115)}  exposed351 {len(exposed351)}  '
          f'never-used {len(never_used)}', flush=True)
    assert len(never_used) >= args.n_dev, 'never-used pool too small'
    train240 = sorted(rng.choice(train_1115, args.n_train, replace=False).tolist())
    dev80 = sorted(rng.choice(never_used, args.n_dev, replace=False).tolist())
    assert not (set(train240) & set(dev80)) and not (set(dev80) & exposed351)

    # --- per-video instants ---
    packed_root = Path(args.release) / 'packed' / 'LIVE_YT_VC'
    rows, wman = [], {}
    # train: read u directly from the cache rows
    tr_u = tr_npz['u'].astype(np.float32)
    tr_frame = tr_npz['frame'].astype(int)
    tr_ratio = tr_npz['ratio'].astype(str)
    per_vid_rows = {}
    for i, v in enumerate(tr_vid.tolist()):
        per_vid_rows.setdefault(v, []).append(i)
    for v in train240:
        idxs = per_vid_rows[v]
        # pick up to 8 instants spread over the cached rows (sorted by frame)
        idxs = sorted(idxs, key=lambda i: tr_frame[i])
        if len(idxs) > args.frames_per_vid:
            pick = np.linspace(0, len(idxs) - 1, args.frames_per_vid).round().astype(int)
            idxs = [idxs[p] for p in dict.fromkeys(pick.tolist())]
        for i in idxs:
            rows.append({'pool': 'train240', 'vid': v, 'frame': int(tr_frame[i]),
                         'ratio': str(tr_ratio[i]), 'cache_row': int(i),
                         'u_path': 'live_train.npz'})
    # dev: compute u from sparse annotations, pick up to 8 packed frames
    n_skipped_gt = 0
    for v in dev80:
        t5r = next(r for r in t5 if r['video_id'] == v)
        W, H = float(t5r['width']), float(t5r['height'])
        ratio = t5r['targetRatioWH']
        ratio_key = f'{ratio[0]}-{ratio[1]}'
        ann = json.loads(Path(t5r['annotation']).read_text())
        gt = {int(f): np.array(b, np.float32)
              for f, b in zip(ann['frames'], ann['boxes_xywh'])}
        pk = np.load(packed_root / f'{v}.npz')
        fidx = sorted(int(f) for f in pk['frame_idx'].astype(int))
        with_gt = [f for f in fidx if f in gt]
        if len(with_gt) < 1:
            n_skipped_gt += 1
            continue
        if len(with_gt) > args.frames_per_vid:
            pick = np.linspace(0, len(with_gt) - 1, args.frames_per_vid).round().astype(int)
            with_gt = [with_gt[p] for p in dict.fromkeys(pick.tolist())]
        boxes = candidate_boxes(W, H, ratio)
        for f in with_gt:
            g = gt[f]
            u = iou(boxes[:, None, :], g[None, None, :]).reshape(-1) \
                if g.ndim == 1 else iou(boxes[:, None, :], g[None]).mean(1)
            rows.append({'pool': 'dev80', 'vid': v, 'frame': int(f),
                         'ratio': ratio_key, 'cache_row': -1,
                         'u': [round(float(x), 5) for x in u.tolist()]})
    print(f'rows: {len(rows)}; dev vids without GT frames skipped: {n_skipped_gt}',
          flush=True)
    out = {
        'protocol': 'R6 S-REREAD pool freeze; seed 20261006; train240 from the '
                    'B3 ancestry (release train split), dev80 from the 156 '
                    'never-used train_index sources, confirm=val178 untouched; '
                    'shortlist top3+g33 to be encoded next; 8 instants per video',
        'seed': args.seed,
        'pool_sizes': {'train240': len(set(r['vid'] for r in rows if r['pool'] == 'train240')),
                       'dev80': len(set(r['vid'] for r in rows if r['pool'] == 'dev80')),
                       'confirm': 178},
        'n_rows': len(rows),
        'dev_never_used_pool': len(never_used),
        'head_sha16': sha16('/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt'),
        'csv': args.out.with_name(args.out.stem + '_rows.csv').as_posix(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    with open(args.out.with_name(args.out.stem + '_rows.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pool', 'vid', 'frame', 'ratio',
                                          'cache_row', 'u_path', 'u'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
