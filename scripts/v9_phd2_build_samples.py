"""Build V8-format candidate-utility samples for the PHD2 fragment pool.

Two things make PHD2 different from the RetargetVid / LIVE pools:

1. There is no human crop GT.  The teacher point is converted into a target
   window - the largest legal window centred on the teacher point along the
   sliding axis - and the per-candidate utility u is the IoU against that
   window.  So PHD2 rows can only enter training through a KD term, never as
   supervised GT.  Rows whose teacher reply failed to parse are dropped rather
   than silently turned into a centre-point target (that failure mode is what
   makes a pseudo-label set teach the model to sit in the middle).

2. The official drop is 28% portrait-source -> 16:9, i.e. y-axis crops out of
   portrait frames.  The rotated fragments (`*_r`, 90 deg CW) are the only
   source of that geometry in the whole project: rotating a landscape frame
   swaps the sliding axis, so V8's rotation produced portrait sources with x-axis
   windows, never portrait sources with y-axis windows.  Here a rotated
   fragment keeps the official target mix, so a rotated fragment with a 16:9
   target is exactly "portrait source, y-axis crop".

Splits are by SOURCE VIDEO (never by fragment) and stratified by the geometry
class, so `phd2_val` mirrors the official task composition and stays a fresh
confirmation pool.
"""
import argparse, json, math, os

# BLAS defaults to one thread per core: 8 workers were each grabbing 24 cores,
# driving the 192-core host to load ~475 and stretching the concurrently running
# 32B teacher's per-batch wall time from 12 s to 33 s.  Pin to one thread per
# worker BEFORE numpy is imported and scale the pool instead.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True)
ap.add_argument('--index-name', default='index_v2.jsonl')
ap.add_argument('--feat-root', type=Path, required=True)
ap.add_argument('--points', type=Path, required=True)
ap.add_argument('--output-dir', type=Path, required=True)
ap.add_argument('--n-cand', type=int, default=129)
ap.add_argument('--workers', type=int, default=24)
ap.add_argument('--val-frac', type=float, default=0.035)
ap.add_argument('--val-sources', type=Path, default=None,
                help='JSON list of held-out source videos. Written on first run '
                     'and reused afterwards: without this the val split drifts '
                     'every time the teacher finishes more fragments, and no two '
                     'runs are comparable.')
ap.add_argument('--seed', type=int, default=20261001)
ap.add_argument('--chunk', type=int, default=512)
ap.add_argument('--target', choices=['gauss', 'iou'], default='gauss',
                help='teacher point -> candidate target; gauss matches the validated '
                     'V8 KD-v2 soft preference, iou is the hard argmax window')
ap.add_argument('--kd-sigma', type=float, default=1.0 / 16,
                help='gaussian width as a fraction of the sliding span')
ap.add_argument('--segments', type=int, default=0,
                help='split each candidate window into K bands along the axis '
                     'PERPENDICULAR to the sliding axis and pool each separately. '
                     'The existing feature set pools the whole window, which '
                     'discards where inside the window the subject sits; for a '
                     'portrait window sliding on x, knowing whether the subject '
                     'is in the top, middle or bottom third is exactly what the '
                     'candidate head cannot recover from win/out/diff. These band '
                     'features are identical across candidates, so they only widen '
                     'the input. 0 = off (keeps the 2305-d V7/V8 layout).')
ap.add_argument('--smooth', type=int, default=0,
                help='temporal smoothing of the teacher point along the fragment '
                     'before the target is derived. The audit measured a p90 frame-'
                     'to-frame jump of 0.25-0.30 in normalised coordinates on 1 s '
                     'keyframes, i.e. the teacher is jittery where it should be '
                     'smooth; V8 KD-v2 gained only +0.003/+0.007, consistent with '
                     'the student averaging that noise instead of the signal. '
                     '1 = 3-point moving average, 2 = 5-point, and so on.')
args = ap.parse_args()

import numpy as np  # noqa: E402
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

NC = args.n_cand


def geom_class(r):
    """Mirror the official drop's task classes."""
    if r.get('rotated'):
        return 'rot_portrait'          # portrait source by construction
    ar = r['W'] / r['H']
    if ar < 0.75:
        return 'portrait_native'
    if ar > 1.5:
        return 'landscape'
    return 'square'


def windows(W, H, rw, rh):
    w, h, axis = geometry(W, H, (rw, rh))
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, NC) if span > 0 else np.zeros(1)
    win = np.zeros((len(offs), 4))
    for j, o in enumerate(offs):
        if axis == 0:
            win[j] = [o, 0, o + w, h]
        elif axis == 1:
            win[j] = [0, o, w, o + h]
        else:
            win[j] = [0, 0, W, H]
    return win, axis, span


def build(row):
    """One row -> (feat (NC,2305) fp16, u (NC,) f32)."""
    f = np.load(args.feat_root / row['vid'] / f"{row['kf']}.npz")
    grid = f['grid'].astype(np.float32)
    fh, fw, D = grid.shape
    W, H = float(row['W']), float(row['H'])
    win, axis, span = windows(W, H, row['rw'], row['rh'])
    flat = grid.reshape(-1, D)
    px_per = np.array([W / fw, H / fh])
    gy, gx = np.mgrid[0:fh, 0:fw]
    # Broadcast instead of a 129-iteration Python loop: the loop version made this
    # stage CPU-bound (192-core host driven to load ~900, which starved the NPU
    # teacher running alongside it - teacher wall time went 12 s -> 33 s per
    # batch).  Building all 129 masks in one shot is ~100x fewer numpy calls.
    cx1 = np.floor(win[:, 0] / px_per[0]).astype(int)
    cx2 = np.maximum(np.ceil(win[:, 2] / px_per[0]).astype(int), cx1 + 1)
    cy1 = np.floor(win[:, 1] / px_per[1]).astype(int)
    cy2 = np.maximum(np.ceil(win[:, 3] / px_per[1]).astype(int), cy1 + 1)
    gx1, gy1 = gx[0, :], gy[:, 0]          # mgrid gives (fh, fw) views
    m = ((gx1[None, None, :] >= cx1[:, None, None]) & (gx1[None, None, :] < cx2[:, None, None]) &
         (gy1[None, :, None] >= cy1[:, None, None]) & (gy1[None, :, None] < cy2[:, None, None]))
    m = m.reshape(len(win), fh * fw).astype(np.float32)
    ms = m.sum(1, keepdims=True)
    winp = (m @ flat) / ms
    outp = ((1 - m) @ flat) / np.maximum((1 - m).sum(1, keepdims=True), 1)
    pos = (np.arange(len(win)) / max(len(win) - 1, 1)).astype(np.float32) if span > 0 \
        else np.zeros(len(win), dtype=np.float32)
    parts = [winp, outp, winp - outp, pos[:, None]]
    if args.segments > 1 and axis is not None:
        # Bands along the axis the window CANNOT slide along.  sel must stay a
        # flat (fh*fw,) mask: a 1-D per-column/row predicate has to be broadcast
        # back over the grid before the reshape, otherwise (129, fh*fw) * (fh,)
        # fails to broadcast.
        cross = 0 if axis == 0 else 1          # axis=0 slides x -> band over y
        edges = np.linspace(0.0, 1.0, args.segments + 1)
        bands = []
        for bi in range(args.segments):
            lo, hi = edges[bi], edges[bi + 1]
            if cross == 0:
                col = (gx1 >= lo * fw) & (gx1 < hi * fw)
                sel = np.broadcast_to(col[None, :], (fh, fw))
            else:
                row = (gy1 >= lo * fh) & (gy1 < hi * fh)
                sel = np.broadcast_to(row[:, None], (fh, fw))
            sel = np.ascontiguousarray(sel).reshape(-1).astype(np.float32)
            bands.append(((m * sel) @ flat) / max(float(sel.sum()), 1.0))
        parts.append(np.repeat(np.concatenate(bands, 0)[None, :], len(win), 0))
    feat = np.concatenate(parts, 1)

    # teacher point -> target window centred on it along the sliding axis
    px, py = row['tx'], row['ty']
    w, h = win[0][2] - win[0][0], win[0][3] - win[0][1]
    if args.target == 'gauss':
        # same soft preference target as the V8 KD-v2 arm (sigma = 1/16 of the
        # sliding span): a jittery teacher point shifts a Gaussian slightly, but
        # makes a hard argmax-IoU target jump to a neighbouring candidate.
        if axis == 0:
            ot = float(np.clip((px * W - w / 2) / span if span > 0 else 0.0, 0, 1))
        elif axis == 1:
            ot = float(np.clip((py * H - h / 2) / span if span > 0 else 0.0, 0, 1))
        else:
            return feat.astype(np.float16), np.ones(len(win), dtype=np.float32)
        offs_n = np.arange(len(win), dtype=np.float32) / max(len(win) - 1, 1)
        u = np.exp(-0.5 * ((offs_n - ot) / args.kd_sigma) ** 2).astype(np.float32)
    else:
        if axis == 0:
            x1 = min(max(px * W - w / 2, 0.0), W - w)
            gt = [x1, 0.0, x1 + w, h]
        elif axis == 1:
            y1 = min(max(py * H - h / 2, 0.0), H - h)
            gt = [0.0, y1, w, y1 + h]
        else:
            gt = [0.0, 0.0, W, H]
        u = iou(win[:, None, :], np.array(gt, dtype=np.float32)[None]).mean(1)
    return feat.astype(np.float16), u.astype(np.float32)


def main():
    rows = [json.loads(l) for l in (args.pool / args.index_name).read_text().splitlines() if l.strip()]
    # attach teacher points
    kept, n_nopoint, n_parsefail = [], 0, 0
    for r in rows:
        p = args.points / f"{r['video_id']}.json"
        if not p.exists():
            n_nopoint += 1
            continue
        rec = json.loads(p.read_text())
        t = rec['ratios']['t']
        pts, st = t['points'], t['status']
        rw, rh = t['ratio']
        assert [rw, rh] == r['targetRatioWH'], r['video_id']
        kfs = rec['keyframes']
        for kf, pt, s in zip(kfs, pts, st):
            if s != 'ok' or pt[2]:
                n_parsefail += 1
                continue
            f = args.feat_root / r['video_id'] / f'{kf}.npz'
            if not f.exists():
                continue
            kept.append({'vid': r['video_id'], 'kf': kf, 'W': r['W'], 'H': r['H'],
                         'rw': rw, 'rh': rh, 'tx': pt[0], 'ty': pt[1],
                         'src': r['src'], 'cls': geom_class(r),
                         'rot': bool(r.get('rotated'))})
    if args.smooth > 0:
        # Smooth per fragment, in keyframe order, over the RAW points, then let
        # build() derive the target. Edge frames use the available window.
        byfrag = defaultdict(list)
        for row in kept:
            byfrag[row['vid']].append(row)
        w = 2 * args.smooth + 1
        for vid, rs in byfrag.items():
            rs.sort(key=lambda r: float(r['kf']))
            px = np.array([r['tx'] for r in rs], dtype=np.float32)
            py = np.array([r['ty'] for r in rs], dtype=np.float32)
            k = np.ones(w, dtype=np.float32) / w
            sx = np.convolve(np.pad(px, (args.smooth, args.smooth), mode="edge"), k, "valid")
            sy = np.convolve(np.pad(py, (args.smooth, args.smooth), mode="edge"), k, "valid")
            for r, a_, b_ in zip(rs, sx, sy):
                r['tx'] = float(np.clip(a_, 0.0, 1.0))
                r['ty'] = float(np.clip(b_, 0.0, 1.0))
    print(f'ROWS fragments={len(rows)} no_points_yet={n_nopoint} '
          f'teacher_parse_fail={n_parsefail} usable_frames={len(kept)}', flush=True)
    if not kept:
        raise SystemExit('no rows with teacher points yet')

    # split by SOURCE VIDEO, stratified by geometry class
    rng = np.random.default_rng(args.seed)
    by_src = defaultdict(list)
    for r in kept:
        by_src[r['src']].append(r)
    srcs = sorted(by_src)
    cls_of = {s: by_src[s][0]['cls'] for s in srcs}
    vsf = args.val_sources or (args.pool / 'val_sources.json')
    if vsf.exists():
        val_src = set(json.loads(vsf.read_text()))
    else:
        val_src = set()
        for c in sorted(set(cls_of.values())):
            pool_c = [s for s in srcs if cls_of[s] == c]
            rng.shuffle(pool_c)
            val_src.update(pool_c[:max(1, int(round(len(pool_c) * args.val_frac)))])
        vsf.write_text(json.dumps(sorted(val_src), indent=0))
        print(f'WROTE val source list -> {vsf} ({len(val_src)} sources)', flush=True)
    val_src = {s for s in val_src if s in by_src}

    groups = {'phd2_val': [], 'phd2_train': []}
    for s in srcs:
        groups['phd2_val' if s in val_src else 'phd2_train'].extend(by_src[s])
    for g in groups.values():
        g.sort(key=lambda r: (r['vid'], r['kf']))

    cc = defaultdict(lambda: defaultdict(int))
    for r in kept:
        cc[r['cls']]['val' if r['src'] in val_src else 'train'] += 1
    print('STRATA', {k: dict(v) for k, v in sorted(cc.items())}, flush=True)
    for k, v in groups.items():
        rc = defaultdict(int)
        for r in v:
            rc[f"{r['rw']}:{r['rh']}"] += 1
        print(f'{k}: {len(v)} frames, {len({r["src"] for r in v})} sources, '
              f'target mix {dict(rc)}', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tag, rs in groups.items():
        if not rs:
            continue
        if (args.output_dir / f'{tag}_feat.npy').exists():
            print(tag, 'cache exists, skip', flush=True)
            continue
        first = build(rs[0])
        D = first[0].shape[-1]
        fm = np.lib.format.open_memmap(args.output_dir / f'{tag}_feat.npy', mode='w+',
                                       dtype=np.float16, shape=(len(rs), first[0].shape[0], D))
        um = np.lib.format.open_memmap(args.output_dir / f'{tag}_u.npy', mode='w+',
                                       dtype=np.float32, shape=(len(rs), first[0].shape[0]))
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for s in range(0, len(rs), args.chunk):
                blk = rs[s:s + args.chunk]
                res = list(ex.map(build, blk, chunksize=8))
                fm[s:s + len(res)] = np.stack([x[0] for x in res])
                um[s:s + len(res)] = np.stack([x[1] for x in res])
                done += len(res)
                if (done // args.chunk) % 20 == 0:
                    print(f'{tag} {done}/{len(rs)}', flush=True)
        meta = [{'vid': r['vid'], 'kf': r['kf'], 'src': r['src'], 'cls': r['cls'],
                 'rot': r['rot'], 'tx': r['tx'], 'ty': r['ty'],
                 'W': r['W'], 'H': r['H'], 'ratio': f'{r["rw"]}:{r["rh"]}'} for r in rs]
        (args.output_dir / f'{tag}_meta.json').write_text(json.dumps(meta))
        print(f'WROTE {tag} n={len(rs)}', flush=True)


if __name__ == '__main__':
    main()