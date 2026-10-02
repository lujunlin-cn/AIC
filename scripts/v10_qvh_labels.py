"""QVHighlights official-release annotations -> 1 s-grid labels for the temporal head.

The official release (highlight_{train,val,test}_release.jsonl) carries two
complementary supervision signals per video, and this script materialises both
on a fixed 1 Hz time grid so a temporal head can be trained on them:

  * saliency_scores  -- a dense ordinal curve.  Shape [n_clips][n_annotators],
    each entry an integer highlight-worthiness score in 0..4.  The per-clip
    annotator MEAN is the regression target used by every QVHighlights model
    (Moment-DETR onward) for the dense saliency objective.  clip i covers the
    time span [i*clip_s, (i+1)*clip_s) where clip_s = duration / n_clips (the
    clip width varies per video; it is NOT a fixed 2 s).

  * relevant_windows -- a sparse list of [t_start, t_end] second-resolution
    highlight intervals (the same supervision as PHD2 GIF intervals).  These
    are converted into the same exp(-d/tau) soft label the PHD2 TCN used, so
    one head can consume both datasets identically.

Both are written per-source-video, NOT per-qid: QVHighlights assigns ~1 query
per video and saliency is query-independent, so a video is the independent
resampling unit for any later source-level bootstrap.

Output: <out>/<vid>.npz with
    t           float32 (n,)   second timestamps of the 1 Hz grid centres
    saliency    float32 (n,)   annotator-mean saliency interpolated to the grid
    soft        float32 (n,)   exp(-d/tau) distance-to-nearest-window soft label
    n_annot     int32   (n,)   annotator count behind each clip's mean (for
                               confidence weighting; low-count clips are noisy)
    meta_json   str            provenance + grid parameters

Nothing here decodes video.  Frame features are produced separately by
v9_phd2_lfm_feats.py once a frames/ tree exists; this script only needs the
annotation files, so it runs on CPU while the teacher job holds the NPU.
"""
import argparse, json, os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--annotations', type=Path, required=True,
                help='a highlight_*_release.jsonl file')
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--grid-s', type=float, default=1.0, help='seconds per grid step')
ap.add_argument('--tau', type=float, default=3.0,
                help='soft-label decay, seconds (matches the PHD2 TCN default)')
ap.add_argument('--saliency-max', type=float, default=4.0,
                help='ordinal top score, used to normalise saliency to 0..1')
args = ap.parse_args()

args.out.mkdir(parents=True, exist_ok=True)


def clip_soft(t, windows, tau):
    """exp(-d/tau) to the nearest [a,b] window; 1.0 inside any window."""
    if not windows:
        return np.zeros_like(t)
    d = np.full_like(t, np.inf)
    for a, b in windows:
        inside = (t >= a) & (t <= b)
        dist = np.where(t < a, a - t, np.where(t > b, t - b, 0.0))
        d = np.minimum(d, dist)
    return np.exp(-d / tau).astype(np.float32)


n_v = 0
for line in args.annotations.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    vid = r['vid']
    dur = float(r['duration'])
    n = int(round(dur / args.grid_s))
    if n <= 0:
        continue
    # grid centred on each second
    t = (np.arange(n, dtype=np.float32) + 0.5) * args.grid_s
    t = np.minimum(t, dur - 1e-3)

    ss = r.get('saliency_scores') or []
    n_clips = len(ss)
    if n_clips:
        clip_s = dur / n_clips
        # per-clip annotator mean and count
        mean = np.array([np.mean(c) if c else 0.0 for c in ss], dtype=np.float32)
        cnt = np.array([len(c) for c in ss], dtype=np.int32)
        # clip i covers [i*clip_s,(i+1)*clip_s): hold its mean across the span
        clip_idx = np.clip((t / clip_s).astype(np.int64), 0, n_clips - 1)
        sal = np.clip(mean[clip_idx] / args.saliency_max, 0.0, 1.0)
        nann = cnt[clip_idx]
    else:
        sal = np.zeros(n, dtype=np.float32)
        nann = np.zeros(n, dtype=np.int32)

    wins = [w for w in (r.get('relevant_windows') or []) if len(w) == 2]
    soft = clip_soft(t, wins, args.tau)

    np.savez_compressed(
        args.out / f'{vid}.npz',
        t=t, saliency=sal, soft=soft, n_annot=nann,
        meta_json=np.asarray(json.dumps({
            'vid': vid, 'duration': dur, 'qid': r.get('qid'),
            'n_clips': n_clips, 'clip_s': dur / n_clips if n_clips else None,
            'n_windows': len(wins), 'grid_s': args.grid_s, 'tau': args.tau,
            'source': args.annotations.name,
        }, sort_keys=True)))
    n_v += 1

print(f'QV_LABELS {args.annotations.name}: {n_v} videos -> {args.out}', flush=True)
