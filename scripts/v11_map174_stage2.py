"""Stage 2: perceptual mapping of the 174 preliminary mp4s to PHD2 intervals.

For each mp4: sign 3 frames (25/50/75 percent) at 64x64 gray.
Candidates: testing.csv rows with |row.duration - mp4.duration| <= tol,
sorted by duration distance, capped, ONLY rows whose source mp4 is on disk.
Match score: mean MAE of the 3 frame pairs (mp4 frac f vs source start+f*dur).

Outputs, per video: verified mapping (best score + runner-up), or
pending_download (candidates exist but no media), or unmatched.

CPU only; PyAV for in-process decoding.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import av

ap = argparse.ArgumentParser()
ap.add_argument('--video-dir', type=Path,
                default=Path('/data/aic/official_test_20260926/extracted/基于视频大模型的通用视频高光剪辑/video'))
ap.add_argument('--testing-csv', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/upstream_repo/testing.csv'))
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--tol', type=float, default=0.15)
ap.add_argument('--max-cands', type=int, default=10)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/map174_stage2.json'))
ap.add_argument('--limit', type=int, default=0, help='debug: only first N mp4s')
args = ap.parse_args()


def sign_frames(path, times):
    """Gray 64x64 frames nearest to each requested second; None on failure."""
    out = []
    try:
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            for t in times:
                c.seek(int(t * av.time_base), stream=st, backward=True)
                got = None
                for frame in c.decode(st):
                    if frame.time is None or frame.time >= t - 0.25:
                        got = frame
                        break
                if got is None:
                    return None
                out.append(got.reformat(width=64, height=64, format='gray').to_ndarray().astype(np.float32))
    except Exception:
        return None
    return out if len(out) == len(times) else None


rows = list(csv.DictReader(args.testing_csv.open()))
for r in rows:
    r['start_f'] = float(r['start'])
    r['dur_f'] = float(r['duration'])
by_dur = sorted(rows, key=lambda r: r['dur_f'])
durs_arr = np.array([r['dur_f'] for r in by_dur])

mp4s = sorted(args.video_dir.glob('*.mp4'), key=lambda p: int(p.stem))
if args.limit:
    mp4s = mp4s[:args.limit]

results, n_verified, n_pending, n_unmatched = [], 0, 0, 0
for i, mp4 in enumerate(mp4s):
    sig = sign_frames(mp4, [0, 0, 0])  # probe readability first
    try:
        with av.open(str(mp4)) as c:
            dur = float(c.duration) / av.time_base
    except Exception:
        dur = None
    if sig is None or dur is None or dur <= 0:
        n_unmatched += 1
        results.append({'video_id': mp4.stem, 'status': 'NO_MEDIA'})
        continue
    sig = sign_frames(mp4, [dur * f for f in (0.25, 0.5, 0.75)])
    if sig is None:
        n_unmatched += 1
        results.append({'video_id': mp4.stem, 'status': 'NO_FRAMES'})
        continue
    lo = np.searchsorted(durs_arr, dur - args.tol)
    hi = np.searchsorted(durs_arr, dur + args.tol)
    cands = sorted(by_dur[lo:hi], key=lambda r: abs(r['dur_f'] - dur))[:args.max_cands]
    scored, pending = [], 0
    for c in cands:
        src = args.media_dir / f"{c['youtubeId']}.mp4"
        if not src.exists():
            pending += 1
            continue
        fr = sign_frames(src, [c['start_f'] + c['dur_f'] * f for f in (0.25, 0.5, 0.75)])
        if fr is None:
            continue
        mae = float(np.mean([np.abs(a - b).mean() for a, b in zip(sig, fr)]))
        scored.append({'yt': c['youtubeId'], 'start': c['start_f'], 'dur': c['dur_f'],
                       'user': c['user_id'], 'is_last': c['is_last'], 'mae': round(mae, 2)})
    scored.sort(key=lambda x: x['mae'])
    rec = {'video_id': mp4.stem, 'dur': round(dur, 3), 'n_cands': len(cands),
           'n_media_missing': pending}
    if scored:
        rec['status'] = 'SCORED'
        rec['best'] = scored[0]
        rec['second'] = scored[1] if len(scored) > 1 else None
        n_verified += 1
    elif pending:
        rec['status'] = 'PENDING_DOWNLOAD'
        n_pending += 1
    else:
        rec['status'] = 'UNMATCHED'
        n_unmatched += 1
    results.append(rec)
    if (i + 1) % 20 == 0:
        print(f'{i + 1}/{len(mp4s)} scored={n_verified} pending={n_pending} unmatched={n_unmatched}', flush=True)

out = {'n': len(results), 'scored': n_verified, 'pending_download': n_pending,
       'unmatched': n_unmatched, 'tol': args.tol, 'results': results}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('n', 'scored', 'pending_download', 'unmatched')}))
