"""Stage 3: widen the duration tolerance for the 45 unmatched mp4s and
emit the deduplicated download list for pending sources.

Reuses the stage-2 logic with tol=0.6 (re-encode can shift duration),
testing.csv only.  Outputs:
  map174_stage3.json  - retry results for unmatched video_ids
  pending_yts.txt     - unique youtubeIds that block pending mappings
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import av

ap = argparse.ArgumentParser()
ap.add_argument('--video-dir', type=Path,
                default=Path('/data/aic/official_test_20260926/extracted/基于视频大模型的通用视频高光剪辑/video'))
ap.add_argument('--stage2', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/map174_stage2.json'))
ap.add_argument('--testing-csv', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/upstream_repo/testing.csv'))
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--tol', type=float, default=0.6)
ap.add_argument('--max-cands', type=int, default=12)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/map174_stage3.json'))
ap.add_argument('--pending-list', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/pending_yts.txt'))
args = ap.parse_args()


def sign_frames(path, times):
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

s2 = json.loads(args.stage2.read_text())
retry_ids = [r['video_id'] for r in s2['results'] if r['status'] == 'UNMATCHED']

# Candidate yts for pending videos: stage2 stored only counts, so re-derive
# from durations (no media access needed).
dur_by_id = {r['video_id']: r['dur'] for r in s2['results']
             if r['status'] == 'PENDING_DOWNLOAD'}
pend_yts = set()
for vid, dur in dur_by_id.items():
    lo = np.searchsorted(durs_arr, dur - 0.15)
    hi = np.searchsorted(durs_arr, dur + 0.15)
    for c in by_dur[lo:hi]:
        pend_yts.add(c['youtubeId'])
args.pending_list.parent.mkdir(parents=True, exist_ok=True)
args.pending_list.write_text('\n'.join(sorted(pend_yts)) + '\n')

results = []
n_scored = 0
for vid in retry_ids:
    mp4 = args.video_dir / f'{vid}.mp4'
    sig = None
    try:
        with av.open(str(mp4)) as c:
            dur = float(c.duration) / av.time_base
    except Exception:
        dur = None
    if dur:
        sig = sign_frames(mp4, [dur * f for f in (0.25, 0.5, 0.75)])
    if sig is None:
        results.append({'video_id': vid, 'status': 'NO_FRAMES'})
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
    rec = {'video_id': vid, 'dur': round(dur, 3), 'n_cands': len(cands),
           'n_media_missing': pending}
    if scored:
        rec['status'] = 'SCORED'
        rec['best'] = scored[0]
        rec['second'] = scored[1] if len(scored) > 1 else None
        n_scored += 1
    elif pending:
        rec['status'] = 'PENDING_DOWNLOAD'
    else:
        rec['status'] = 'UNMATCHED'
    results.append(rec)

out = {'retried': len(retry_ids), 'scored': n_scored,
       'pending_yts': len(pend_yts), 'results': results, 'tol': args.tol}
print(json.dumps({k: out[k] for k in ('retried', 'scored', 'pending_yts')}))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1))
