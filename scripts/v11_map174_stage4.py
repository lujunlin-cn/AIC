"""Stage 4: precise alignment scan - where does the mp4 start inside the
source video, relative to the mapped interval start?

If offset ~= 0, the mp4 IS the interval (GT would be all-positive frames,
and the 174-GT main line premise collapses).  If the offset is significant,
the GIF covers only a sub-range of the mp4 and the real F gate is
constructible.

For each tight-mapped video: scan candidate mp4 start offsets in
[start - lead, start + lead] on the source timeline; align by matching the
mp4 middle frame against source frames at (cand_start + 0.5*mp4_dur);
report the offset minimizing MAE.
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
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--mae-thresh', type=float, default=90)
ap.add_argument('--lead', type=float, default=8.0)
ap.add_argument('--step', type=float, default=0.2)
ap.add_argument('--limit', type=int, default=8)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/map174_stage4_offsets.json'))
args = ap.parse_args()


def frame_at(c, st, t):
    c.seek(int(t * av.time_base), stream=st, backward=True)
    for frame in c.decode(st):
        if frame.time is None or frame.time >= t - 0.25:
            return frame.reformat(width=64, height=64, format='gray').to_ndarray().astype(np.float32)
    return None


s2 = json.loads(args.stage2.read_text())
tight = [r for r in s2['results']
         if r['status'] == 'SCORED' and r['best']['mae'] < args.mae_thresh][:args.limit]

rows = list(csv.DictReader(open(args.testing_csv))) if False else None
res = []
for r in tight:
    b = r['best']
    mp4 = args.video_dir / f"{r['video_id']}.mp4"
    src = args.media_dir / f"{b['yt']}.mp4"
    if not src.exists():
        continue
    try:
        cm = av.open(str(mp4))
        cs = av.open(str(src))
    except Exception:
        continue
    mp4_dur = r['dur']
    mid_mp4 = frame_at(cm, cm.streams.video[0], mp4_dur * 0.5)
    if mid_mp4 is None:
        continue
    cands = np.arange(b['start'] - args.lead, b['start'] + args.lead + 1e-9, args.step)
    maes = []
    for c0 in cands:
        f = frame_at(cs, cs.streams.video[0], c0 + 0.5 * mp4_dur)
        maes.append(float(np.abs(mid_mp4 - f).mean()) if f is not None else 1e9)
    maes = np.array(maes)
    i = int(maes.argmin())
    cm.close()
    cs.close()
    off = float(cands[i] - b['start'])
    res.append({'video_id': r['video_id'], 'yt': b['yt'], 'interval_start': b['start'],
                'mp4_dur': r['dur'], 'best_offset': round(off, 2),
                'best_mae': round(float(maes[i]), 1),
                'second_mae': round(float(np.partition(maes, 1)[1]), 1) if len(maes) > 1 else None,
                'interpretation': 'mp4==interval' if abs(off) < 0.3 else
                                  ('gif_inside_mp4' if off < -0.3 else 'gif_starts_after_mp4')})
    print(res[-1], flush=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res, indent=1))
print(json.dumps({'n': len(res),
                  'offset_zero': sum(1 for x in res if abs(x['best_offset']) < 0.3),
                  'offset_negative': sum(1 for x in res if x['best_offset'] <= -0.3),
                  'offset_positive': sum(1 for x in res if x['best_offset'] >= 0.3)}))
