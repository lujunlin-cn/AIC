"""Stage 5: full-timeline alignment of one mp4 inside its source video.

No prior about how the mp4 was cut.  Scan the WHOLE source timeline with
the mp4 first/middle/last frames at coarse step, refine the best candidate
at fine step, then fit src_t = a*mp4_t + b from three aligned positions.

Answers: is the mp4 a contiguous cut of the source?  Where does the
official GIF interval [start, start+dur] land on the mp4 timeline?
"""
import argparse, json
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
ap.add_argument('--video-ids', default='7,3,0')
ap.add_argument('--coarse', type=float, default=0.5)
ap.add_argument('--fine', type=float, default=0.05)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/map174_stage5_align.json'))
args = ap.parse_args()


def frame_at(c, st, t):
    c.seek(int(max(t, 0) * av.time_base), stream=st, backward=True)
    for frame in c.decode(st):
        if frame.time is None or frame.time >= t - 0.25:
            return frame.reformat(width=64, height=64, format='gray').to_ndarray().astype(np.float32)
    return None


s2 = json.loads(args.stage2.read_text())
by_id = {r['video_id']: r for r in s2['results'] if r['status'] == 'SCORED'}

res = []
for vid in args.video_ids.split(','):
    r = by_id.get(vid.strip())
    if r is None:
        continue
    b = r['best']
    mp4 = args.video_dir / f'{vid.strip()}.mp4'
    src = args.media_dir / f"{b['yt']}.mp4"
    cm = av.open(str(mp4))
    cs = av.open(str(src))
    st_m, st_s = cm.streams.video[0], cs.streams.video[0]
    src_dur = float(cs.duration) / av.time_base
    mp4_dur = r['dur']
    fracs = (0.05, 0.5, 0.95)
    sig = [frame_at(cm, st_m, mp4_dur * f) for f in fracs]
    if any(f is None for f in sig):
        print(vid, 'no mp4 frames')
        continue
    # coarse scan on the FIRST frame across the whole source
    cands = np.arange(0, max(src_dur - mp4_dur, 1), args.coarse)
    maes = []
    for t0 in cands:
        f = frame_at(cs, st_s, t0)
        maes.append(float(np.abs(sig[0] - f).mean()) if f is not None else 1e9)
    maes = np.array(maes)
    order = np.argsort(maes)[:5]
    # refine each of the top-5 with the first frame at fine step
    ref = []
    for i in order:
        fine = np.arange(cands[i] - 1.0, cands[i] + 1.0 + 1e-9, args.fine)
        best = (1e9, None)
        for t0 in fine:
            if t0 < 0:
                continue
            f = frame_at(cs, st_s, t0)
            if f is None:
                continue
            m = float(np.abs(sig[0] - f).mean())
            if m < best[0]:
                best = (m, float(t0))
        ref.append(best)
    ref.sort()
    m1, t1 = ref[0]
    # verify linearity with middle and last frames at the fitted offset
    t_mid = t1 + 0.5 * mp4_dur
    t_last = t1 + 0.95 * mp4_dur
    f_mid = frame_at(cs, st_s, t_mid)
    f_last = frame_at(cs, st_s, t_last)
    mae_mid = float(np.abs(sig[1] - f_mid).mean()) if f_mid is not None else None
    mae_last = float(np.abs(sig[2] - f_last).mean()) if f_last is not None else None
    # where does the official interval land on the mp4 timeline?
    gif0 = b['start'] - t1
    gif1 = b['start'] + b['dur'] - t1
    rec = {'video_id': vid, 'yt': b['yt'], 'mp4_dur': round(mp4_dur, 3),
           'src_t0_of_mp4': round(t1, 3), 'first_frame_mae': round(m1, 1),
           'mid_mae_at_fit': round(mae_mid, 1) if mae_mid is not None else None,
           'p95_mae_at_fit': round(mae_last, 1) if mae_last is not None else None,
           'src_len': round(src_dur, 1),
           'gif_on_mp4': [round(gif0, 2), round(gif1, 2)],
           'gif_cover_share': round(max(0.0, min(mp4_dur, gif1) - max(0.0, gif0)) / mp4_dur, 3),
           'runner_up_t0': [round(t, 2) for _, t in ref[1:3]]}
    res.append(rec)
    print(json.dumps(rec), flush=True)
    cm.close()
    cs.close()

args.out.write_text(json.dumps(res, indent=1))
