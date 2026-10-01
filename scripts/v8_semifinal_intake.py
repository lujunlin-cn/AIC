"""Build the semifinal (426-video) intake index in the frozen V7 schema.

The semifinal drop ships only per-video jsonl with targetRatioWH - no frame
count, no fps, no path.  The inference chain (obs cache -> keyframes -> head
-> max_window_release) needs the same index fields the 174-video preliminary
run used, so we probe each container with PyAV and emit
aic.input_index.v1 records with identical semantics:

  coordinate_convention = coded_pixels_no_autorotate   (PyAV 17 vs 15 pixel
      micro-differences are known and stay self-consistent inside this run)
  time_base = 1/90000, pts_verified = true, source_sha256 = sha256 of the
      media file (same field the preliminary index carried).

No ground truth is read: the semifinal jsonl holds targetRatioWH only.
"""
import argparse, hashlib, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--test-root', type=Path,
                default=Path('/data/aic/semifinal_test/复赛-基于视频大模型的通用视频高光剪辑'))
ap.add_argument('--output-dir', type=Path, default=Path('/data/aic/semifinal_20261001/intake'))
ap.add_argument('--workers', type=int, default=16)
args = ap.parse_args()

import av  # noqa: E402


def probe(vid):
    media = args.test_root / 'video' / f'{vid}.mp4'
    meta = json.loads((args.test_root / 'jsonl' / f'{vid}.jsonl').read_text().strip())
    h = hashlib.sha256()
    with open(media, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    with av.open(str(media)) as c:
        st = c.streams.video[0]
        rate = float(st.average_rate) if st.average_rate else 30.0
        n = st.frames
        if not n:
            n = int(round(float(st.duration * st.time_base) * rate)) if st.duration else 0
        w, hgt = st.codec_context.width, st.codec_context.height
        tb = str(st.time_base) if st.time_base else '1/90000'
        rot = 0.0
    return {'video_id': vid, 'video_path': str(media), 'width': w, 'height': hgt,
            'fps': rate, 'frame_count': int(n),
            'duration': round(int(n) / rate, 4) if rate else 0.0,
            'targetRatioWH': meta.get('targetRatioWH'), 'rotation': rot,
            'time_base': tb, 'pts_verified': True, 'has_audio': True,
            'coordinate_convention': 'coded_pixels_no_autorotate',
            'index_schema_version': 'aic.input_index.v1',
            'intake_version': 'AIC_EVAL_INTAKE_V1',
            'source_sha256': h.hexdigest()}


vids = sorted((p.stem for p in (args.test_root / 'video').glob('*.mp4')),
              key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
print(f'{len(vids)} semifinal videos', flush=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
rows = []
with ProcessPoolExecutor(max_workers=args.workers) as ex:
    for i, r in enumerate(ex.map(probe, vids, chunksize=4)):
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(vids)}', flush=True)
rows.sort(key=lambda r: int(r['video_id']) if r['video_id'].isdigit() else r['video_id'])
out = args.output_dir / 'index.enriched.jsonl'
with open(out, 'w') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
print('ratios:', {tuple(r['targetRatioWH'] or ()) for r in rows})
print('total frames:', sum(r['frame_count'] for r in rows))
print('WROTE', out)
