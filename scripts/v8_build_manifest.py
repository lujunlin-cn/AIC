"""V8: build the training/eval manifest for multi-geometry candidate-utility heads.

Coordinate system: frames are resized to long side 640 BEFORE everything, so
LIVE native is 640x360 and rotated RV is 360x640.  All GT boxes are stored in
that same frame coordinate system (ltrb, half-open-consistent floats).

Sources (exposure ledger respected):
  rv_train  001-030 + 621-700 (110)   native + rot90cw views, 2 ratios each
  rv_dev    601-620 (20)              selection set (native + rot)
  rv_diag   031-100 (70)              diagnostic only (native + rot)
  live_*    frozen release train/dev/confirmation (1115/124/227), 9:16 native
Output: one jsonl at --output with one row per (source, frame).
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--release', default='/data/aic/external_datasets/_releases/spatial_crop_v1')
ap.add_argument('--ann', default='/data/aic/experiments_910a/LFM450_EVAL_V1/annotations')
ap.add_argument('--output', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl'))
args = ap.parse_args()

import sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, '/root/AIC/ext_data')
from aicext.release import Release  # noqa: E402

rows = []
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}

# --- RetargetVid: native (640x360) + rotated (360x640) views
rv_map = {}
for p in sorted(Path(args.ann).glob('annotator_1/*.txt')):
    vid, r = p.stem.split('_')
    rv_map.setdefault(vid, []).append(r)
for vid, ratios in sorted(rv_map.items()):
    if vid <= '030' or '621' <= vid <= '700':
        split = 'rv_train'
    elif '601' <= vid <= '620':
        split = 'rv_dev'
    elif '031' <= vid <= '100':
        split = 'rv_diag'
    else:
        continue  # exposed_eval / confirmation sources: not used in V8 training
    for r in sorted(set(ratios)):
        gt = np.maximum(np.stack([np.loadtxt(Path(args.ann) / f'annotator_{i}' / f'{vid}_{r}.txt',
                                             delimiter=',') for i in range(1, 7)]), 0)
        for kf in range(gt.shape[1]):
            b = gt[:, kf]  # (6,4) ltrb in 640x360
            if (b[:, 2] - b[:, 0]).min() <= 0 or (b[:, 3] - b[:, 1]).min() <= 0:
                continue
            rows.append({'src': 'rv_native', 'split': split, 'vid': vid, 'frame': kf, 'ratio': r,
                         'W': 640.0, 'H': 360.0, 'gt': b.tolist()})
            rot = np.stack([360.0 - b[:, 3], b[:, 0], 360.0 - b[:, 1], b[:, 2]], 1)  # 90 cw
            rows.append({'src': 'rv_rot', 'split': split, 'vid': vid, 'frame': kf,
                         'ratio': '3-1' if r == '1-3' else '1-3',
                         'W': 360.0, 'H': 640.0, 'gt': rot.tolist()})

# --- LIVE-YT-VC native: 640x360 frame, 9:16 window on x axis
rel = Release('spatial_crop_v1', path=Path(args.release))
for sp in ('train', 'dev', 'confirmation'):
    for m in rel.manifest(sp):
        vid = m['unit_id'].split(':', 1)[1]
        pk = np.load(Path(args.release) / 'packed' / 'LIVE_YT_VC' / f'{vid}.npz')
        fidx = pk['frame_idx'].astype(int)
        boxes = pk['ltrb_raw'].astype(np.float64)  # (1,30,4) original px
        sx, sy = 640.0 / m['width'], 360.0 / m['height']
        for j, f in enumerate(fidx):
            b = boxes[0, j] * np.array([sx, sy, sx, sy])
            if b[2] - b[0] <= 0 or b[3] - b[1] <= 0:
                continue
            rows.append({'src': f'live_{sp}', 'split': f'live_{sp}', 'vid': vid, 'frame': int(f),
                         'ratio': m['ratio_key'], 'W': 640.0, 'H': 360.0, 'gt': [b.tolist()]})

args.output.parent.mkdir(parents=True, exist_ok=True)
with open(args.output, 'w') as f:
    for row in rows:
        f.write(json.dumps(row) + '\n')
import collections  # noqa: E402
c = collections.Counter((r['src'], r['split']) for r in rows)
print(json.dumps({f'{k[0]}/{k[1]}': v for k, v in sorted(c.items())}, indent=1))
print('total rows:', len(rows))
