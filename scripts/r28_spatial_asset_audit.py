#!/usr/bin/env python3
"""Read-only audit of deployed B3 geometry replay and Qwen checkpoint completeness.

No decoding, GT, training, model inference or candidate packaging is performed.
"""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from aic.max_window_path import geometry, qwen_centres, ema_offsets, to_crops


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def audit_model(root):
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    actual = {}
    shards = []
    for name in sorted(set(index['weight_map'].values())):
        path = root / name
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
        data_bytes = path.stat().st_size - 8 - n
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        for key, v in tensors.items():
            if key in actual:
                raise ValueError('duplicate tensor ' + key)
            if not 0 <= v['data_offsets'][0] <= v['data_offsets'][1] <= data_bytes:
                raise ValueError('invalid offsets ' + key)
            actual[key] = name
        shards.append({'name': name, 'bytes': path.stat().st_size,
                       'sha256': sha(path), 'stored_elements': sum(math.prod(v['shape']) for v in tensors.values()),
                       'dtypes': dict(Counter(v['dtype'] for v in tensors.values()))})
    if actual != index['weight_map']:
        raise ValueError('indexed tensor/shard mapping mismatch')
    return {'path': str(root), 'index_sha256': sha(root / 'model.safetensors.index.json'),
            'indexed_tensors': len(actual), 'mapping_exact': True, 'shards': shards,
            'stored_tensor_elements': sum(x['stored_elements'] for x in shards),
            'parameter_note': 'Stored tensor elements are a checkpoint audit, not a runtime unique-parameter certification. Check tied weights and buffers before submission.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/data/aic/semifinal_20261001'))
    ap.add_argument('--model', type=Path, default=Path('/data/aic/pretrained/Qwen3.5-4B'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    pred = root / 'submissions/LFM_V8_B3_SEMIFINAL/predictions.jsonl'
    axes = Counter()
    sources = Counter()
    errors = []
    maxdiff = 0.
    frames = 0
    vids = 0
    statuses = Counter()
    allframes = 0
    for line in pred.open():
        r = json.loads(line)
        vid = r['video_id']
        with np.load(root / 'obs_cache' / f'{vid}_t.npz') as z:
            W, H = int(z['W']), int(z['H'])
            w, h, axis = geometry(W, H, r['targetRatioWH'])
            axes[str(axis)] += 1
            ref = np.asarray([p['bboxes'] for p in r['predictions']])
            n = len(z['raw'])
            if [p['frame'] for p in r['predictions']] != list(range(n)):
                raise ValueError('frame mask mismatch ' + vid)
            if axis is None:
                replay = np.asarray(to_crops(W, H, r['targetRatioWH'], np.zeros(n)))
                sources['full'] += n
            else:
                q = json.loads((root / 'points_B3/points' / f'{vid}.json').read_text())
                statuses.update(q['ratios']['t']['status'])
                centre, src = qwen_centres(q['ratios']['t']['points'], q['keyframes'], z['reset'], z['raw'][:, axis], axis)
                sources.update(src.tolist())
                replay = np.asarray(to_crops(W, H, r['targetRatioWH'], ema_offsets(centre, z['reset'], W, H, r['targetRatioWH'], alpha=.25)))
            d = float(np.max(np.abs(ref - replay)))
            maxdiff = max(maxdiff, d)
            if d > 1e-9:
                errors.append({'video_id': vid, 'max_abs_diff_px': d})
            frames += n
            allframes += int(axis is None) * n
            vids += 1
    missing = [p for p in ['/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt', '/data/aic/pretrained/lfm2_5_vl_450m'] if not Path(p).exists()]
    report = {'experiment_id': 'R28_SPATIAL_AUDIT', 'videos': vids, 'frames': frames,
              'prediction_sha256': sha(pred), 'axis_video_counts': dict(axes),
              'fullframe_geometry_frames': allframes, 'source_frame_counts': dict(sources),
              'point_status_counts': dict(statuses), 'max_abs_replay_diff_px': maxdiff,
              'replay_mismatches': errors, 'replay_exact': not errors, 'missing_recorded_assets': missing,
              'qwen_checkpoint': audit_model(args.model),
              'scope': 'No official video decoding or GT access. Replay uses cached inference points and observations only. No protected public source access.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
